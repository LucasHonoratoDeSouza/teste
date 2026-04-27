import asyncio
import csv
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from config import (
    BET_COOLDOWN_SECONDS,
    KELLY_FRACTION,
    LOCK_PROFIT_BID,
    LOCK_PROFIT_TIME_TO_EXPIRY_SECONDS,
    MAX_ENTRY_PRICE,
    MAX_SPREAD_PERCENT,
    MAX_OPEN_BETS_PER_MARKET,
    MAX_OPEN_BETS_PER_OUTCOME,
    MAX_OPEN_RISK_PER_MARKET_USD,
    MIN_EDGE,
    MIN_ENTRY_TIME_TO_EXPIRY_SECONDS,
    MIN_HOLD_BEFORE_SELL_SECONDS,
    ONLINE_CALIBRATION_ENABLED,
    ONLINE_CALIBRATION_MIN_ROUNDS,
    ONLINE_CALIBRATION_MIN_SAMPLES,
    ONLINE_CALIBRATION_WINDOW_SAMPLES,
    PAPER_BANKROLL_USD,
    SELL_OVERPRICED_EDGE,
)
from data_engine import DataEngine, utc_now
from features import FeatureEngineer


class ForwardTestEngine:
    def __init__(
        self,
        model,
        initial_market=None,
        duration_seconds=None,
        results_file="results.csv",
        status_file="status.json",
    ):
        self.data_engine = DataEngine()
        self.fe = FeatureEngineer()
        self.model = model
        self.initial_market = initial_market
        self.duration_seconds = duration_seconds
        self.status_file = status_file

        self.active_bets = []
        self.completed_bets = []
        self.last_bet_time = utc_now() - timedelta(days=1)
        self.csv_file = results_file
        self._last_status_print = None
        self._last_monitor_at = utc_now() - timedelta(days=1)
        self._last_status_file_at = utc_now() - timedelta(days=1)
        self.stop_new_entries = False
        self.pending_calibration_samples = {}
        self.live_calibration_samples = []
        self.live_calibration_rounds = set()
        self.live_calibration_generation = 0
        self.last_live_calibration_generation = -1

        self._ensure_csv()

    def _ensure_csv(self):
        if os.path.exists(self.csv_file):
            return
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "round_expiry",
                    "S0",
                    "direction",
                    "time_to_expiry_secs",
                    "P_model",
                    "P_mkt_paid",
                    "EV",
                    "bet_size_USD",
                    "status",
                    "close_price",
                    "PnL_USD",
                ]
            )

    def _write_status_file(self, now, force=False):
        if not self.status_file:
            return
        if not force and (now - self._last_status_file_at).total_seconds() < 10:
            return

        self._last_status_file_at = now
        total_pnl = sum(bet.get("pnl", 0.0) for bet in self.completed_bets)
        status = {
            "timestamp": now.isoformat(),
            "market_slug": self.data_engine.market_slug,
            "round_expiry": self.data_engine.round_expiry.isoformat() if self.data_engine.round_expiry else None,
            "S0": self.data_engine.S0,
            "chainlink_price": self.data_engine.current_btc_price,
            "chainlink_timestamp": (
                self.data_engine.chainlink_timestamp.isoformat()
                if self.data_engine.chainlink_timestamp else None
            ),
            "open_bets": len(self.active_bets),
            "completed_bets": len(self.completed_bets),
            "paper_pnl": total_pnl,
            "stop_new_entries": self.stop_new_entries,
            "live_calibration_samples": len(self.live_calibration_samples),
            "live_calibration_rounds": len(self.live_calibration_rounds),
            "pending_calibration_rounds": len(self.pending_calibration_samples),
            "results_file": self.csv_file,
        }

        tmp_path = f"{self.status_file}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, self.status_file)

    async def run(self):
        print("Iniciando engines: Binance book, Chainlink RTDS, Gamma discovery e CLOB WS...")
        await self.data_engine.warmup()

        tasks = [
            asyncio.create_task(self.data_engine.run_binance_ws()),
            asyncio.create_task(self.data_engine.run_chainlink_rtds()),
            asyncio.create_task(self.data_engine.run_market_discovery(self.initial_market)),
            asyncio.create_task(self.data_engine.run_polymarket_ws()),
            asyncio.create_task(self.data_engine.run_polymarket_book_refresh()),
            asyncio.create_task(self.data_engine.run_history_recorder()),
        ]

        started_at = utc_now()
        print("Forward test real iniciado. O sistema só abre paper trades quando Chainlink e CLOB estão vivos.")
        print(f"Log de paper trades: {self.csv_file}")

        try:
            while True:
                await asyncio.sleep(1)
                now = utc_now()

                if self.duration_seconds and (now - started_at).total_seconds() >= self.duration_seconds:
                    print("Duração configurada atingida; encerrando forward test.")
                    self._write_status_file(now, force=True)
                    return

                self._resolve_live_calibration_samples(now)
                self._resolve_expired_bets(now)
                self._write_status_file(now)

                readiness_issue = self._readiness_issue(now)
                if readiness_issue:
                    self._print_status(now, readiness_issue)
                    continue

                history_df = pd.DataFrame(self.data_engine.history)
                up_quote = self.data_engine.get_quote("UP")
                down_quote = self.data_engine.get_quote("DOWN")

                features = self.fe.extract_realtime_features(
                    current_time=now,
                    btc_price=self.data_engine.current_btc_price,
                    S0=self.data_engine.S0,
                    bids=self.data_engine.order_book["bids"],
                    asks=self.data_engine.order_book["asks"],
                    p_mkt=up_quote.mid,
                    history_df=history_df,
                    round_expiry=self.data_engine.round_expiry,
                )

                if self._spread_too_wide(up_quote, down_quote):
                    self._print_status(
                        now,
                        f"Spread alto. UP={up_quote.spread:.3f}, DOWN={down_quote.spread:.3f}, limite={MAX_SPREAD_PERCENT:.3f}",
                    )
                    continue

                feature_array = np.array(features.values, dtype=np.float32)
                feature_tensor = torch.tensor(feature_array).unsqueeze(0)
                raw_p_up = float(self.model.predict_raw_proba(feature_tensor)[0])
                p_up_model = float(self.model.calibrate_raw_proba([raw_p_up])[0])
                p_up_model = float(np.clip(p_up_model, 0.001, 0.999))
                self._record_live_calibration_sample(now, raw_p_up)

                self._manage_open_positions(now, p_up_model)

                candidates = [
                    {
                        "direction": "UP",
                        "p_model": p_up_model,
                        "cost": up_quote.best_ask,
                        "ev": p_up_model - up_quote.best_ask,
                        "quote": up_quote,
                    },
                    {
                        "direction": "DOWN",
                        "p_model": 1.0 - p_up_model,
                        "cost": down_quote.best_ask,
                        "ev": (1.0 - p_up_model) - down_quote.best_ask,
                        "quote": down_quote,
                    },
                ]
                best = max(candidates, key=lambda item: item["ev"])

                if (now - self.last_bet_time).total_seconds() >= BET_COOLDOWN_SECONDS:
                    opened, block_reason = self._try_open_paper_bet(now, best, features)
                    if not opened:
                        self._print_monitor(now, features, p_up_model, up_quote, down_quote, best, block_reason)
                else:
                    self._print_monitor(now, features, p_up_model, up_quote, down_quote, best)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _readiness_issue(self, now):
        if not self.data_engine.has_fresh_chainlink_price():
            return "Aguardando preço Chainlink RTDS fresco..."
        if self.data_engine.current_btc_price is None:
            return "Aguardando preço BTC..."
        if len(self.data_engine.history) < 300:
            return f"Aguardando histórico preencher ({len(self.data_engine.history)}/300s)..."
        if not self.data_engine.market_is_ready():
            if self.data_engine.market_wait_reason:
                return self.data_engine.market_wait_reason
            return "Aguardando mercado Polymarket atual e books UP/DOWN..."
        if self.data_engine.round_expiry <= now:
            return "Mercado atual expirado; aguardando próxima rodada..."
        if not self.data_engine.accepting_orders:
            return "Mercado encontrado, mas Gamma ainda marca acceptingOrders=false..."

        for outcome in ("UP", "DOWN"):
            quote = self.data_engine.get_quote(outcome)
            if quote.updated_at is None or (now - quote.updated_at).total_seconds() > 20:
                return f"Aguardando book CLOB fresco para {outcome}..."
        return None

    def _spread_too_wide(self, up_quote, down_quote):
        return (
            up_quote.spread is None
            or down_quote.spread is None
            or up_quote.spread > MAX_SPREAD_PERCENT
            or down_quote.spread > MAX_SPREAD_PERCENT
        )

    def _try_open_paper_bet(self, now, candidate, features):
        block_reason = self._entry_block_reason(candidate, features)
        if block_reason:
            return False, block_reason

        self.last_bet_time = now

        min_shares = self.data_engine.order_min_size
        target_stake = self._kelly_stake(candidate["p_model"], candidate["cost"])
        target_stake = min(target_stake, self._remaining_market_risk(candidate["direction"]))
        target_stake = max(target_stake, min_shares * candidate["cost"])

        max_fill_price = min(MAX_ENTRY_PRICE, candidate["p_model"] - MIN_EDGE)
        fill = candidate["quote"].estimate_buy(target_stake, max_price=max_fill_price)
        if fill["shares"] < min_shares:
            return False, "liquidez insuficiente no ask dentro do preço máximo"

        stake = float(fill["cost"])
        shares = float(fill["shares"])
        avg_price = float(fill["avg_price"])
        realized_ev = candidate["p_model"] - avg_price
        if realized_ev <= MIN_EDGE:
            return False, "EV caiu após simular fill pelo book"

        bet = {
            "timestamp": now,
            "market_slug": self.data_engine.market_slug,
            "round_expiry": self.data_engine.round_expiry,
            "S0": self.data_engine.S0,
            "direction": candidate["direction"],
            "token_id": candidate["quote"].token_id,
            "time_to_expiry_secs": float(features["time_to_expiry"]),
            "P_model": candidate["p_model"],
            "P_mkt_paid": avg_price,
            "EV": realized_ev,
            "bet_size_USD": stake,
            "shares": shares,
            "status": "OPEN",
        }
        self.active_bets.append(bet)

        print(
            f"\n[PAPER TRADE ABERTO] {now.strftime('%H:%M:%S')} UTC | "
            f"{bet['direction']} | cost ask={bet['P_mkt_paid']:.3f} | "
            f"model={bet['P_model']:.2%} | EV={bet['EV']:.2%} | stake=${stake:.2f} | "
            f"shares={shares:.2f} | slug={bet['market_slug']}"
        )
        return True, None

    def _entry_block_reason(self, candidate, features):
        if candidate["ev"] <= MIN_EDGE:
            return f"sem edge suficiente ({candidate['ev']:.2%})"
        if features["time_to_expiry"] < MIN_ENTRY_TIME_TO_EXPIRY_SECONDS:
            return f"muito perto do vencimento ({features['time_to_expiry']:.0f}s)"
        if candidate["cost"] >= MAX_ENTRY_PRICE:
            return f"entrada cara demais ({candidate['cost']:.3f})"
        if candidate["quote"].best_ask_size < self.data_engine.order_min_size:
            return "top ask menor que tamanho mínimo"
        if self._open_count_for_market() >= MAX_OPEN_BETS_PER_MARKET:
            return "limite de posições abertas na rodada"
        if self._open_count_for_outcome(candidate["direction"]) >= MAX_OPEN_BETS_PER_OUTCOME:
            return f"já existe posição aberta em {candidate['direction']}"
        min_stake = self.data_engine.order_min_size * candidate["cost"]
        if self._remaining_market_risk(candidate["direction"]) < min_stake:
            return "limite de risco aberto na rodada"
        return None

    def _kelly_stake(self, p_win, cost):
        if cost <= 0 or cost >= 1:
            return 0.0
        full_kelly = (p_win - cost) / (1.0 - cost)
        fractional_kelly = max(0.0, min(0.20, full_kelly * KELLY_FRACTION))
        return PAPER_BANKROLL_USD * fractional_kelly

    def _open_count_for_market(self):
        return len([bet for bet in self.active_bets if bet["market_slug"] == self.data_engine.market_slug])

    def _open_count_for_outcome(self, direction):
        return len([
            bet for bet in self.active_bets
            if bet["market_slug"] == self.data_engine.market_slug and bet["direction"] == direction
        ])

    def _open_risk_for_market(self):
        return sum(
            bet["bet_size_USD"]
            for bet in self.active_bets
            if bet["market_slug"] == self.data_engine.market_slug
        )

    def _remaining_market_risk(self, direction):
        if self._open_count_for_outcome(direction) >= MAX_OPEN_BETS_PER_OUTCOME:
            return 0.0
        return max(0.0, MAX_OPEN_RISK_PER_MARKET_USD - self._open_risk_for_market())

    def _manage_open_positions(self, now, p_up_model):
        bets_to_remove = []

        for bet in self.active_bets:
            if bet["market_slug"] != self.data_engine.market_slug:
                continue

            quote = self.data_engine.get_quote(bet["direction"])
            if quote.best_bid is None:
                continue

            held_model_prob = p_up_model if bet["direction"] == "UP" else 1.0 - p_up_model
            sell_fill = quote.estimate_sell(bet["shares"])
            if not sell_fill["filled"]:
                continue

            exit_bid = float(sell_fill["avg_price"])
            pnl = float(sell_fill["proceeds"]) - bet["bet_size_USD"]
            seconds_left = max(0.0, (bet["round_expiry"] - now).total_seconds())
            held_seconds = (now - bet["timestamp"]).total_seconds()
            reason = self._early_exit_reason(bet, held_model_prob, exit_bid, pnl, seconds_left, held_seconds)
            if reason is None:
                continue

            bet["status"] = reason
            bet["close_price"] = self.data_engine.current_btc_price
            bet["pnl"] = pnl
            bet["exit_price"] = exit_bid
            bet["exit_timestamp"] = now

            self.completed_bets.append(bet)
            bets_to_remove.append(bet)
            self._append_result_csv(bet)

            print(
                f"\n[PAPER TRADE VENDIDO] {now.strftime('%H:%M:%S')} UTC | "
                f"{bet['direction']} | bid médio={exit_bid:.3f} | pago={bet['P_mkt_paid']:.3f} | "
                f"model={held_model_prob:.2%} | {reason} | PnL=${pnl:.2f}"
            )

        for bet in bets_to_remove:
            self.active_bets.remove(bet)

        if bets_to_remove:
            self._print_performance_report()

    def _early_exit_reason(self, bet, held_model_prob, exit_bid, pnl, seconds_left, held_seconds):
        if seconds_left <= LOCK_PROFIT_TIME_TO_EXPIRY_SECONDS and exit_bid >= LOCK_PROFIT_BID and pnl > 0:
            return "SOLD_LOCK_PROFIT"
        if held_seconds < MIN_HOLD_BEFORE_SELL_SECONDS:
            return None
        if exit_bid >= held_model_prob + SELL_OVERPRICED_EDGE:
            return "SOLD_OVERPRICED"
        return None

    def _resolve_expired_bets(self, now):
        bets_to_remove = []

        for bet in self.active_bets:
            if now < bet["round_expiry"]:
                continue

            if (
                self.data_engine.chainlink_timestamp is not None
                and self.data_engine.chainlink_timestamp < bet["round_expiry"]
                and now < bet["round_expiry"] + timedelta(seconds=10)
            ):
                continue

            outcome_price = self._resolution_price_for_expiry(bet["round_expiry"])
            if outcome_price is None:
                continue

            outcome_up = outcome_price >= bet["S0"]
            won = (bet["direction"] == "UP" and outcome_up) or (bet["direction"] == "DOWN" and not outcome_up)
            pnl = (bet["shares"] * 1.0 - bet["bet_size_USD"]) if won else -bet["bet_size_USD"]

            bet["status"] = "WON" if won else "LOST"
            bet["close_price"] = outcome_price
            bet["pnl"] = pnl

            self.completed_bets.append(bet)
            bets_to_remove.append(bet)
            self._append_result_csv(bet)

            print(
                f"\n[RODADA ENCERRADA] {bet['direction']} | S0={bet['S0']:.4f} | "
                f"final Chainlink={outcome_price:.4f} | {bet['status']} | PnL=${pnl:.2f}"
            )

        for bet in bets_to_remove:
            self.active_bets.remove(bet)

        if bets_to_remove:
            self._print_performance_report()

    def _record_live_calibration_sample(self, now, raw_p_up):
        if not ONLINE_CALIBRATION_ENABLED:
            return
        if self.data_engine.market_slug is None or self.data_engine.round_expiry is None or self.data_engine.S0 is None:
            return

        slug = self.data_engine.market_slug
        self.pending_calibration_samples.setdefault(slug, {
            "round_expiry": self.data_engine.round_expiry,
            "S0": self.data_engine.S0,
            "samples": [],
        })
        bucket = self.pending_calibration_samples[slug]
        bucket["samples"].append({
            "timestamp": now,
            "raw_p_up": raw_p_up,
        })

    def _resolve_live_calibration_samples(self, now):
        if not ONLINE_CALIBRATION_ENABLED or not self.pending_calibration_samples:
            return

        resolved_slugs = []
        for slug, bucket in self.pending_calibration_samples.items():
            round_expiry = bucket["round_expiry"]
            if now < round_expiry + timedelta(seconds=3):
                continue
            if self.data_engine.chainlink_timestamp is None or self.data_engine.chainlink_timestamp < round_expiry:
                continue
            if self.data_engine.current_btc_price is None:
                continue

            outcome_price = self._resolution_price_for_expiry(round_expiry)
            if outcome_price is None:
                continue

            label = 1.0 if outcome_price >= bucket["S0"] else 0.0
            for sample in bucket["samples"]:
                self.live_calibration_samples.append({
                    "raw_p_up": sample["raw_p_up"],
                    "label": label,
                    "round": slug,
                })
            self.live_calibration_rounds.add(slug)
            resolved_slugs.append(slug)

        for slug in resolved_slugs:
            del self.pending_calibration_samples[slug]

        if resolved_slugs:
            self.live_calibration_generation += 1
            self._trim_live_calibration_window()
            self._maybe_update_live_calibrator()

    def _trim_live_calibration_window(self):
        if len(self.live_calibration_samples) <= ONLINE_CALIBRATION_WINDOW_SAMPLES:
            return

        self.live_calibration_samples = self.live_calibration_samples[-ONLINE_CALIBRATION_WINDOW_SAMPLES:]
        self.live_calibration_rounds = {sample["round"] for sample in self.live_calibration_samples}

    def _maybe_update_live_calibrator(self):
        sample_count = len(self.live_calibration_samples)
        round_count = len(self.live_calibration_rounds)
        if sample_count < ONLINE_CALIBRATION_MIN_SAMPLES:
            return
        if round_count < ONLINE_CALIBRATION_MIN_ROUNDS:
            return
        if self.live_calibration_generation == self.last_live_calibration_generation:
            return

        raw_preds = np.array([sample["raw_p_up"] for sample in self.live_calibration_samples], dtype=np.float64)
        labels = np.array([sample["label"] for sample in self.live_calibration_samples], dtype=np.float64)
        if len(np.unique(labels)) < 2:
            print(
                f"[CALIBRAÇÃO LIVE] Aguardando outcomes dos dois lados "
                f"({round_count} rounds, {sample_count} amostras)."
            )
            return

        self.model.fit_calibrator_from_raw(raw_preds, labels)
        self.last_live_calibration_generation = self.live_calibration_generation
        brier = float(np.mean((self.model.calibrate_raw_proba(raw_preds) - labels) ** 2))
        print(
            f"[CALIBRAÇÃO LIVE] Calibrador atualizado com {sample_count} amostras "
            f"de {round_count} rounds. Brier={brier:.4f}"
        )

    def _resolution_price_for_expiry(self, round_expiry):
        for row in sorted(self.data_engine.chainlink_history, key=lambda item: item["timestamp"]):
            if row["timestamp"] >= round_expiry:
                return row["price"]
        if self.data_engine.chainlink_timestamp and self.data_engine.chainlink_timestamp >= round_expiry:
            return self.data_engine.current_btc_price
        return None

    def _append_result_csv(self, bet):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    bet["timestamp"].isoformat(),
                    bet["round_expiry"].isoformat(),
                    bet["S0"],
                    bet["direction"],
                    bet["time_to_expiry_secs"],
                    bet["P_model"],
                    bet["P_mkt_paid"],
                    bet["EV"],
                    bet["bet_size_USD"],
                    bet["status"],
                    bet["close_price"],
                    bet["pnl"],
                ]
            )

    def _print_status(self, now, message):
        if self._last_status_print == message and (now - self._last_monitor_at).total_seconds() < 5:
            return
        self._last_status_print = message
        self._last_monitor_at = now
        print(f"[{now.strftime('%H:%M:%S')}] {message}")

    def _print_monitor(self, now, features, p_up_model, up_quote, down_quote, best, block_reason=None):
        if (now - self._last_monitor_at).total_seconds() < 5:
            return
        self._last_monitor_at = now
        block_suffix = f" | Block: {block_reason}" if block_reason else ""
        print(
            f"[{now.strftime('%H:%M:%S')}] T={features['time_to_expiry']:.0f}s | "
            f"Chainlink={self.data_engine.current_btc_price:.2f} vs S0={self.data_engine.S0:.2f} | "
            f"Model UP={p_up_model:.2%} | UP ask={up_quote.best_ask:.3f} DOWN ask={down_quote.best_ask:.3f} | "
            f"Best={best['direction']} EV={best['ev']:.2%}{block_suffix}"
        )

    def _print_performance_report(self):
        total_bets = len(self.completed_bets)
        if total_bets == 0:
            return

        won_bets = len([b for b in self.completed_bets if b["pnl"] > 0])
        win_rate = won_bets / total_bets
        total_pnl = sum(b["pnl"] for b in self.completed_bets)
        avg_ev = np.mean([b["EV"] for b in self.completed_bets])

        print("\n" + "=" * 55)
        print(" RELATÓRIO DE PERFORMANCE PAPER (AO VIVO)")
        print("=" * 55)
        print(f"Total de trades concluídos : {total_bets}")
        print(f"Trades vencedores          : {won_bets}")
        print(f"Trades perdedores          : {total_bets - won_bets}")
        print(f"Win rate                   : {win_rate:.2%}")
        print(f"EV médio de entrada        : {avg_ev:.2%}")
        print("-" * 55)
        print(f"PnL total paper            : ${total_pnl:.2f}")
        print(f"Trades abertos             : {len(self.active_bets)}")
        print("=" * 55 + "\n")
