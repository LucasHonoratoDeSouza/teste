import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import websockets
from websockets.exceptions import ConnectionClosed

from config import (
    BINANCE_WS_URL,
    BOOK_REFRESH_SECONDS,
    CHAINLINK_SYMBOL,
    POLYMARKET_CLOB_URL,
    POLYMARKET_GAMMA_URL,
    POLYMARKET_MARKET_WS_URL,
    POLYMARKET_RECURRENCE_SECONDS,
    POLYMARKET_RTDS_WS_URL,
    POLYMARKET_SERIES_SLUG,
    REQUEST_TIMEOUT,
)


@dataclass
class OutcomeQuote:
    token_id: str = ""
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bids: list[tuple[float, float]] = None
    asks: list[tuple[float, float]] = None
    last_trade_price: Optional[float] = None
    tick_size: Optional[float] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        self.bids = self.bids or []
        self.asks = self.asks or []
        if self.best_bid is None and self.bids:
            self.best_bid = max(price for price, _ in self.bids)
        if self.best_ask is None and self.asks:
            self.best_ask = min(price for price, _ in self.asks)

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def is_ready(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None

    @property
    def best_bid_size(self) -> float:
        if self.best_bid is None:
            return 0.0
        return self._size_at_price(self.bids, self.best_bid)

    @property
    def best_ask_size(self) -> float:
        if self.best_ask is None:
            return 0.0
        return self._size_at_price(self.asks, self.best_ask)

    @property
    def has_depth(self) -> bool:
        return bool(self.bids and self.asks)

    def estimate_buy(self, budget_usd: float, max_price: float = 1.0) -> dict[str, float | bool]:
        spent = 0.0
        shares = 0.0
        last_price = 0.0

        for price, size in sorted(self.asks):
            if price > max_price or budget_usd <= spent:
                break
            level_budget = price * size
            take_budget = min(level_budget, budget_usd - spent)
            take_shares = take_budget / price
            spent += take_budget
            shares += take_shares
            last_price = price

        return {
            "filled": spent >= budget_usd - 1e-9,
            "cost": spent,
            "shares": shares,
            "avg_price": spent / shares if shares else 0.0,
            "last_price": last_price,
        }

    def estimate_sell(self, shares_to_sell: float, min_price: float = 0.0) -> dict[str, float | bool]:
        proceeds = 0.0
        shares = 0.0
        last_price = 0.0

        for price, size in sorted(self.bids, reverse=True):
            if price < min_price or shares_to_sell <= shares:
                break
            take_shares = min(size, shares_to_sell - shares)
            proceeds += take_shares * price
            shares += take_shares
            last_price = price

        return {
            "filled": shares >= shares_to_sell - 1e-9,
            "proceeds": proceeds,
            "shares": shares,
            "avg_price": proceeds / shares if shares else 0.0,
            "last_price": last_price,
        }

    def _size_at_price(self, levels: list[tuple[float, float]], target_price: float) -> float:
        return sum(size for price, size in levels if abs(price - target_price) < 1e-9)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


class DataEngine:
    def __init__(self):
        self.session = requests.Session()

        self.current_btc_price: Optional[float] = None
        self.chainlink_price: Optional[float] = None
        self.chainlink_timestamp: Optional[datetime] = None
        self.chainlink_history: list[dict[str, Any]] = []
        self.binance_price: Optional[float] = None
        self.price_source = "none"

        self.order_book = {"bids": {}, "asks": {}}
        self.history: list[dict[str, Any]] = []

        self.market_slug: Optional[str] = None
        self.market_title: Optional[str] = None
        self.condition_id: Optional[str] = None
        self.event_start: Optional[datetime] = None
        self.round_expiry: Optional[datetime] = None
        self.S0: Optional[float] = None
        self.accepting_orders = False
        self.order_min_size = 5.0
        self.order_tick_size = 0.01
        self.market_resolved = False

        self.outcome_tokens: dict[str, str] = {}
        self.quotes = {"UP": OutcomeQuote(), "DOWN": OutcomeQuote()}

        # Backwards-compatible fields used by older forward-test code.
        self.polymarket_token_id: Optional[str] = None
        self.polymarket_price = 0.50
        self.polymarket_spread = 1.0

        self._market_version = 0
        self._last_market_error: Optional[str] = None
        self._rtds_connected_once = False
        self.market_wait_reason: Optional[str] = None

    async def warmup(self):
        print("Realizando warmup com dados históricos Binance para preencher features iniciais...")
        try:
            data = self._request_json(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1s", "limit": 350},
            )
            history_data = []
            for kline in data:
                ts = datetime.fromtimestamp(kline[0] / 1000.0, tz=timezone.utc)
                history_data.append({"timestamp": ts, "price": float(kline[4])})

            self.history = history_data
            if data:
                self.binance_price = float(data[-1][4])
                if self.current_btc_price is None:
                    self.current_btc_price = self.binance_price
                    self.price_source = "binance_warmup"

            print(f"Warmup concluído: {len(self.history)} pontos. Aguardando Chainlink RTDS para operar.")
        except Exception as exc:
            print(f"Erro no warmup Binance: {exc}")

    def _request_json(self, url: str, params: Optional[dict[str, Any]] = None) -> Any:
        response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    async def run_binance_ws(self):
        while True:
            try:
                async for ws in websockets.connect(BINANCE_WS_URL, ping_interval=20, ping_timeout=20):
                    try:
                        async for message in ws:
                            data = json.loads(message)

                            if data.get("e") == "trade":
                                self.binance_price = float(data["p"])
                                if self.chainlink_price is None:
                                    self.current_btc_price = self.binance_price
                                    self.price_source = "binance_fallback"
                            elif "lastUpdateId" in data:
                                self.order_book["bids"] = {float(p): float(v) for p, v in data.get("bids", [])}
                                self.order_book["asks"] = {float(p): float(v) for p, v in data.get("asks", [])}
                    except ConnectionClosed:
                        print("Binance WS fechado, reconectando...")
                        continue
            except Exception as exc:
                print(f"Binance WS erro: {exc}")
                await asyncio.sleep(2)

    async def run_chainlink_rtds(self):
        """Stream de preço Chainlink usado pela própria regra do mercado BTC Up/Down."""
        subscription = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": CHAINLINK_SYMBOL}),
                }
            ],
        }

        while True:
            try:
                async with websockets.connect(POLYMARKET_RTDS_WS_URL, ping_interval=None) as ws:
                    await ws.send(json.dumps(subscription))
                    heartbeat = asyncio.create_task(self._send_text_heartbeat(ws, "PING", 5))
                    if not self._rtds_connected_once:
                        print(f"RTDS Chainlink conectado: {CHAINLINK_SYMBOL}")
                        self._rtds_connected_once = True
                    try:
                        while True:
                            try:
                                message = await asyncio.wait_for(ws.recv(), timeout=10)
                            except asyncio.TimeoutError:
                                raise TimeoutError("RTDS sem updates; renovando assinatura para buscar snapshot fresco")
                            if message in ("PING", "PONG"):
                                continue
                            self._handle_rtds_message(message)
                    finally:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat
            except Exception as exc:
                if not isinstance(exc, TimeoutError):
                    print(f"RTDS Chainlink desconectado: {exc}. Reconectando...")
                await asyncio.sleep(2)

    async def _send_text_heartbeat(self, ws, payload: str, interval_seconds: int):
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await ws.send(payload)
            except Exception:
                return

    def _handle_rtds_message(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        payload = data.get("payload")
        if not isinstance(payload, dict):
            return

        snapshot = payload.get("data")
        if isinstance(snapshot, list):
            for point in snapshot:
                if isinstance(point, dict):
                    self._record_chainlink_price(point.get("value"), point.get("timestamp"))
            return

        symbol = str(payload.get("symbol", "")).lower()
        if symbol != CHAINLINK_SYMBOL.lower():
            return

        self._record_chainlink_price(payload.get("value"), payload.get("timestamp") or data.get("timestamp"))

    def _record_chainlink_price(self, value: Any, ts_ms: Any):
        price = safe_float(value)
        if price is None or ts_ms is None:
            return

        timestamp = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)

        self.chainlink_price = price
        self.chainlink_timestamp = timestamp
        self.current_btc_price = price
        self.price_source = "chainlink_rtds"
        self.chainlink_history.append({"timestamp": timestamp, "price": price})

        cutoff = utc_now() - timedelta(minutes=10)
        self.chainlink_history = [row for row in self.chainlink_history if row["timestamp"] >= cutoff]

    def has_fresh_chainlink_price(self, max_age_seconds: int = 15) -> bool:
        if self.chainlink_price is None or self.chainlink_timestamp is None:
            return False
        return (utc_now() - self.chainlink_timestamp).total_seconds() <= max_age_seconds

    async def run_market_discovery(self, initial_market: Optional[str] = None):
        loop = asyncio.get_running_loop()
        initial_slug = self.extract_market_slug(initial_market) if initial_market else None
        loaded_initial = False

        while True:
            now = utc_now()
            slug = initial_slug if initial_slug and not loaded_initial else self.current_round_slug(now)

            needs_load = slug != self.market_slug or not self.outcome_tokens
            if self.round_expiry and now >= self.round_expiry + timedelta(seconds=2):
                needs_load = True

            if needs_load:
                loaded = await loop.run_in_executor(None, self.load_market_by_slug, slug)
                loaded_initial = loaded_initial or bool(initial_slug and loaded)

            await asyncio.sleep(2)

    def current_round_slug(self, now: Optional[datetime] = None) -> str:
        now = now or utc_now()
        epoch = int(now.timestamp())
        round_start = epoch - (epoch % POLYMARKET_RECURRENCE_SECONDS)
        return f"{POLYMARKET_SERIES_SLUG}-{round_start}"

    def extract_market_slug(self, user_input: str) -> str:
        raw = user_input.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            path_parts = [part for part in urlparse(raw).path.split("/") if part]
            if "event" in path_parts:
                idx = path_parts.index("event")
                if idx + 1 < len(path_parts):
                    return path_parts[idx + 1]
            if path_parts:
                return path_parts[-1]
        if "event/" in raw:
            return raw.split("event/", 1)[1].split("?", 1)[0].split("/", 1)[0]
        if "events/" in raw:
            return raw.split("events/", 1)[1].split("?", 1)[0].split("/", 1)[0]
        return raw.split("?", 1)[0].strip("/")

    def set_polymarket_id(self, user_input: str) -> bool:
        slug = self.extract_market_slug(user_input)
        return self.load_market_by_slug(slug)

    def load_market_by_slug(self, slug: str) -> bool:
        try:
            market_data = self._fetch_market_by_slug(slug)
            if not market_data:
                self._print_market_error_once(f"Mercado não encontrado: {slug}")
                return False

            event_data = self._first_event(market_data)
            outcomes = [str(item).upper() for item in parse_json_list(market_data.get("outcomes"))]
            token_ids = [str(item) for item in parse_json_list(market_data.get("clobTokenIds"))]

            if len(outcomes) != len(token_ids) or not token_ids:
                self._print_market_error_once(f"Mercado sem clobTokenIds válidos: {slug}")
                return False

            token_map = dict(zip(outcomes, token_ids))
            if "UP" not in token_map or "DOWN" not in token_map:
                self._print_market_error_once(f"Mercado não parece ser Up/Down: {slug} outcomes={outcomes}")
                return False

            event_start = parse_datetime(market_data.get("eventStartTime") or event_data.get("startTime"))
            round_expiry = parse_datetime(market_data.get("endDate") or event_data.get("endDate"))
            price_to_beat = self._extract_price_to_beat(event_data, market_data, event_start)

            if price_to_beat is None or round_expiry is None:
                self._set_market_wait_reason(slug, event_start, round_expiry)
                return False

            old_tokens = self.outcome_tokens.copy()
            new_quotes = {
                "UP": OutcomeQuote(token_id=token_map["UP"]),
                "DOWN": OutcomeQuote(token_id=token_map["DOWN"]),
            }

            self.market_slug = slug
            self.market_title = market_data.get("question") or event_data.get("title") or slug
            self.condition_id = market_data.get("conditionId")
            self.event_start = event_start
            self.round_expiry = round_expiry
            self.S0 = price_to_beat
            self.accepting_orders = bool(market_data.get("acceptingOrders"))
            self.order_min_size = safe_float(market_data.get("orderMinSize")) or 5.0
            self.order_tick_size = safe_float(market_data.get("orderPriceMinTickSize")) or 0.01
            self.market_resolved = False
            self.outcome_tokens = token_map
            self.quotes = new_quotes
            self.polymarket_token_id = token_map["UP"]
            self._last_market_error = None
            self.market_wait_reason = None

            self.refresh_polymarket_books()
            self._update_compat_market_fields()

            if old_tokens != self.outcome_tokens:
                self._market_version += 1
                print("\n" + "*" * 72)
                print(f"Mercado Polymarket conectado: {self.market_title}")
                print(f"Slug: {self.market_slug}")
                print(f"Alvo Chainlink priceToBeat: {self.S0:.4f} | Expira UTC: {self.round_expiry.strftime('%H:%M:%S')}")
                print(f"UP token: {self.outcome_tokens['UP'][:12]}... | DOWN token: {self.outcome_tokens['DOWN'][:12]}...")
                print("*" * 72 + "\n")

            return True
        except Exception as exc:
            self._print_market_error_once(f"Erro ao carregar mercado {slug}: {exc}")
            return False

    def _set_market_wait_reason(
        self,
        slug: str,
        event_start: Optional[datetime],
        round_expiry: Optional[datetime],
    ):
        now = utc_now()
        if event_start and round_expiry and now > event_start + timedelta(seconds=15):
            message = (
                f"Iniciado tarde na rodada {event_start.strftime('%H:%M')}-"
                f"{round_expiry.strftime('%H:%M')} UTC; aguardando próxima rodada em "
                f"{round_expiry.strftime('%H:%M:%S')} UTC para capturar o S0 Chainlink."
            )
        else:
            message = f"Mercado sem priceToBeat/endDate ainda: {slug}"

        self.market_wait_reason = message
        self._print_market_error_once(message)

    def _fetch_market_by_slug(self, slug: str) -> Optional[dict[str, Any]]:
        markets = self._request_json(f"{POLYMARKET_GAMMA_URL}/markets", params={"slug": slug})
        if isinstance(markets, list) and markets:
            orderbook_markets = [m for m in markets if m.get("enableOrderBook")]
            return orderbook_markets[0] if orderbook_markets else markets[0]
        return None

    def _first_event(self, market_data: dict[str, Any]) -> dict[str, Any]:
        events = market_data.get("events")
        if isinstance(events, list) and events:
            return events[0] if isinstance(events[0], dict) else {}
        return {}

    def _extract_price_to_beat(
        self,
        event_data: dict[str, Any],
        market_data: dict[str, Any],
        event_start: Optional[datetime],
    ) -> Optional[float]:
        for container in (event_data, market_data):
            metadata = container.get("eventMetadata") if isinstance(container, dict) else None
            if isinstance(metadata, dict):
                value = safe_float(metadata.get("priceToBeat"))
                if value is not None:
                    return value
        return self._chainlink_price_at_round_start(event_start)

    def _chainlink_price_at_round_start(self, event_start: Optional[datetime]) -> Optional[float]:
        if event_start is None:
            return None

        candidates = [
            row for row in self.chainlink_history
            if 0 <= (row["timestamp"] - event_start).total_seconds() <= 10
        ]
        if not candidates:
            return None
        first_tick = min(candidates, key=lambda row: row["timestamp"])
        return safe_float(first_tick["price"])

    def _print_market_error_once(self, message: str):
        if self._last_market_error != message:
            print(f"Polymarket: {message}")
            self._last_market_error = message

    def refresh_polymarket_books(self):
        for outcome, token_id in self.outcome_tokens.items():
            try:
                book = self._request_json(f"{POLYMARKET_CLOB_URL}/book", params={"token_id": token_id})
                self._update_quote_from_book(outcome, book)
            except Exception as exc:
                print(f"Erro ao buscar book {outcome}: {exc}")

    async def run_polymarket_book_refresh(self):
        loop = asyncio.get_running_loop()
        while True:
            if self.outcome_tokens:
                await loop.run_in_executor(None, self.refresh_polymarket_books)
            await asyncio.sleep(BOOK_REFRESH_SECONDS)

    async def run_polymarket_ws(self):
        while True:
            if not self.outcome_tokens:
                await asyncio.sleep(1)
                continue

            version = self._market_version
            token_ids = [self.outcome_tokens["UP"], self.outcome_tokens["DOWN"]]
            subscription = {
                "assets_ids": token_ids,
                "type": "market",
                "custom_feature_enabled": True,
            }

            try:
                async with websockets.connect(POLYMARKET_MARKET_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(subscription))
                    print(f"CLOB WS conectado para {self.market_slug}")

                    while version == self._market_version:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=35)
                        except asyncio.TimeoutError:
                            await ws.send("PING")
                            continue

                        if message in ("PING", "PONG"):
                            continue
                        self._handle_polymarket_ws_message(message)
            except Exception as exc:
                print(f"CLOB WS desconectado: {exc}. Reconectando...")
                await asyncio.sleep(2)

    def _handle_polymarket_ws_message(self, message: str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    self._handle_polymarket_event(item)
            return

        if isinstance(payload, dict):
            self._handle_polymarket_event(payload)

    def _handle_polymarket_event(self, event: dict[str, Any]):
        event_type = event.get("event_type")

        if event_type == "book":
            outcome = self._outcome_for_asset(event.get("asset_id"))
            if outcome:
                self._update_quote_from_book(outcome, event)
        elif event_type == "price_change":
            for change in event.get("price_changes", []):
                outcome = self._outcome_for_asset(change.get("asset_id"))
                if outcome:
                    self._update_quote(
                        outcome,
                        best_bid=safe_float(change.get("best_bid")),
                        best_ask=safe_float(change.get("best_ask")),
                    )
        elif event_type == "best_bid_ask":
            outcome = self._outcome_for_asset(event.get("asset_id"))
            if outcome:
                self._update_quote(
                    outcome,
                    best_bid=safe_float(event.get("best_bid")),
                    best_ask=safe_float(event.get("best_ask")),
                )
        elif event_type == "last_trade_price":
            outcome = self._outcome_for_asset(event.get("asset_id"))
            price = safe_float(event.get("price"))
            if outcome and price is not None:
                self.quotes[outcome].last_trade_price = price
                self.quotes[outcome].updated_at = utc_now()
        elif event_type == "market_resolved":
            self.market_resolved = True

        self._update_compat_market_fields()

    def _outcome_for_asset(self, asset_id: Any) -> Optional[str]:
        if asset_id is None:
            return None
        asset = str(asset_id)
        for outcome, token_id in self.outcome_tokens.items():
            if token_id == asset:
                return outcome
        return None

    def _update_quote_from_book(self, outcome: str, book: dict[str, Any]):
        bids = self._parse_book_levels(book.get("bids", []))
        asks = self._parse_book_levels(book.get("asks", []))
        self._update_quote(
            outcome,
            best_bid=max([price for price, _ in bids]) if bids else None,
            best_ask=min([price for price, _ in asks]) if asks else None,
            bids=bids,
            asks=asks,
            tick_size=safe_float(book.get("tick_size")),
        )

    def _parse_book_levels(self, raw_levels: Any) -> list[tuple[float, float]]:
        levels = []
        if not isinstance(raw_levels, list):
            return levels

        for level in raw_levels:
            if not isinstance(level, dict):
                continue
            price = safe_float(level.get("price"))
            size = safe_float(level.get("size") or level.get("shares") or level.get("quantity"))
            if price is None or size is None or size <= 0:
                continue
            levels.append((price, size))
        return levels

    def _update_quote(
        self,
        outcome: str,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        bids: Optional[list[tuple[float, float]]] = None,
        asks: Optional[list[tuple[float, float]]] = None,
        tick_size: Optional[float] = None,
    ):
        quote_state = self.quotes[outcome]
        if best_bid is not None:
            quote_state.best_bid = best_bid
        if best_ask is not None:
            quote_state.best_ask = best_ask
        if bids is not None:
            quote_state.bids = sorted(bids, reverse=True)
        if asks is not None:
            quote_state.asks = sorted(asks)
        if tick_size is not None:
            quote_state.tick_size = tick_size
        quote_state.updated_at = utc_now()

    def _update_compat_market_fields(self):
        up_quote = self.quotes.get("UP")
        if up_quote and up_quote.mid is not None:
            self.polymarket_price = up_quote.mid
        if up_quote and up_quote.spread is not None:
            self.polymarket_spread = up_quote.spread

    def get_quote(self, outcome: str) -> OutcomeQuote:
        return self.quotes[outcome.upper()]

    def market_is_ready(self) -> bool:
        return (
            self.S0 is not None
            and self.round_expiry is not None
            and "UP" in self.outcome_tokens
            and "DOWN" in self.outcome_tokens
            and self.quotes["UP"].is_ready
            and self.quotes["DOWN"].is_ready
            and self.quotes["UP"].has_depth
            and self.quotes["DOWN"].has_depth
        )

    async def run_history_recorder(self):
        while True:
            await asyncio.sleep(1)
            if self.current_btc_price is None:
                continue

            now = utc_now()
            self.history.append({"timestamp": now, "price": self.current_btc_price})

            cutoff = now - timedelta(seconds=350)
            self.history = [row for row in self.history if row["timestamp"] >= cutoff]
