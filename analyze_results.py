from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.csv")
OUTPUT_DIR = Path("analysis")
BUCKET_SIZE = 10


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("results.csv vazio")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["round_expiry"] = pd.to_datetime(df["round_expiry"], utc=True, errors="coerce")
    df["PnL_USD"] = pd.to_numeric(df["PnL_USD"], errors="coerce")
    df["bet_size_USD"] = pd.to_numeric(df["bet_size_USD"], errors="coerce")
    df["P_model"] = pd.to_numeric(df["P_model"], errors="coerce")
    df["P_mkt_paid"] = pd.to_numeric(df["P_mkt_paid"], errors="coerce")
    df["is_win"] = df["PnL_USD"] > 0
    df["trade_index"] = range(1, len(df) + 1)
    df["cum_pnl"] = df["PnL_USD"].cumsum()
    df["cum_peak"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["cum_pnl"] - df["cum_peak"]
    df["bucket"] = ((df["trade_index"] - 1) // BUCKET_SIZE) + 1
    return df


def summarize(df: pd.DataFrame) -> None:
    gross_profit = df.loc[df["PnL_USD"] > 0, "PnL_USD"].sum()
    gross_loss = -df.loc[df["PnL_USD"] < 0, "PnL_USD"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    settled_mask = df["status"].isin(["WON", "LOST"])
    settled = df.loc[settled_mask]
    early_exits = df.loc[~settled_mask]

    print("=== RESUMO GERAL ===")
    print(f"trades: {len(df)}")
    print(f"pnl_total_usd: {df['PnL_USD'].sum():.2f}")
    print(f"pnl_medio_usd: {df['PnL_USD'].mean():.2f}")
    print(f"mediana_pnl_usd: {df['PnL_USD'].median():.2f}")
    print(f"win_rate_pnl: {df['is_win'].mean() * 100:.2f}%")
    print(f"profit_factor: {profit_factor:.3f}")
    print(f"max_drawdown_usd: {df['drawdown'].min():.2f}")
    print(f"maior_gain_usd: {df['PnL_USD'].max():.2f}")
    print(f"maior_loss_usd: {df['PnL_USD'].min():.2f}")
    print()

    print("=== POR STATUS ===")
    status_summary = (
        df.groupby("status")
        .agg(
            trades=("status", "size"),
            pnl_total=("PnL_USD", "sum"),
            pnl_medio=("PnL_USD", "mean"),
            win_rate=("is_win", "mean"),
        )
        .sort_values("pnl_total")
    )
    for status, row in status_summary.iterrows():
        print(
            f"{status}: trades={int(row['trades'])}, pnl_total={row['pnl_total']:.2f}, "
            f"pnl_medio={row['pnl_medio']:.2f}, win_rate={row['win_rate'] * 100:.2f}%"
        )
    print()

    if not settled.empty:
        print("=== SÓ RODADAS LIQUIDADAS ===")
        print(f"trades: {len(settled)}")
        print(f"win_rate_liquidado: {(settled['status'] == 'WON').mean() * 100:.2f}%")
        print(f"pnl_total_liquidado: {settled['PnL_USD'].sum():.2f}")
        print()

    if not early_exits.empty:
        print("=== SÓ SAÍDAS ANTECIPADAS ===")
        print(f"trades: {len(early_exits)}")
        print(f"win_rate_antecipada: {early_exits['is_win'].mean() * 100:.2f}%")
        print(f"pnl_total_antecipado: {early_exits['PnL_USD'].sum():.2f}")
        print()

    print("=== WIN RATE POR BLOCO DE 10 ===")
    bucket_summary = (
        df.groupby("bucket")
        .agg(
            start_trade=("trade_index", "min"),
            end_trade=("trade_index", "max"),
            win_rate=("is_win", "mean"),
            pnl_total=("PnL_USD", "sum"),
        )
    )
    for bucket, row in bucket_summary.iterrows():
        print(
            f"bloco_{int(bucket):02d} trades_{int(row['start_trade'])}-{int(row['end_trade'])}: "
            f"win_rate={row['win_rate'] * 100:.2f}%, pnl_total={row['pnl_total']:.2f}"
        )


def plot_equity_curve(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["trade_index"], df["cum_pnl"], color="#1565c0", linewidth=2)
    ax.fill_between(df["trade_index"], df["cum_pnl"], 0, color="#90caf9", alpha=0.35)
    ax.axhline(0, color="#424242", linewidth=1, linestyle="--")
    ax.set_title("Curva de PnL Acumulado por Ordem de Fechamento")
    ax.set_xlabel("Trade")
    ax.set_ylabel("PnL acumulado (USD)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close(fig)


def plot_bucket_win_rate(df: pd.DataFrame, output_dir: Path) -> None:
    bucket_summary = (
        df.groupby("bucket")
        .agg(
            win_rate=("is_win", "mean"),
            start_trade=("trade_index", "min"),
            end_trade=("trade_index", "max"),
        )
        .reset_index()
    )
    labels = [
        f"{int(row.start_trade)}-{int(row.end_trade)}"
        for row in bucket_summary.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(labels, bucket_summary["win_rate"] * 100, marker="o", linewidth=2, color="#2e7d32")
    ax.axhline(df["is_win"].mean() * 100, color="#c62828", linestyle="--", linewidth=1.5, label="média geral")
    ax.set_title("Win Rate por Blocos de 10 Trades")
    ax.set_xlabel("Faixa de trades")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "win_rate_per_10_trades.png", dpi=160)
    plt.close(fig)


def plot_status_pnl(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby("status")
        .agg(trades=("status", "size"), pnl_total=("PnL_USD", "sum"))
        .sort_values("pnl_total")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(summary["status"], summary["trades"], color="#6a1b9a")
    axes[0].set_title("Quantidade por Status")
    axes[0].set_ylabel("Trades")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#c62828" if value < 0 else "#2e7d32" for value in summary["pnl_total"]]
    axes[1].bar(summary["status"], summary["pnl_total"], color=colors)
    axes[1].axhline(0, color="#424242", linewidth=1, linestyle="--")
    axes[1].set_title("PnL Total por Status")
    axes[1].set_ylabel("PnL total (USD)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "status_breakdown.png", dpi=160)
    plt.close(fig)


def plot_pnl_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df["PnL_USD"], bins=30, color="#ef6c00", edgecolor="white")
    ax.axvline(df["PnL_USD"].mean(), color="#1565c0", linestyle="--", linewidth=1.5, label="média")
    ax.axvline(df["PnL_USD"].median(), color="#2e7d32", linestyle=":", linewidth=1.5, label="mediana")
    ax.set_title("Distribuição de PnL por Trade")
    ax.set_xlabel("PnL por trade (USD)")
    ax.set_ylabel("Frequência")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pnl_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_results(RESULTS_PATH)
    summarize(df)
    plot_equity_curve(df, OUTPUT_DIR)
    plot_bucket_win_rate(df, OUTPUT_DIR)
    plot_status_pnl(df, OUTPUT_DIR)
    plot_pnl_distribution(df, OUTPUT_DIR)
    print()
    print(f"graficos_salvos_em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
