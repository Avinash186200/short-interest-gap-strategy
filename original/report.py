"""
report.py — aggregate trades.parquet into a results summary:
  - by side x gap_threshold x year:  n_trades, hit_rate, avg_return, sum_return, sharpe-ish
  - cumulative equity curve (1% position size, equal weight, daily)
  - drawdown
Writes output/results_summary.md and output/equity_curve.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
OUT_MD = ROOT / "output" / "results_summary.md"
OUT_EQ = ROOT / "output" / "equity_curve.csv"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["side", "gap_threshold_pct", "year"])
    s = g["realized_return_pct"].agg(
        n_trades="size",
        avg_ret_pct="mean",
        median_ret_pct="median",
        sum_ret_pct="sum",
        std_ret_pct="std",
    ).reset_index()
    hits = g.apply(lambda x: (x["realized_return_pct"] > 0).mean() * 100, include_groups=False).rename("hit_rate_pct").reset_index()
    out = s.merge(hits, on=["side", "gap_threshold_pct", "year"])
    out["avg_per_trade_R"] = out["avg_ret_pct"] / out["std_ret_pct"].replace(0, np.nan)
    return out.sort_values(["side", "gap_threshold_pct", "year"])


def equity_curve(df: pd.DataFrame, per_trade_size_pct: float = 1.0) -> pd.DataFrame:
    """Daily compounded equity assuming 1% position per trade, equal weight per day."""
    df = df.copy()
    df["pnl_pct_capital"] = df["realized_return_pct"] * (per_trade_size_pct / 100.0)
    daily = df.groupby(["trade_date", "side", "gap_threshold_pct"])["pnl_pct_capital"].sum().reset_index()
    out = []
    for (side, thr), g in daily.groupby(["side", "gap_threshold_pct"]):
        g = g.sort_values("trade_date").copy()
        g["cum_pct"] = g["pnl_pct_capital"].cumsum()
        g["cum_mult"] = (1 + g["pnl_pct_capital"] / 100.0).cumprod()
        g["dd_pct"] = (g["cum_mult"] / g["cum_mult"].cummax() - 1) * 100
        g["side"] = side
        g["gap_threshold_pct"] = thr
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    if not TRADES.exists():
        print(f"[report] missing {TRADES}")
        return 1
    df = pd.read_parquet(TRADES)
    df["year"] = df["trade_date"].astype(str).str[:4]
    df = df.dropna(subset=["realized_return_pct"])
    print(f"[report] {len(df):,} trades, {df['ticker'].nunique()} tickers, "
          f"{df['trade_date'].nunique()} trade dates", flush=True)

    s = summarize(df)
    eq = equity_curve(df)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Backtest Results", "",
             f"- Total trades: {len(df):,}",
             f"- Tickers: {df['ticker'].nunique()}",
             f"- Date range: {df['trade_date'].min()} -> {df['trade_date'].max()}",
             "",
             "## Per-side / threshold / year",
             ""]
    lines.append(s.to_markdown(index=False, floatfmt=".2f"))

    overall = df.groupby(["side", "gap_threshold_pct"])["realized_return_pct"].agg(["count","mean","median","sum","std"])
    overall["hit_rate_pct"] = df.groupby(["side", "gap_threshold_pct"])["realized_return_pct"].apply(lambda x: (x>0).mean()*100)
    lines += ["", "## Overall (all years pooled)", "", overall.reset_index().to_markdown(index=False, floatfmt=".2f")]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {OUT_MD}", flush=True)

    if not eq.empty:
        OUT_EQ.parent.mkdir(parents=True, exist_ok=True)
        eq.to_csv(OUT_EQ, index=False)
        print(f"[report] wrote {OUT_EQ}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
