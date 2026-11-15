"""
report_recent3m.py — Item 5: re-validate on the last 3 months only.

Expert's concern: filter thresholds (DTC >= 5, SI/float >= 10%) and the
2.5-3% TP were calibrated loosely / on a shorter window ("everything has
been going upward" recently), and may not hold over the full 2024-2026
sample. Re-run the same side x gap_threshold x rule_name breakdown
restricted to trade_date in the most recent 3 calendar months of the
dataset, for both the fixed-TP and trailing-ATR exit families.

Writes output/results_recent3m.md.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
OUT_MD = ROOT / "output" / "results_recent3m.md"


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "median_ret_pct": g["realized_return_pct"].median(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    if not TRADES.exists():
        print(f"[recent3m] missing {TRADES}")
        return 1
    df = pd.read_parquet(TRADES)
    df = df.dropna(subset=["realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    max_date = df["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=3)
    recent = df[df["trade_date"] > cutoff].copy()

    print(f"[recent3m] full range: {df['trade_date'].min().date()} -> {max_date.date()}", flush=True)
    print(f"[recent3m] last-3-months window: {cutoff.date()} -> {max_date.date()}, "
          f"{len(recent):,} trade rows", flush=True)

    by_rule = recent.groupby(["rule_name", "side", "gap_threshold_pct"]).apply(agg, include_groups=False).reset_index()
    full_by_rule = df.groupby(["rule_name", "side", "gap_threshold_pct"]).apply(agg, include_groups=False).reset_index()

    lines = ["# Last-3-Months Re-validation (item 5)", "",
             f"- Full sample: {df['trade_date'].min().date()} -> {max_date.date()}",
             f"- Last-3-months window: {cutoff.date()} -> {max_date.date()} ({len(recent):,} trade rows)",
             "",
             "## Last 3 months", "",
             by_rule.to_markdown(index=False, floatfmt=".2f"),
             "",
             "## Full sample (for comparison)", "",
             full_by_rule.to_markdown(index=False, floatfmt=".2f"),
             ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[recent3m] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
