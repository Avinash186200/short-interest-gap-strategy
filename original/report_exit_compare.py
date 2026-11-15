"""
report_exit_compare.py — Item 2 validation: does the regime-sized
trailing-ATR exit (backtest.py: rule_name == "trailing_atr") improve on the
fixed 2.5-3% take-profit exit (rule_name == "hold_above_open_945_tp3" /
"failed_bounce_1000_tp_2.5")?

Compares the two rule families head-to-head on the SAME (trade_date, ticker,
side, gap_threshold_pct) entries, by side / gap_threshold / year and overall.

Writes output/exit_comparison.md.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
OUT_MD = ROOT / "output" / "exit_comparison.md"

FIXED_RULES = {"LONG": "hold_above_open_945_tp3", "SHORT": "failed_bounce_1000_tp_2.5"}


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "median_ret_pct": g["realized_return_pct"].median(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
        "std_ret_pct": g["realized_return_pct"].std(),
    })


def main() -> int:
    if not TRADES.exists():
        print(f"[exit_compare] missing {TRADES}")
        return 1
    df = pd.read_parquet(TRADES)
    df["year"] = df["trade_date"].astype(str).str[:4]
    df = df.dropna(subset=["realized_return_pct"])

    is_fixed = (
        ((df["side"] == "LONG") & (df["rule_name"] == FIXED_RULES["LONG"]))
        | ((df["side"] == "SHORT") & (df["rule_name"] == FIXED_RULES["SHORT"]))
    )
    is_trailing = df["rule_name"] == "trailing_atr"
    df = df[is_fixed | is_trailing].copy()
    df["exit_family"] = np.where(is_trailing[df.index], "trailing_atr", "fixed_tp")

    print(f"[exit_compare] fixed_tp={ (df['exit_family']=='fixed_tp').sum():,}, "
          f"trailing_atr={(df['exit_family']=='trailing_atr').sum():,}", flush=True)

    # head-to-head on common (trade_date, ticker, side, gap_threshold_pct)
    key = ["trade_date", "ticker", "side", "gap_threshold_pct"]
    wide = df.pivot_table(index=key, columns="exit_family", values="realized_return_pct", aggfunc="first")
    wide = wide.dropna(subset=["fixed_tp", "trailing_atr"]).reset_index()
    wide["year"] = wide["trade_date"].astype(str).str[:4]

    by_side_thr_year = (
        df.groupby(["exit_family", "side", "gap_threshold_pct", "year"]).apply(agg, include_groups=False).reset_index()
    )
    overall = df.groupby(["exit_family", "side", "gap_threshold_pct"]).apply(agg, include_groups=False).reset_index()

    # paired comparison: avg of (trailing - fixed) per matched trade
    wide["delta"] = wide["trailing_atr"] - wide["fixed_tp"]
    paired = wide.groupby(["side", "gap_threshold_pct"]).agg(
        n_matched=("delta", "size"),
        avg_delta_pct=("delta", "mean"),
        median_delta_pct=("delta", "median"),
        pct_trailing_better=("delta", lambda x: (x > 0).mean() * 100),
        sum_fixed=("fixed_tp", "sum"),
        sum_trailing=("trailing_atr", "sum"),
    ).reset_index()

    lines = ["# Exit Rule Comparison: fixed TP vs trailing-ATR (item 2)", "",
             f"- Trailing exit params: act={1.0}xATR, trail={0.5}xATR, hard_stop={1.0}xATR "
             "(see TRAIL_*_MULT in backtest.py)",
             "",
             "## By exit family / side / threshold / year",
             "",
             by_side_thr_year.to_markdown(index=False, floatfmt=".2f"),
             "",
             "## Overall (all years pooled)",
             "",
             overall.to_markdown(index=False, floatfmt=".2f"),
             "",
             "## Paired comparison (same trade, trailing - fixed)",
             "",
             paired.to_markdown(index=False, floatfmt=".3f"),
             ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[exit_compare] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
