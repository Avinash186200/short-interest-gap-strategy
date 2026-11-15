"""
report_regime_gate.py — Item 5 follow-up: does regime-gating the trailing-ATR
exit (only trusting it when the current ATR% is itself elevated vs its own
trailing 60-session history) recover an edge that holds in the recent
3-month window, where the ungated trailing exit was negative across the
board?

For each trailing_atr trade, atr_pctile (written by backtest.py) is the
rolling percentile rank (0-100) of that session's ATR% within its own
trailing 60-session history. We split trades into:
  - "high regime"  : atr_pctile >= GATE_PCTILE  (volatility itself elevated)
  - "low regime"   : atr_pctile <  GATE_PCTILE

and compare hit rate / sum return in each bucket, for the full sample and
for the last-3-months window (same window as report_recent3m.py), by side
and gap threshold.

Writes output/regime_gate.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
OUT_MD = ROOT / "output" / "regime_gate.md"

GATE_PCTILE = 50.0
RECENT_MONTHS = 3


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def breakdown(df: pd.DataFrame, label: str) -> list[str]:
    d = df.dropna(subset=["atr_pctile"]).copy()
    d["regime"] = d["atr_pctile"].apply(lambda p: "high (>=50pct)" if p >= GATE_PCTILE else "low (<50pct)")
    by = (
        d.groupby(["side", "gap_threshold_pct", "regime"])
        .apply(agg, include_groups=False)
        .reset_index()
        .sort_values(["side", "gap_threshold_pct", "regime"])
    )
    return [f"### {label}", "", by.to_markdown(index=False, floatfmt=".2f"), ""]


def main() -> int:
    if not TRADES.exists():
        print(f"[report_regime_gate] missing {TRADES}")
        return 1

    trades = pd.read_parquet(TRADES)
    df = trades[trades["rule_name"] == "trailing_atr"].copy()
    if "atr_pctile" not in df.columns:
        print("[report_regime_gate] no atr_pctile column -- rerun backtest.py")
        return 1
    df = df.dropna(subset=["realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    max_date = df["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=RECENT_MONTHS)
    recent = df[df["trade_date"] > cutoff]

    lines = [
        "# Regime-Gated Trailing-ATR Exit (item 5 follow-up)",
        "",
        f"- Gate: split trailing_atr trades by `atr_pctile` (rolling percentile rank "
        f"of session ATR% within its own trailing 60-session history) at "
        f"{GATE_PCTILE:.0f}th percentile.",
        f"- Full sample: {len(df):,} trailing_atr trades with atr_pctile.",
        f"- Recent window: trade_date > {cutoff.date()} ({len(recent):,} of "
        f"{len(df):,} rows, matches report_recent3m.py window).",
        "",
        "## Full sample, split by regime",
        "",
    ]
    lines += breakdown(df, "Full sample")
    lines += ["## Last 3 months, split by regime", ""]
    lines += breakdown(recent, "Last 3 months")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_regime_gate] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
