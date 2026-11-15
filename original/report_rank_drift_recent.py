"""
report_rank_drift_recent.py — last 3/4-month breakdown of the drift_low
top-5 LONG strategy (the best result so far, see
output/rank_drift_strategy.md), per Expert's original request to re-check
recent performance (matches the framing of item 5's last-3-months check,
report_recent3m.py, but for the new drift_low selection rule).

Writes output/rank_drift_recent.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "rank_drift_recent.md"

RULE = "hold_above_open_945_tp3"
TOP_N = 5


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    trades = pd.read_parquet(TRADES)
    drift = pd.read_parquet(DRIFT)

    df = trades.merge(
        drift,
        left_on=["settlement_date", "ticker"],
        right_on=["settlementDate", "symbolCode"],
        how="left",
    )
    df = df.dropna(subset=["price_drift_pct", "volume_ratio", "realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    sub = df[(df["side"] == "LONG") & (df["rule_name"] == RULE)].copy()
    sub["rank"] = sub.groupby(["trade_date", "gap_threshold_pct"])["price_drift_pct"] \
        .rank(method="first", ascending=True)
    sel = sub[sub["rank"] <= TOP_N].copy()
    max_date = sel["trade_date"].max()

    lines = [
        "# drift_low Top-5 LONG Strategy — Recent (Last 3/4 Months)",
        "",
        f"Same selection as output/rank_drift_strategy.md (rank LONG "
        f"candidates by price_drift_pct ascending, trade top {TOP_N}/day). "
        f"Per Expert's original request, re-checking the last 3-4 months "
        f"specifically (data through {max_date.date()}).",
        "",
        "## Last 3 / 4 months, by gap threshold",
        "",
    ]

    rows = []
    for months in [3, 4]:
        cutoff = max_date - pd.DateOffset(months=months)
        recent = sel[sel["trade_date"] >= cutoff]
        for thr in [2.5, 3.0]:
            m = agg(recent[recent["gap_threshold_pct"] == thr])
            rows.append({"window": f"last {months}m (>= {cutoff.date()})", "gap_threshold_pct": thr, **m.to_dict()})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("## Monthly breakdown (both thresholds combined)")
    lines.append("")
    sel["year_month"] = sel["trade_date"].dt.to_period("M").astype(str)
    monthly = sel.groupby("year_month").apply(agg, include_groups=False).reset_index()
    lines.append(monthly.tail(8).to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append(
        "**Read:** the last 3 months (>= 2026-02-22) are net positive "
        "(+70.85% @2.5%, +42.48% @3.0%), and the last 4 months are also "
        "positive (+52.79%/+53.54%) - this confirms the OOS result holds "
        "in the specific recent window Expert asked about. However, the "
        "monthly breakdown shows it is NOT uniform within that window: "
        "March 2026 was still negative (-42.85%), with the positive result "
        "driven by a strong April + May recovery (+57.95%/+85.31%). This "
        "lines up with the equity-curve drawdown noted in "
        "output/rank_drift_strategy.md - the strategy was still "
        "recovering from its Oct'25-Mar'26 drawdown through March, then "
        "turned sharply positive in Apr-May."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_rank_drift_recent] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
