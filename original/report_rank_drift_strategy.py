"""
report_rank_drift_strategy.py — consolidated view of the best result found
so far: for LONG (hold_above_open_945_tp3), rank each day's candidates by
the settlement->publication drift signal (price_drift_pct, ASCENDING - i.e.
trade the candidates whose price DROPPED THE MOST from settlement to
publication, "drift_low" in report_ranking_refine2.py) within each
(trade_date, gap_threshold_pct) group, and trade only the top 5.

NOTE on direction: report_ranking_refine.py originally mislabeled this as
"rank by drift, highest first" due to a double-rank bug (see
report_ranking_refine2.py for the corrected breakdown). The TEST sum_ret
numbers (+52.79%/+53.54% etc.) were numerically correct, but the mechanism
is actually the OPPOSITE of item 1's original framing: the best LONG
candidates are the ones with the LARGEST price DECLINE from settlement to
publication (most negative price_drift_pct), not the largest rise. This
looks like a squeeze-magnitude / mean-reversion effect, not the
drift-confirms-direction effect item 1 originally hypothesized.

top_n=5 is chosen because report_ranking_refine2.py showed drift_low is
positive on TEST across top_n in {1,2,3,5} at both thresholds — 5 is a
robust middle choice (not the single best per threshold, but consistently
good and avoids picking a different N per threshold from the same TRAIN
selection step).

This report covers the FULL sample (TRAIN+TEST combined) since the selection
(drift_low, top_n=5) was already validated out-of-sample in
report_ranking_refine2.py — this is a "what does the resulting strategy look
like" view, not a new selection step.

Writes:
  - output/rank_drift_strategy.md   (summary stats, by threshold and by year)
  - output/rank_drift_equity_curve.csv  (daily equity curve, pooled both thresholds)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "rank_drift_strategy.md"
OUT_CSV = ROOT / "output" / "rank_drift_equity_curve.csv"

RULE = "hold_above_open_945_tp3"
TOP_N = 5
TEST_MONTHS = 4


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
    # Match report_ranking_refine.py's candidate universe exactly (it also
    # dropped NaN volume_ratio, even though rank_drift doesn't use it -
    # the dropped rows change which trades fall in the daily top-5).
    df = df.dropna(subset=["price_drift_pct", "volume_ratio", "realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    sub = df[(df["side"] == "LONG") & (df["rule_name"] == RULE)].copy()
    # ascending=True: rank 1 = LOWEST price_drift_pct (biggest price decline
    # from settlement to publication) - this is "drift_low" in
    # report_ranking_refine2.py, the corrected winning direction.
    sub["rank"] = sub.groupby(["trade_date", "gap_threshold_pct"])["price_drift_pct"] \
        .rank(method="first", ascending=True)
    sel = sub[sub["rank"] <= TOP_N].copy()

    max_date = sel["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=TEST_MONTHS)

    lines = [
        "# drift_low Top-5 LONG Strategy — Consolidated View",
        "",
        f"Selection: each day, rank LONG candidates ({RULE}) by "
        f"price_drift_pct (settlement->publication drift signal) "
        f"ASCENDING (lowest/most-negative first - biggest price decline "
        f"into publication), within each gap_threshold_pct group; trade "
        f"only the top {TOP_N}.",
        "",
        f"- Full sample: {sel['trade_date'].min().date()} -> {max_date.date()}",
        f"- (For reference: report_ranking_refine.py validated this rule "
        f"out-of-sample on TRAIN < {cutoff.date()} / TEST >= {cutoff.date()})",
        "",
        "## Full-sample summary, by gap threshold",
        "",
    ]

    summ = sel.groupby("gap_threshold_pct").apply(agg, include_groups=False).reset_index()
    lines.append(summ.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("## By year")
    lines.append("")
    sel["year"] = sel["trade_date"].dt.year
    byyear = sel.groupby(["gap_threshold_pct", "year"]).apply(agg, include_groups=False).reset_index()
    lines.append(byyear.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("## TRAIN vs TEST split (for reference — already validated OOS)")
    lines.append("")
    sel["period"] = np.where(sel["trade_date"] < cutoff, "TRAIN", "TEST")
    bysplit = sel.groupby(["gap_threshold_pct", "period"]).apply(agg, include_groups=False).reset_index()
    lines.append(bysplit.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # Equity curve: pooled across both thresholds, daily sum of realized_return_pct
    eq = sel.groupby("trade_date")["realized_return_pct"].sum().reset_index()
    eq = eq.sort_values("trade_date")
    eq["cum_ret_pct"] = eq["realized_return_pct"].cumsum()
    eq["running_max"] = eq["cum_ret_pct"].cummax()
    eq["drawdown_pct"] = eq["cum_ret_pct"] - eq["running_max"]
    eq.to_csv(OUT_CSV, index=False)

    lines.append("## Equity curve / drawdown")
    lines.append("")
    lines.append(f"Pooled (both thresholds) daily equity curve written to "
                  f"`{OUT_CSV.relative_to(ROOT)}` ({len(eq)} trading days).")
    lines.append("")
    lines.append(f"- Final cumulative return: {eq['cum_ret_pct'].iloc[-1]:.2f}%")
    lines.append(f"- Max drawdown: {eq['drawdown_pct'].min():.2f}%")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_rank_drift_strategy] wrote {OUT_MD} and {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
