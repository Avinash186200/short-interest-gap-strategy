"""
report_selected.py — the PRIMARY strategy report, built on
data/trades_selected.parquet (output of select_signals.py).

Reports three things:
  1. The live strategy: LONG, drift_low top-5 selection + portfolio
     controls (live_trade=True - 5-day per-ticker cooling-off applied on
     top of a -18% wide stop, see output/combined_long_controls.md /
     output/risk_metrics.md). This is the current best validated spec.
  1b. The candidate-selection-only baseline (selected=True, no controls),
      shown alongside 1 to make the controls' lift visible.
  2. Everything else (SHORT, and LONG candidates outside the top-5), shown
     separately and clearly labeled "not selected / no validated edge" -
     kept visible for context, not silently dropped.

Writes output/results_summary_selected.md and
output/equity_curve_selected.csv (live strategy only).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
OUT_MD = ROOT / "output" / "results_summary_selected.md"
OUT_EQ = ROOT / "output" / "equity_curve_selected.csv"


def agg(g: pd.DataFrame, ret_col: str = "realized_return_pct") -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g[ret_col] > 0).mean() * 100,
        "avg_ret_pct": g[ret_col].mean(),
        "sum_ret_pct": g[ret_col].sum(),
    })


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[report_selected] missing {TRADES_SEL} - run select_signals.py first")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["year"] = df["trade_date"].dt.year

    selected = df[df["selected"]].copy()
    live = df[df["live_trade"]].copy()
    not_selected = df[~df["selected"]].copy()

    lines = [
        "# Primary Strategy Report (selected signals)",
        "",
        "Built on `data/trades_selected.parquet` (output of "
        "select_signals.py). Splits the full trade universe into the "
        "**live strategy** (LONG, `drift_low_top5` selection + portfolio "
        "controls - the current best validated spec), the **selection-only "
        "baseline** (same candidates, no controls, for comparison), and "
        "**everything else** (SHOWN FOR CONTEXT, not part of the strategy - "
        "no validated edge).",
        "",
        "## 1. Live strategy: drift_low top-5 + cooling-off + wide stop",
        "",
        "Each day, rank LONG `hold_above_open_945_tp3` candidates by "
        "settlement->publication price_drift_pct ASCENDING (biggest "
        "pre-publication price decline first) within each "
        "gap_threshold_pct group; trade only the top 5 "
        "(`output/ranking_refine2_corrected.md`, `output/rank_drift_recent.md`). "
        "On top of that, each trade's exit is re-simulated with a -18% wide "
        "hard stop, and a ticker is skipped for 5 days after any "
        "stop-adjusted loss (`output/combined_long_controls.md`, "
        "`output/risk_metrics.md`). Both controls were chosen on TRAIN "
        "(trade_date < 2026-01-22) and validated once on held-out TEST "
        "before being folded in here.",
        "",
    ]

    summ = live.groupby("gap_threshold_pct").apply(
        lambda g: agg(g, "stop_adjusted_return_pct"), include_groups=False).reset_index()
    lines.append("### Full-sample summary")
    lines.append("")
    lines.append(summ.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    byyear = live.groupby(["gap_threshold_pct", "year"]).apply(
        lambda g: agg(g, "stop_adjusted_return_pct"), include_groups=False).reset_index()
    lines.append("### By year")
    lines.append("")
    lines.append(byyear.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    baseline_summ = selected.groupby("gap_threshold_pct").apply(agg, include_groups=False).reset_index()
    lines.append("### 1b. Selection-only baseline (no controls, for comparison)")
    lines.append("")
    lines.append(baseline_summ.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append(
        "Controls roughly double full-sample sum_ret per threshold and lift "
        "Sharpe 0.61->2.27 (2.5%) / 0.38->2.14 (3.0%) - see "
        "`output/risk_metrics.md` for the full risk-adjusted comparison."
    )
    lines.append("")

    # Equity curve (pooled both thresholds, live strategy)
    eq = live.groupby("trade_date")["stop_adjusted_return_pct"].sum().reset_index().sort_values("trade_date")
    eq = eq.rename(columns={"stop_adjusted_return_pct": "realized_return_pct"})
    eq["cum_ret_pct"] = eq["realized_return_pct"].cumsum()
    eq["running_max"] = eq["cum_ret_pct"].cummax()
    eq["drawdown_pct"] = eq["cum_ret_pct"] - eq["running_max"]
    eq.to_csv(OUT_EQ, index=False)

    lines.append("### Equity curve")
    lines.append("")
    lines.append(f"Pooled (both thresholds) daily equity curve written to "
                  f"`{OUT_EQ.relative_to(ROOT)}` ({len(eq)} trading days).")
    lines.append(f"- Final cumulative return: {eq['cum_ret_pct'].iloc[-1]:.2f}%")
    lines.append(f"- Peak: {eq['cum_ret_pct'].max():.2f}% on {eq.loc[eq['cum_ret_pct'].idxmax(), 'trade_date'].date()}")
    lines.append(f"- Max drawdown: {eq['drawdown_pct'].min():.2f}%")
    lines.append("")
    lines.append(
        "**Caveat:** see output/rank_drift_strategy.md for the severe "
        "intervening drawdown of the selection-only baseline (peaked "
        "+446.6% on 2025-10-13, drew down to ~+123% by 2026-03-31, "
        "recovered by 2026-05-22). The controls reduce but do not "
        "eliminate this - max drawdown above is smaller than the "
        "baseline's, but path still matters, not just the endpoint."
    )
    lines.append("")

    # Everything else - context only
    lines.append("## 2. Not selected / no validated edge (context only)")
    lines.append("")
    lines.append(
        "These are NOT part of the strategy - shown for transparency. "
        "SHORT (either rule) has no validated selection or exit rule "
        "(see NEXT_ITERATION_PLAN.md, train/test split + ranking-refinement "
        "follow-ups - all negative OOS). LONG candidates outside the daily "
        "top-5 are excluded by definition of the selection rule."
    )
    lines.append("")
    ns_summ = not_selected.groupby(["side", "rule_name", "gap_threshold_pct"]).apply(agg, include_groups=False).reset_index()
    lines.append(ns_summ.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_selected] wrote {OUT_MD} and {OUT_EQ}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
