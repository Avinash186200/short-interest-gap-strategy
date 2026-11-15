"""
report_signal_sanitycheck.py — sanity-checks the `drift_low` top-5 LONG
selection mechanism (see select_signals.py, output/results_summary_selected.md)
against individual examples and sub-population breakdowns, to test the
"biggest pre-publication price drop -> biggest post-signal squeeze"
interpretation.

Looks at:
  1. The most-extreme negative-drift trades (the textbook "squeeze" cases)
     and whether they actually win.
  2. Correlation between price_drift_pct / volume_ratio and realized return
     across all selected trades.
  3. Performance by within-day rank position (1 = most negative drift, ...
     5 = least).
  4. How often the daily top-5 cutoff is actually binding (>5 candidates
     that day), and on those binding days, whether selected vs not-selected
     candidates differ in performance and in price_drift_pct.
  5. A look at the extreme positive-drift outliers (>100%) excluded by the
     selection, to flag possible data-quality artifacts.

Writes output/signal_sanitycheck.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
OUT_MD = ROOT / "output" / "signal_sanitycheck.md"

LONG_PRIMARY_RULE = "hold_above_open_945_tp3"


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[sanitycheck] missing {TRADES_SEL} - run select_signals.py first")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[df["selected"]].copy()

    lines = [
        "# Signal sanity check: `drift_low` top-5 LONG selection",
        "",
        "Tests the 'biggest pre-publication price drop -> biggest "
        "post-signal squeeze' interpretation of `drift_low` against "
        "individual examples and sub-population breakdowns.",
        "",
        "## 1. Most-extreme negative-drift selected trades",
        "",
        "These are the textbook 'squeeze' candidates - the most negative "
        "price_drift_pct values in the selected universe.",
        "",
    ]
    cols = ["trade_date", "ticker", "gap_threshold_pct", "gap_pct",
            "price_drift_pct", "volume_ratio", "entry_price", "exit_price",
            "exit_rule", "realized_return_pct"]
    top = sel.sort_values("price_drift_pct").head(10)
    lines.append(top[cols].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append(
        "Mixed: most of these hit the +3% TP, but a couple (WW, EOSE) lose "
        "money on EOD exit - the magnitude of the pre-publication drop does "
        "not guarantee a squeeze."
    )
    lines.append("")

    lines.append("## 2. Correlation: price_drift_pct / volume_ratio vs realized return")
    lines.append("")
    corr_drift = sel[["price_drift_pct", "realized_return_pct"]].corr().iloc[0, 1]
    corr_vol = sel[["volume_ratio", "realized_return_pct"]].corr().iloc[0, 1]
    lines.append(f"- corr(price_drift_pct, realized_return_pct) = **{corr_drift:.3f}**")
    lines.append(f"- corr(volume_ratio, realized_return_pct) = **{corr_vol:.3f}**")
    lines.append("")
    lines.append(
        "Both essentially zero. The *magnitude* of price_drift_pct does not "
        "linearly predict trade outcome - the edge (if any) is not "
        "\"more negative drift = bigger squeeze\" in a continuous sense."
    )
    lines.append("")

    lines.append("## 3. Performance by within-day rank position (1=most negative drift)")
    lines.append("")
    sel["rank"] = sel.groupby(["trade_date", "gap_threshold_pct"])["price_drift_pct"] \
        .rank(method="first", ascending=True)
    g = sel.groupby("rank")["realized_return_pct"].agg(
        n="count", avg_ret="mean", sum_ret="sum",
        hit_rate=lambda x: (x > 0).mean() * 100,
    ).reset_index()
    lines.append(g.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append(
        "Rank 1 (most negative drift that day) is the strongest single "
        "rank (+193.1% sum_ret, 64.5% hit), but ranks 2 and 5 are negative "
        "- not a clean monotonic relationship."
    )
    lines.append("")

    lines.append("## 4. Is the daily top-5 cutoff actually binding?")
    lines.append("")
    long_primary = df[(df["side"] == "LONG") & (df["rule_name"] == LONG_PRIMARY_RULE)
                       & df["price_drift_pct"].notna()].copy()
    n_cand = long_primary.groupby(["trade_date", "gap_threshold_pct"]).size().rename("n_cand")
    long_primary = long_primary.merge(n_cand, on=["trade_date", "gap_threshold_pct"])

    binding = long_primary[long_primary["n_cand"] > 5]
    nonbinding = long_primary[long_primary["n_cand"] <= 5]

    lines.append(
        f"{len(nonbinding):,} of {len(long_primary):,} candidate-rows "
        f"({100 * len(nonbinding) / len(long_primary):.0f}%) are on days "
        f"with <=5 candidates, where the top-5 cutoff selects "
        f"*everything* (no actual filtering). The remaining "
        f"{len(binding):,} rows ({100 * len(binding) / len(long_primary):.0f}%) "
        f"are on days where the cutoff is genuinely binding."
    )
    lines.append("")

    lines.append("### On binding days: selected vs not-selected")
    lines.append("")
    g2 = binding.groupby("selected")["realized_return_pct"].agg(
        n="count", avg_ret="mean", sum_ret="sum",
        hit_rate=lambda x: (x > 0).mean() * 100,
    ).reset_index()
    lines.append(g2.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    drift_stats = binding.groupby("selected")["price_drift_pct"].describe()[["mean", "50%", "max"]]
    lines.append("price_drift_pct, selected vs not (binding days only):")
    lines.append("")
    lines.append(drift_stats.to_markdown(floatfmt=".2f"))
    lines.append("")
    lines.append(
        f"**This is the real evidence for the selection rule.** On the "
        f"{len(binding):,} binding-day candidates, selected trades "
        f"(mean drift {drift_stats.loc[True, 'mean']:.1f}%, median "
        f"{drift_stats.loc[True, '50%']:.1f}%) earn "
        f"{g2.loc[g2['selected'], 'sum_ret'].iloc[0]:.2f}% total "
        f"({g2.loc[g2['selected'], 'hit_rate'].iloc[0]:.1f}% hit), while "
        f"not-selected trades (mean drift "
        f"{drift_stats.loc[False, 'mean']:.1f}%, median "
        f"{drift_stats.loc[False, '50%']:.1f}%) lose "
        f"{g2.loc[~g2['selected'], 'sum_ret'].iloc[0]:.2f}% total "
        f"({g2.loc[~g2['selected'], 'hit_rate'].iloc[0]:.1f}% hit). On "
        f"non-binding days (everything auto-selected), sum_ret = "
        f"{nonbinding['realized_return_pct'].sum():.2f}%, hit = "
        f"{(nonbinding['realized_return_pct'] > 0).mean() * 100:.1f}%."
    )
    lines.append("")

    lines.append("## 5. Extreme positive-drift outliers excluded by selection")
    lines.append("")
    extreme = long_primary[(~long_primary["selected"]) & (long_primary["price_drift_pct"] > 100)]
    lines.append(
        f"{len(extreme)} candidate-rows have price_drift_pct > 100% and "
        f"were excluded. sum_ret = {extreme['realized_return_pct'].sum():.2f}%, "
        f"hit = {(extreme['realized_return_pct'] > 0).mean() * 100:.1f}%."
    )
    lines.append("")
    lines.append(extreme.sort_values("price_drift_pct", ascending=False)[cols].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append(
        "AKTS's +13,971% drift (2026-03-31) is almost certainly a data "
        "artifact (settlement-date price vs publication-date price off by "
        "~140x - likely a stock split not reflected in one of the two "
        "price series) rather than a real move; it happened to win (+3% "
        "TP) so doesn't hurt this group's numbers. CAR's +103% drift trade "
        "lost -45.57% on a $797 -> $434 move - also looks like an "
        "unadjusted-split artifact. These outliers are a small share (7 "
        "rows) of the 813 not-selected binding-day candidates and are not "
        "the main driver of section 4's result."
    )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "The simple story \"bigger pre-publication price drop -> bigger "
        "squeeze\" does NOT hold at the individual-trade level (near-zero "
        "correlation, mixed outcomes even among the most extreme drops). "
        "What the data does show is a **real, if noisy, separation on the "
        "subset of days where the top-5 cutoff actually binds**: candidates "
        "with deeply negative drift (median -3% on binding days) "
        "outperform candidates with flat/positive drift (median +1.3%) by "
        "a wide margin (+0.03% vs -1.02% avg return, 62% vs 45% hit rate). "
        "On the majority of days (48% of candidate-rows) the cutoff doesn't "
        "filter anything, so the strategy's overall positive sum_ret is a "
        "mix of (a) this binding-day selection effect (modestly positive, "
        "+18.8% sum_ret) and (b) the baseline edge of the LONG "
        "`hold_above_open_945_tp3` rule on low-candidate-count days "
        "(+247.4% sum_ret) - i.e. **most of the headline +266% comes from "
        "days where the rule barely filters anything**, not from the "
        "ranking mechanism itself. The ranking mechanism's incremental "
        "contribution (+18.8% vs what would otherwise be -828.6% on those "
        "binding days) is real but smaller in absolute terms than the "
        "non-binding-day baseline."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sanitycheck] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
