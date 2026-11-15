"""
report_threshold_overlap.py — integrity check: how independent are the
2.5% and 3.0% gap-threshold results actually? Every 3.0% candidate also
satisfies the 2.5% gap condition (gap_pct >= 3.0 implies gap_pct >= 2.5),
so the two thresholds' top-5 selections are NOT independent samples - this
checks how much they overlap, since "robust at both thresholds" has been
used throughout this investigation as a robustness signal and that
signal's strength depends on how independent the two populations are.

Not a strategy change - a finding about how to interpret existing results.

Output: output/threshold_overlap.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path("<local-path>")
OUT_MD = ROOT / "output" / "threshold_overlap.md"

df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()

t25 = sel[sel["gap_threshold_pct"] == 2.5].copy()
t30 = sel[sel["gap_threshold_pct"] == 3.0].copy()

key25 = set(zip(t25["trade_date"], t25["ticker"]))
key30 = set(zip(t30["trade_date"], t30["ticker"]))
overlap = key25 & key30

t30["key"] = list(zip(t30["trade_date"], t30["ticker"]))
t30["is_overlap"] = t30["key"].isin(overlap)

pct_30_in_25 = len(overlap) / len(key30) * 100
pct_25_in_30 = len(overlap) / len(key25) * 100

overlap_sum = t30[t30["is_overlap"]]["realized_return_pct"].sum()
unique_sum = t30[~t30["is_overlap"]]["realized_return_pct"].sum()
overlap_n = int(t30["is_overlap"].sum())
unique_n = int((~t30["is_overlap"]).sum())

lines = [
    "# Integrity Check: 2.5% vs 3.0% Gap Threshold Overlap",
    "",
    "Every trade satisfying the 3.0% gap threshold also satisfies the 2.5% "
    "threshold (gap_pct >= 3.0 implies gap_pct >= 2.5) - the two thresholds "
    "are nested, not independent. This checks how much the two reported "
    "strategies (`+168.07% @2.5%` / `+98.07% @3.0%`) actually overlap at "
    "the trade level, since this investigation has repeatedly used "
    "\"works at both thresholds\" as a robustness signal (e.g. drift_low "
    "top-N, the joint top_n/cooloff re-tune) — that signal is weaker than "
    "it looks if the two samples are mostly the same trades.",
    "",
    f"- 2.5% selected trades: {len(t25)} ({len(key25)} unique date/ticker pairs)",
    f"- 3.0% selected trades: {len(t30)} ({len(key30)} unique date/ticker pairs)",
    f"- Overlapping date/ticker pairs: {len(overlap)}",
    f"- **{pct_30_in_25:.1f}% of the 3.0% population's (date,ticker) keys also "
    f"appear in the 2.5% population.**",
    f"- {pct_25_in_30:.1f}% of the 2.5% population's keys also appear in 3.0%.",
    "",
    "## Sum_ret decomposition of the 3.0% headline result",
    "",
    f"- From trades that ALSO appear in 2.5% (n={overlap_n}): {overlap_sum:.2f}%",
    f"- From trades UNIQUE to 3.0% (n={unique_n}): {unique_sum:.2f}%",
    f"- Total: {overlap_sum + unique_sum:.2f}%",
    "",
    "## Interpretation",
    "",
    "**This is not a bug** — the headline LONG report correctly presents "
    "2.5% and 3.0% as two separate, mutually-exclusive deployment choices "
    "(never summed together as if running both simultaneously on the same "
    "capital). The issue is purely about *how much independent evidence* "
    "a result that 'holds at both thresholds' actually represents:",
    "",
    f"With ~{pct_30_in_25:.0f}% of the 3.0% trades being literally the same "
    "trade as a 2.5% trade (same entry, same exit, same return), a finding "
    "that's positive at both thresholds is much closer to ONE piece of "
    "evidence checked twice than two independent confirmations. Any future "
    "write-up describing a result as 'robust across both gap thresholds' "
    "should note this overlap rather than implying two separate validation "
    "samples.",
    "",
    "**Practical implication for deployment:** since a live system would "
    "pick ONE gap threshold (not run both), and 2.5% has both the larger "
    "trade count and the larger full-sample edge (+168.07% vs +98.07%), "
    "2.5% is the natural default — the 3.0% number mostly restates a "
    "subset of the same evidence at lower n, it isn't a genuinely larger "
    "or different opportunity set.",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")
print(f"\n{pct_30_in_25:.1f}% of 3.0% trades also appear in 2.5% population")
print(f"3.0% headline sum_ret {overlap_sum+unique_sum:.2f}%: overlap contributes {overlap_sum:.2f}% (n={overlap_n}), unique contributes {unique_sum:.2f}% (n={unique_n})")
