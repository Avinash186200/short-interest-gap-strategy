"""
report_topn_cooloff_joint.py — jointly re-tunes the drift_low top-N
selection size together with the cooling-off-after-a-loss control, since
both interact (a wider top-N gives cooling-off more candidates to rotate
into when a name is sidelined). Previously top_n=5 was fixed and
cooling-off was tuned separately on top of it (output/portfolio_controls.md).

Re-derives drift_low ranking directly from trades.parquet + drift_signal.parquet
(not the already-fixed-at-5 trades_selected.parquet) so top_n can vary from
1 to 10. 2.5%/3.0% gap thresholds processed separately throughout.

Train/test disciplined: (top_n, cooloff_days) pair chosen on TRAIN sum_ret,
evaluated ONCE on TEST.

Output: output/topn_cooloff_joint.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path("<local-path>")
OUT_MD = ROOT / "output" / "topn_cooloff_joint.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
LONG_RULE = "hold_above_open_945_tp3"
THRESHOLDS = [2.5, 3.0]
TOP_NS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
COOLOFFS = [0, 5, 10, 15, 20]

trades = pd.read_parquet(ROOT / "data" / "trades.parquet")
drift = pd.read_parquet(ROOT / "data" / "drift_signal.parquet")

df = trades.merge(
    drift[["settlementDate", "symbolCode", "price_drift_pct", "volume_ratio"]],
    left_on=["settlement_date", "ticker"],
    right_on=["settlementDate", "symbolCode"],
    how="left",
).drop(columns=["settlementDate", "symbolCode"])

df["trade_date"] = pd.to_datetime(df["trade_date"])

long_cand = df[(df["side"] == "LONG") & (df["rule_name"] == LONG_RULE) & df["price_drift_pct"].notna()].copy()
long_cand["rank"] = long_cand.groupby(["trade_date", "gap_threshold_pct"])["price_drift_pct"] \
    .rank(method="first", ascending=True)


def perf(s):
    n = len(s)
    hit = (s["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = s["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)


def apply_cooling_off(data, cooloff_days):
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    ticker_loss_date = {}
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]; tk = row["ticker"]
        last_loss = ticker_loss_date.get(tk)
        if last_loss is not None and (td - last_loss).days <= cooloff_days:
            keep.append(False)
        else:
            keep.append(True)
            if row["realized_return_pct"] < 0:
                ticker_loss_date[tk] = td
    data = data.copy(); data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")


def tbl(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        out.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(out)


lines = [
    "# Joint Re-Tuning: drift_low top-N x Cooling-off",
    "",
    "Previously top_n=5 was fixed (output/ranking_refine2_corrected.md) and "
    "cooling-off was tuned separately on top of it at 5d "
    "(output/portfolio_controls.md), giving TEST +111.07%/+110.32%. This "
    "jointly grid-searches (top_n, cooloff_days) together in case a "
    "different top_n is better once cooling-off is already active.",
    "",
    "Train/test disciplined: pair chosen on TRAIN sum_ret, evaluated ONCE "
    "on TEST. 2.5%/3.0% thresholds reported separately.",
    "",
]

summary = []
for thr in THRESHOLDS:
    pop_all = long_cand[long_cand["gap_threshold_pct"] == thr]
    results = []
    for n in TOP_NS:
        pop = pop_all[pop_all["rank"] <= n]
        for cd in COOLOFFS:
            f = pop if cd == 0 else apply_cooling_off(pop, cd)
            tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
            te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])
            results.append({"top_n": n, "cooloff": cd, "train_n": tr[0],
                             "train_hit": tr[1], "train_sum": tr[2],
                             "test_n": te[0], "test_hit": te[1], "test_sum": te[2]})
    grid_df = pd.DataFrame(results)
    best = grid_df.loc[grid_df["train_sum"].idxmax()]

    # reference: top_n=5, cooloff=5 (the previously-adopted combo)
    ref = grid_df[(grid_df["top_n"] == 5) & (grid_df["cooloff"] == 5)].iloc[0]

    lines.append(f"## Gap threshold {thr}%")
    lines.append("")
    lines.append("Full grid (TRAIN sum_ret heatmap-style table):")
    lines.append("")
    pivot_train = grid_df.pivot(index="top_n", columns="cooloff", values="train_sum")
    pivot_test = grid_df.pivot(index="top_n", columns="cooloff", values="test_sum")
    lines.append("**TRAIN sum_ret** (rows=top_n, cols=cooloff days)")
    lines.append("")
    lines.append(tbl(["top_n"] + [str(c) for c in COOLOFFS],
                      [[n] + [round(pivot_train.loc[n, c], 1) for c in COOLOFFS] for n in TOP_NS]))
    lines.append("")
    lines.append("**TEST sum_ret** (rows=top_n, cols=cooloff days)")
    lines.append("")
    lines.append(tbl(["top_n"] + [str(c) for c in COOLOFFS],
                      [[n] + [round(pivot_test.loc[n, c], 1) for c in COOLOFFS] for n in TOP_NS]))
    lines.append("")

    verdict = "PASS" if best["test_sum"] > ref["test_sum"] else "FAIL"
    lines += [
        f"**Previously-adopted (top_n=5, cooloff=5d):** TRAIN {ref['train_sum']:.2f}% -> "
        f"TEST {ref['test_sum']:.2f}%",
        "",
        f"**Joint TRAIN-best (top_n={int(best['top_n'])}, cooloff={int(best['cooloff'])}d):** "
        f"TRAIN {best['train_sum']:.2f}% -> TEST {best['test_sum']:.2f}% -> "
        f"**{verdict}** (vs the previously-adopted combo)",
        "", "---", "",
    ]
    summary.append((thr, ref, best))
    print(f"\n=== {thr}% ===")
    print(f"Previously-adopted (n=5,cool=5): TRAIN {ref['train_sum']:.2f}% -> TEST {ref['test_sum']:.2f}%")
    print(f"Joint TRAIN-best (n={int(best['top_n'])},cool={int(best['cooloff'])}): TRAIN {best['train_sum']:.2f}% -> TEST {best['test_sum']:.2f}%")

lines += [
    "## Conclusion", "",
    "The pure joint TRAIN-best pick differs by threshold (top_n=7 @2.5%, "
    "top_n=8 @3.0%) and is mixed on TEST: marginally better at 2.5% "
    "(+112.45% vs +111.07%, noise-level), clearly worse at 3.0% (+63.02% "
    "vs +110.32%) — neither threshold's pure TRAIN-best generalizes "
    "cleanly to the other, so the formal verdict is **inconclusive / not "
    "adopted**.",
    "",
    "Observation (not a formal result): **top_n=6** sits near the top of "
    "both TRAIN and TEST at BOTH thresholds simultaneously (TRAIN "
    "422.3/320.4, TEST 116.7/111.6 @2.5%/@3.0%) - more consistent across "
    "thresholds than the literal per-threshold argmax. This cross-"
    "threshold consistency is a useful robustness signal, but flagging "
    "top_n=6 this way is an eyeball observation on the full grid rather "
    "than a clean train-only selection, so treat it as a candidate worth "
    "re-checking once more data accumulates, not an adopted change.",
    "",
    "**Current top_n=5 remains a defensible, simple choice** - it's never "
    "far from the best cell in either threshold's TEST table.",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
