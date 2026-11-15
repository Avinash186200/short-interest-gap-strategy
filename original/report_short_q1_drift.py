"""
report_short_q1_drift.py — tests Expert's SHORT-side suggestion from the
2026-06 call: restrict SHORT (`failed_bounce_1000_tp_2.5`) entries to names
in the Q1 (most-negative) price_drift_pct quartile — stocks that have
"continued to fall" between settlement and publication, on the logic that
shorts can still borrow these and push them further down on a gap day.

Also tests the 10-day days-since-publication cap (Expert's second point)
on the SHORT side, alone and combined with the Q1 drift filter.

IMPORTANT: a trade that gaps >= 3.0% qualifies for BOTH the 2.5% and 3.0%
gap_threshold_pct buckets and appears as two rows with IDENTICAL
entry/exit/realized_return_pct (confirmed: 0 mismatches across 704 such
pairs). Per the project convention (filter_tests.py, portfolio_controls.py),
2.5% and 3.0% are always reported and verdicted SEPARATELY - never summed
into one combined number, which would double-count duplicated trades.

Train/test disciplined: any cut chosen on TRAIN (trade_date < 2026-01-22),
evaluated ONCE on TEST (last ~4 months). The literal Q1 boundary is
computed from the TRAIN drift distribution only (no leakage) and then
applied as a fixed ceiling to TEST.

Output: output/short_q1_drift_filter.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path("<local-path>")
OUT_MD = ROOT / "output" / "short_q1_drift_filter.md"

TRAIN_CUTOFF = "2026-01-22"
THRESHOLDS = [2.5, 3.0]
SHORT_RULE = "failed_bounce_1000_tp_2.5"

df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["publication_date"] = pd.to_datetime(df["publication_date"])

short_all = df[(df["side"] == "SHORT") & (df["rule_name"] == SHORT_RULE)].copy()
short_all = short_all.dropna(subset=["price_drift_pct", "realized_return_pct"])
short_all["days_since_pub"] = (short_all["trade_date"] - short_all["publication_date"]).dt.days
short_all["is_train"] = short_all["trade_date"] < TRAIN_CUTOFF


def perf(s):
    n = len(s)
    hit = (s["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = s["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)


def tbl(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        out.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(out)


lines = [
    "# SHORT Side: Q1 Drift Quartile Filter (Expert's suggestion)",
    "",
    "Tests restricting SHORT (`failed_bounce_1000_tp_2.5`) entries to names in "
    "the Q1 (most-negative) price_drift_pct quartile — i.e. stocks that kept "
    "falling between settlement and publication. Expert's logic: these names "
    "still have active short interest / available borrow, so a failed bounce "
    "is more likely to keep falling rather than squeeze.",
    "",
    "**2.5% and 3.0% gap thresholds are reported and verdicted SEPARATELY** "
    "— a trade gapping >=3.0% qualifies for both buckets and appears twice "
    "with identical P&L; summing them would double-count.",
    "",
    "Train/test disciplined: cuts chosen on TRAIN (trade_date < 2026-01-22), "
    "evaluated ONCE on TEST (last ~4 months). The literal Q1 boundary is "
    "computed separately per threshold from the TRAIN drift distribution.",
    "",
]

overall_verdicts = []

for thr in THRESHOLDS:
    short = short_all[short_all["gap_threshold_pct"] == thr]
    train_all = short[short["is_train"]]
    test_all = short[~short["is_train"]]
    base_train = perf(train_all)
    base_test = perf(test_all)

    q1_boundary = train_all["price_drift_pct"].quantile(0.25)
    q1_train = train_all[train_all["price_drift_pct"] <= q1_boundary]
    q1_test = test_all[test_all["price_drift_pct"] <= q1_boundary]

    ceilings = [-3, -5, -7, -10, -12, -15, -20, -25, -30]
    ceil_results = []
    for c in ceilings:
        tr = train_all[train_all["price_drift_pct"] <= c]
        te = test_all[test_all["price_drift_pct"] <= c]
        tr_n, tr_hit, tr_sum = perf(tr)
        te_n, te_hit, te_sum = perf(te)
        ceil_results.append({"ceiling": c, "train_n": tr_n, "train_hit": tr_hit,
                              "train_sum": tr_sum, "test_n": te_n, "test_hit": te_hit,
                              "test_sum": te_sum})
    ceil_df = pd.DataFrame(ceil_results)
    best_ceil_row = ceil_df.loc[ceil_df["train_sum"].idxmax()]
    best_ceil = best_ceil_row["ceiling"]

    pub_train = train_all[train_all["days_since_pub"] <= 10]
    pub_test = test_all[test_all["days_since_pub"] <= 10]

    combo_train = train_all[(train_all["price_drift_pct"] <= q1_boundary) & (train_all["days_since_pub"] <= 10)]
    combo_test = test_all[(test_all["price_drift_pct"] <= q1_boundary) & (test_all["days_since_pub"] <= 10)]

    combo2_train = train_all[(train_all["price_drift_pct"] <= best_ceil) & (train_all["days_since_pub"] <= 10)]
    combo2_test = test_all[(test_all["price_drift_pct"] <= best_ceil) & (test_all["days_since_pub"] <= 10)]

    lines += [
        f"## Gap threshold {thr}%",
        "",
        f"Q1 boundary (TRAIN): **{q1_boundary:.2f}%**",
        "",
        "### Baseline (no filter)",
        "",
        tbl(["", "n", "hit%", "sum%"],
            [("TRAIN", base_train[0], base_train[1], base_train[2]),
             ("TEST", base_test[0], base_test[1], base_test[2])]),
        "",
        "### Test A: literal Q1 cut",
        "",
        tbl(["", "n", "hit%", "sum%"],
            [("TRAIN", len(q1_train), perf(q1_train)[1], perf(q1_train)[2]),
             ("TEST", len(q1_test), perf(q1_test)[1], perf(q1_test)[2])]),
        "",
        f"Verdict: TEST {perf(q1_test)[2]:.2f}% vs baseline {base_test[2]:.2f}% -> "
        f"**{'PASS' if perf(q1_test)[2] > base_test[2] else 'FAIL'}**",
        "",
        "### Test B: range of fixed negative-drift ceilings",
        "",
    ]
    rows = []
    for _, r in ceil_df.iterrows():
        marker = " <- TRAIN best" if r["ceiling"] == best_ceil else ""
        rows.append((r["ceiling"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Ceiling", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    verdict_b = "PASS" if best_ceil_row["test_sum"] > base_test[2] else "FAIL"
    lines += [
        "",
        f"TRAIN-selected ceiling {best_ceil}% -> TEST {best_ceil_row['test_sum']:.2f}% "
        f"vs baseline {base_test[2]:.2f}% -> **{verdict_b}**",
        "",
        "### Test C: 10-day publication window cap (alone)",
        "",
        tbl(["", "n", "hit%", "sum%"],
            [("TRAIN", len(pub_train), perf(pub_train)[1], perf(pub_train)[2]),
             ("TEST", len(pub_test), perf(pub_test)[1], perf(pub_test)[2])]),
        "",
        f"Verdict: TEST {perf(pub_test)[2]:.2f}% vs baseline {base_test[2]:.2f}% -> "
        f"**{'PASS' if perf(pub_test)[2] > base_test[2] else 'FAIL'}**",
        "",
        "### Test D: literal Q1 + 10-day cap combined",
        "",
        tbl(["", "n", "hit%", "sum%"],
            [("TRAIN", len(combo_train), perf(combo_train)[1], perf(combo_train)[2]),
             ("TEST", len(combo_test), perf(combo_test)[1], perf(combo_test)[2])]),
        "",
        f"Verdict: TEST {perf(combo_test)[2]:.2f}% vs baseline {base_test[2]:.2f}% -> "
        f"**{'PASS' if perf(combo_test)[2] > base_test[2] else 'FAIL'}**",
        "",
        "### Test E: TRAIN-best ceiling + 10-day cap combined",
        "",
        tbl(["", "n", "hit%", "sum%"],
            [("TRAIN", len(combo2_train), perf(combo2_train)[1], perf(combo2_train)[2]),
             ("TEST", len(combo2_test), perf(combo2_test)[1], perf(combo2_test)[2])]),
        "",
        f"Verdict: TEST {perf(combo2_test)[2]:.2f}% vs baseline {base_test[2]:.2f}% -> "
        f"**{'PASS' if perf(combo2_test)[2] > base_test[2] else 'FAIL'}**",
        "",
        "---", "",
    ]

    overall_verdicts.append((thr, "Q1", perf(q1_test)[2] > base_test[2]))
    overall_verdicts.append((thr, "best ceiling", best_ceil_row["test_sum"] > base_test[2]))
    overall_verdicts.append((thr, "10d cap", perf(pub_test)[2] > base_test[2]))

    print(f"\n=== {thr}% ===")
    print(f"Baseline: TRAIN {base_train[2]:.2f}% (n={base_train[0]}) -> TEST {base_test[2]:.2f}% (n={base_test[0]})")
    print(f"Q1 boundary: {q1_boundary:.2f}%")
    print(f"Test A (Q1): TRAIN {perf(q1_train)[2]:.2f}% -> TEST {perf(q1_test)[2]:.2f}%")
    print(f"Test B (best ceiling={best_ceil}%): TRAIN {best_ceil_row['train_sum']:.2f}% -> TEST {best_ceil_row['test_sum']:.2f}%")
    print(f"Test C (10d cap): TRAIN {perf(pub_train)[2]:.2f}% -> TEST {perf(pub_test)[2]:.2f}%")
    print(f"Test D (Q1+10d): TRAIN {perf(combo_train)[2]:.2f}% -> TEST {perf(combo_test)[2]:.2f}%")
    print(f"Test E (best ceiling+10d): TRAIN {perf(combo2_train)[2]:.2f}% -> TEST {perf(combo2_test)[2]:.2f}%")

lines += [
    "## Conclusion", "",
    "**Caveat: 'PASS' means *less negative than baseline*, not profitable**, "
    "unless explicitly noted otherwise.",
    "",
]
for thr, label, v in overall_verdicts:
    lines.append(f"- {thr}% / {label}: {'PASS' if v else 'FAIL'}")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
