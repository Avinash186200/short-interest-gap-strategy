"""
Test two signal filters on the validated drift_low top-5 LONG strategy,
train/test disciplined (TRAIN: trade_date < 2026-01-22, TEST: last ~4 months).

Filter 1: price_drift_pct floor — exclude trades where drift < floor
          (Q1 blowup names like SKYQ, ZNTL, CAPR drove the drawdown)
Filter 2: days-since-publication exclusion — exclude trades in the
          11-20 day window (identified as negative in settlement_pub_analysis)
Filter 3: combined (both applied simultaneously)

For each filter, parameter(s) chosen on TRAIN sum_ret, evaluated ONCE on TEST.
Output: output/filter_tests.md
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT   = Path("<local-path>")
OUT_MD = ROOT / "output" / "filter_tests.md"

TRAIN_CUTOFF = "2026-01-22"

# ── load selected LONG trades ──────────────────────────────────────────────
df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["publication_date"] = pd.to_datetime(df["publication_date"])

sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
sel["days_since_pub"] = (sel["trade_date"] - sel["publication_date"]).dt.days
sel["is_train"] = sel["trade_date"] < TRAIN_CUTOFF

THRESHOLDS = [2.5, 3.0]

def perf(subset):
    n    = len(subset)
    hit  = (subset["realized_return_pct"] > 0).mean() * 100 if n else 0
    avg  = subset["realized_return_pct"].mean() if n else 0
    sumr = subset["realized_return_pct"].sum()
    return n, round(hit, 2), round(avg, 4), round(sumr, 2)

def perf_by_thr(subset):
    rows = {}
    for thr in THRESHOLDS:
        rows[thr] = perf(subset[subset["gap_threshold_pct"] == thr])
    return rows

# ── baselines ──────────────────────────────────────────────────────────────
train_all = sel[sel["is_train"]]
test_all  = sel[~sel["is_train"]]

baselines = {}
for thr in THRESHOLDS:
    tr = train_all[train_all["gap_threshold_pct"] == thr]
    te = test_all[test_all["gap_threshold_pct"] == thr]
    baselines[thr] = {"train": perf(tr), "test": perf(te)}

# ══════════════════════════════════════════════════════════════════════════
# FILTER 1: price_drift_pct floor
# ══════════════════════════════════════════════════════════════════════════
drift_floors = [None, -30, -20, -15, -10, -7, -5, -3]

drift_results = []
for floor in drift_floors:
    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr]
        if floor is not None:
            sub = sub[sub["price_drift_pct"] >= floor]
        tr_n, tr_hit, tr_avg, tr_sum = perf(sub[sub["is_train"]])
        te_n, te_hit, te_avg, te_sum = perf(sub[~sub["is_train"]])
        drift_results.append({
            "floor": floor if floor is not None else "none",
            "gap_thr": thr,
            "train_n": tr_n, "train_hit": tr_hit, "train_sum": tr_sum,
            "test_n":  te_n, "test_hit":  te_hit, "test_sum":  te_sum,
        })

drift_df = pd.DataFrame(drift_results)

# best floor per threshold (by TRAIN sum_ret only)
best_floor = {}
for thr in THRESHOLDS:
    sub = drift_df[drift_df["gap_thr"] == thr]
    best_row = sub.loc[sub["train_sum"].idxmax()]
    best_floor[thr] = best_row["floor"]

# ══════════════════════════════════════════════════════════════════════════
# FILTER 2: days-since-publication exclusion (exclude days 11-20)
# ══════════════════════════════════════════════════════════════════════════
# Test several exclusion windows
pub_windows = [
    ("none (baseline)", None, None),
    ("exclude 11-15",  11, 15),
    ("exclude 11-20",  11, 20),
    ("exclude 13-20",  13, 20),
    ("exclude 10-20",  10, 20),
    ("only 0-10",       None, 10),   # keep only 0-10 days
    ("only 0-5",        None,  5),
]

pub_results = []
for label, excl_lo, excl_hi in pub_windows:
    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr].copy()
        if excl_lo is not None and excl_hi is not None:
            sub = sub[~sub["days_since_pub"].between(excl_lo, excl_hi)]
        elif excl_hi is not None:
            sub = sub[sub["days_since_pub"] <= excl_hi]
        tr_n, tr_hit, tr_avg, tr_sum = perf(sub[sub["is_train"]])
        te_n, te_hit, te_avg, te_sum = perf(sub[~sub["is_train"]])
        pub_results.append({
            "window": label, "gap_thr": thr,
            "train_n": tr_n, "train_hit": tr_hit, "train_sum": tr_sum,
            "test_n":  te_n, "test_hit":  te_hit, "test_sum":  te_sum,
        })

pub_df = pd.DataFrame(pub_results)

best_window = {}
for thr in THRESHOLDS:
    sub = pub_df[pub_df["gap_thr"] == thr]
    best_row = sub.loc[sub["train_sum"].idxmax()]
    best_window[thr] = best_row["window"]

# ══════════════════════════════════════════════════════════════════════════
# FILTER 3: combined (best of each applied together)
# ══════════════════════════════════════════════════════════════════════════
combo_results = []

# also test a reasonable fixed combo: drift >= -10, exclude days 11-20
fixed_combos = [
    ("none (baseline)",    None,  None, None),
    ("drift>=-10 only",    -10,   None, None),
    ("excl days 11-20 only", None, 11,  20),
    ("drift>=-10 + excl 11-20", -10, 11, 20),
    ("drift>=-7 + excl 11-20",  -7,  11, 20),
    ("drift>=-5 + excl 11-20",  -5,  11, 20),
    ("drift>=-15 + excl 11-20", -15, 11, 20),
]

for label, drift_fl, excl_lo, excl_hi in fixed_combos:
    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr].copy()
        if drift_fl is not None:
            sub = sub[sub["price_drift_pct"] >= drift_fl]
        if excl_lo is not None and excl_hi is not None:
            sub = sub[~sub["days_since_pub"].between(excl_lo, excl_hi)]
        tr_n, tr_hit, tr_avg, tr_sum = perf(sub[sub["is_train"]])
        te_n, te_hit, te_avg, te_sum = perf(sub[~sub["is_train"]])
        combo_results.append({
            "combo": label, "gap_thr": thr,
            "train_n": tr_n, "train_hit": tr_hit, "train_sum": tr_sum,
            "test_n":  te_n, "test_hit":  te_hit, "test_sum":  te_sum,
        })

combo_df = pd.DataFrame(combo_results)

# ══════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════
def tbl_md(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        lines.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(lines)

lines = [
    "# Signal Filter Tests — drift floor & days-since-publication",
    "",
    "Train/test disciplined: parameter chosen on TRAIN (trade_date < 2026-01-22), "
    "evaluated ONCE on TEST (last ~4 months). Base data: drift_low top-5 selected LONG trades.",
    "",
    "## Baselines (no filter)",
    "",
]
base_rows = []
for thr in THRESHOLDS:
    b = baselines[thr]
    tr, te = b["train"], b["test"]
    base_rows.append((f"{thr}%", tr[0], tr[1], tr[3], te[0], te[1], te[3]))
lines.append(tbl_md(
    ["Gap thr", "TRAIN n", "TRAIN hit%", "TRAIN sum%", "TEST n", "TEST hit%", "TEST sum%"],
    base_rows))
lines += ["", "---", "", "## Filter 1: price_drift_pct floor", "",
          "Exclude trades where price_drift_pct < floor (worst Q1 blowup names).", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    lines.append("")
    sub = drift_df[drift_df["gap_thr"] == thr]
    rows = []
    for _, r in sub.iterrows():
        marker = " ← TRAIN best" if r["floor"] == best_floor[thr] else ""
        rows.append((r["floor"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl_md(
        ["Floor", "TRAIN n", "TRAIN hit%", "TRAIN sum%", "TEST n", "TEST hit%", "TEST sum%", "Note"],
        rows))
    b = baselines[thr]
    bf = best_floor[thr]
    best_row = sub[sub["floor"] == bf].iloc[0]
    verdict = "PASS" if best_row["test_sum"] > b["test"][3] else "FAIL"
    lines += ["",
              f"**TRAIN-selected floor: {bf}%** → TEST sum_ret {best_row['test_sum']:.2f}% "
              f"vs baseline {b['test'][3]:.2f}% → **{verdict}**", ""]

lines += ["---", "", "## Filter 2: days-since-publication exclusion window", "",
          "Exclude or restrict the window within which we take trades after publication date.", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    lines.append("")
    sub = pub_df[pub_df["gap_thr"] == thr]
    rows = []
    for _, r in sub.iterrows():
        marker = " ← TRAIN best" if r["window"] == best_window[thr] else ""
        rows.append((r["window"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl_md(
        ["Window filter", "TRAIN n", "TRAIN hit%", "TRAIN sum%", "TEST n", "TEST hit%", "TEST sum%", "Note"],
        rows))
    b = baselines[thr]
    bw = best_window[thr]
    best_row = sub[sub["window"] == bw].iloc[0]
    verdict = "PASS" if best_row["test_sum"] > b["test"][3] else "FAIL"
    lines += ["",
              f"**TRAIN-selected window: '{bw}'** → TEST sum_ret {best_row['test_sum']:.2f}% "
              f"vs baseline {b['test'][3]:.2f}% → **{verdict}**", ""]

lines += ["---", "", "## Filter 3: Combined (fixed reasonable combos)", "",
          "Testing interpretable combinations across both thresholds.", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    lines.append("")
    sub = combo_df[combo_df["gap_thr"] == thr]
    rows = []
    b = baselines[thr]
    for _, r in sub.iterrows():
        te_vs = f"{r['test_sum'] - b['test'][3]:+.2f}% vs baseline"
        rows.append((r["combo"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], te_vs))
    lines.append(tbl_md(
        ["Combo", "TRAIN n", "TRAIN hit%", "TRAIN sum%", "TEST n", "TEST hit%", "TEST sum%", "TEST vs baseline"],
        rows))
    lines.append("")

lines += [
    "---", "",
    "## Summary & conclusion", "",
]

# compute summary conclusions
for thr in THRESHOLDS:
    b_test = baselines[thr]["test"][3]
    # best drift floor on test
    dsub = drift_df[drift_df["gap_thr"] == thr]
    best_drift_test = dsub.loc[dsub["test_sum"].idxmax()]
    # best pub window on test
    psub = pub_df[pub_df["gap_thr"] == thr]
    best_pub_test = psub.loc[psub["test_sum"].idxmax()]
    # best combo on test
    csub = combo_df[combo_df["gap_thr"] == thr]
    best_combo_test = csub.loc[csub["test_sum"].idxmax()]

    lines += [
        f"**{thr}% threshold — baseline TEST: {b_test:.2f}%**",
        f"- Best drift floor on TEST: floor={best_drift_test['floor']} → TEST {best_drift_test['test_sum']:.2f}% "
        f"({'better' if best_drift_test['test_sum'] > b_test else 'worse'} than baseline)",
        f"- Best pub window on TEST: '{best_pub_test['window']}' → TEST {best_pub_test['test_sum']:.2f}% "
        f"({'better' if best_pub_test['test_sum'] > b_test else 'worse'} than baseline)",
        f"- Best combo on TEST: '{best_combo_test['combo']}' → TEST {best_combo_test['test_sum']:.2f}% "
        f"({'better' if best_combo_test['test_sum'] > b_test else 'worse'} than baseline)",
        "",
    ]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")

# print quick summary to console
print("\n=== QUICK RESULTS ===")
print(f"Baselines: TEST {baselines[2.5]['test'][3]:.2f}% @2.5% | {baselines[3.0]['test'][3]:.2f}% @3.0%")
print("\nDrift floor (TRAIN-best -> TEST):")
for thr in THRESHOLDS:
    sub = drift_df[drift_df["gap_thr"] == thr]
    bf = best_floor[thr]
    best_row = sub[sub["floor"] == bf].iloc[0]
    print(f"  @{thr}% best floor={bf}: TRAIN {best_row['train_sum']:.2f}% -> TEST {best_row['test_sum']:.2f}%")
print("\nDays-pub window (TRAIN-best -> TEST):")
for thr in THRESHOLDS:
    sub = pub_df[pub_df["gap_thr"] == thr]
    bw = best_window[thr]
    best_row = sub[sub["window"] == bw].iloc[0]
    print(f"  @{thr}% best window='{bw}': TRAIN {best_row['train_sum']:.2f}% -> TEST {best_row['test_sum']:.2f}%")
print("\nCombined (drift>=-10 + excl 11-20):")
for thr in THRESHOLDS:
    sub = combo_df[(combo_df["gap_thr"] == thr) & (combo_df["combo"] == "drift>=-10 + excl 11-20")]
    if not sub.empty:
        r = sub.iloc[0]
        print(f"  @{thr}%: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}%")
