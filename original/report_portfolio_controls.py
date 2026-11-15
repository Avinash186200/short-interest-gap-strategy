"""
Portfolio-level risk controls on the drift_low top-5 LONG strategy.
Applied CHRONOLOGICALLY (each trade's eligibility depends on prior history),
unlike the per-trade filters that failed previously.

Controls tested:
  A. Ticker rolling frequency cap — max N trades per ticker in any 30-day
     rolling window. Prevents concentration in names that keep re-appearing.
  B. Ticker cooling-off after a loss — skip a ticker for X days after any
     losing trade on it.
  C. Combined A + B (best params from each).

Train/test disciplined: parameter chosen on TRAIN (trade_date < 2026-01-22),
evaluated ONCE on TEST (last ~4 months).

Output: output/portfolio_controls.md
"""
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT   = Path("<local-path>")
OUT_MD = ROOT / "output" / "portfolio_controls.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS   = [2.5, 3.0]

# ── load selected LONG drift_low top-5 trades ──────────────────────────────
df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])

sel = (df[(df["selected"]) & (df["side"] == "LONG")]
       .sort_values(["trade_date", "gap_threshold_pct", "ticker"])
       .reset_index(drop=True))

def perf(subset):
    n    = len(subset)
    hit  = (subset["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = subset["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)

def baselines():
    rows = {}
    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr]
        tr = sub[sub["trade_date"] < TRAIN_CUTOFF]
        te = sub[sub["trade_date"] >= TRAIN_CUTOFF]
        rows[thr] = {"train": perf(tr), "test": perf(te)}
    return rows

BASE = baselines()

# ══════════════════════════════════════════════════════════════════════════
# Helper: apply a portfolio-level filter chronologically
# Returns the filtered dataframe.
# ══════════════════════════════════════════════════════════════════════════
def apply_rolling_freq_cap(data, max_trades, window_days=30):
    """Keep a trade only if ticker has appeared < max_trades times in the
    last window_days calendar days (not counting the current trade)."""
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    ticker_dates = defaultdict(list)  # ticker -> sorted list of past trade dates
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]
        tk = row["ticker"]
        # count appearances in window
        cutoff = td - pd.Timedelta(days=window_days)
        past = [d for d in ticker_dates[tk] if d >= cutoff]
        if len(past) < max_trades:
            keep.append(True)
        else:
            keep.append(False)
        ticker_dates[tk].append(td)
    data = data.copy()
    data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")

def apply_cooling_off(data, cooloff_days):
    """Skip a ticker for cooloff_days calendar days after any losing trade."""
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    ticker_loss_date = {}  # ticker -> date of last loss
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]
        tk = row["ticker"]
        last_loss = ticker_loss_date.get(tk)
        if last_loss is not None and (td - last_loss).days <= cooloff_days:
            keep.append(False)
        else:
            keep.append(True)
            if row["realized_return_pct"] < 0:
                ticker_loss_date[tk] = td
    data = data.copy()
    data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")

# ══════════════════════════════════════════════════════════════════════════
# CONTROL A: rolling frequency cap
# ══════════════════════════════════════════════════════════════════════════
freq_caps    = [1, 2, 3, 4, 5]
window_days  = 30

freq_results = []
for thr in THRESHOLDS:
    sub = sel[sel["gap_threshold_pct"] == thr].copy()
    # baseline (no cap)
    tr = sub[sub["trade_date"] < TRAIN_CUTOFF]
    te = sub[sub["trade_date"] >= TRAIN_CUTOFF]
    freq_results.append({"cap": "none", "gap_thr": thr,
                         "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                         "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})
    for cap in freq_caps:
        filtered = apply_rolling_freq_cap(sub, max_trades=cap, window_days=window_days)
        tr = filtered[filtered["trade_date"] < TRAIN_CUTOFF]
        te = filtered[filtered["trade_date"] >= TRAIN_CUTOFF]
        freq_results.append({"cap": cap, "gap_thr": thr,
                             "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                             "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

freq_df = pd.DataFrame(freq_results)

# TRAIN-best per threshold
best_cap = {}
for thr in THRESHOLDS:
    sub = freq_df[freq_df["gap_thr"] == thr]
    best_cap[thr] = sub.loc[sub["train_sum"].idxmax(), "cap"]

# ══════════════════════════════════════════════════════════════════════════
# CONTROL B: cooling-off after a loss
# ══════════════════════════════════════════════════════════════════════════
cooloffs = [0, 5, 10, 15, 20, 30]

cool_results = []
for thr in THRESHOLDS:
    sub = sel[sel["gap_threshold_pct"] == thr].copy()
    for days in cooloffs:
        if days == 0:
            filtered = sub
        else:
            filtered = apply_cooling_off(sub, cooloff_days=days)
        tr = filtered[filtered["trade_date"] < TRAIN_CUTOFF]
        te = filtered[filtered["trade_date"] >= TRAIN_CUTOFF]
        cool_results.append({"cooloff": days, "gap_thr": thr,
                             "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                             "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

cool_df = pd.DataFrame(cool_results)

best_cooloff = {}
for thr in THRESHOLDS:
    sub = cool_df[cool_df["gap_thr"] == thr]
    best_cooloff[thr] = sub.loc[sub["train_sum"].idxmax(), "cooloff"]

# ══════════════════════════════════════════════════════════════════════════
# CONTROL C: combined (best cap + best cooloff, fixed reasonable combos)
# ══════════════════════════════════════════════════════════════════════════
combos = [
    ("none", None, 0),
    ("freq cap=2", 2, 0),
    ("freq cap=3", 3, 0),
    ("cooloff=10", None, 10),
    ("cooloff=20", None, 20),
    ("cap=2 + cooloff=10", 2, 10),
    ("cap=2 + cooloff=20", 2, 20),
    ("cap=3 + cooloff=10", 3, 10),
    ("cap=3 + cooloff=20", 3, 20),
]

combo_results = []
for thr in THRESHOLDS:
    sub = sel[sel["gap_threshold_pct"] == thr].copy()
    for label, cap, cooloff in combos:
        f = sub.copy()
        if cap is not None:
            f = apply_rolling_freq_cap(f, max_trades=cap, window_days=30)
        if cooloff > 0:
            f = apply_cooling_off(f, cooloff_days=cooloff)
        tr = f[f["trade_date"] < TRAIN_CUTOFF]
        te = f[f["trade_date"] >= TRAIN_CUTOFF]
        combo_results.append({"combo": label, "gap_thr": thr,
                              "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                              "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

combo_df = pd.DataFrame(combo_results)

# ══════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════
def tbl(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        out.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(out)

lines = [
    "# Portfolio-Level Risk Controls",
    "",
    "Unlike the per-trade filters (drift floor, days-since-pub, liquidity floor, "
    "stop-loss — all failed OOS), these controls are applied **chronologically**: "
    "each trade's eligibility depends on the prior history of trades in that ticker. "
    "Train/test disciplined: parameter chosen on TRAIN (< 2026-01-22), evaluated ONCE on TEST.",
    "",
    "## Baselines (no control)",
    "",
]
base_rows = []
for thr in THRESHOLDS:
    b = BASE[thr]
    tr, te = b["train"], b["test"]
    base_rows.append((f"{thr}%", tr[0], tr[1], tr[2], te[0], te[1], te[2]))
lines.append(tbl(["Gap thr", "TRAIN n", "TRAIN hit%", "TRAIN sum%", "TEST n", "TEST hit%", "TEST sum%"], base_rows))

lines += ["", "---", "",
          "## Control A: Ticker rolling frequency cap (30-day window)", "",
          "Max N trades per ticker in any 30-day rolling window. Prevents the same "
          "blowup name from re-appearing repeatedly (SKYQ had 4 trades in the drawdown "
          "window alone).", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = freq_df[freq_df["gap_thr"] == thr]
    rows = []
    for _, r in sub.iterrows():
        marker = " <- TRAIN best" if r["cap"] == best_cap[thr] else ""
        rows.append((r["cap"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Max per 30d", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    b_test = BASE[thr]["test"][2]
    bc = best_cap[thr]
    best_r = freq_df[(freq_df["gap_thr"] == thr) & (freq_df["cap"] == bc)].iloc[0]
    verdict = "PASS" if best_r["test_sum"] > b_test else "FAIL"
    lines += ["",
              f"**TRAIN-best cap={bc}** -> TEST {best_r['test_sum']:.2f}% vs baseline {b_test:.2f}% -> **{verdict}**",
              ""]

lines += ["---", "",
          "## Control B: Cooling-off period after a losing trade", "",
          "Skip a ticker for N calendar days after any losing trade on it. "
          "Tests whether momentum/streak avoidance helps.", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = cool_df[cool_df["gap_thr"] == thr]
    rows = []
    for _, r in sub.iterrows():
        marker = " <- TRAIN best" if r["cooloff"] == best_cooloff[thr] else ""
        rows.append((f"{int(r['cooloff'])}d", r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Cooling-off", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    b_test = BASE[thr]["test"][2]
    bc = best_cooloff[thr]
    best_r = cool_df[(cool_df["gap_thr"] == thr) & (cool_df["cooloff"] == bc)].iloc[0]
    verdict = "PASS" if best_r["test_sum"] > b_test else "FAIL"
    lines += ["",
              f"**TRAIN-best cooloff={bc}d** -> TEST {best_r['test_sum']:.2f}% vs baseline {b_test:.2f}% -> **{verdict}**",
              ""]

lines += ["---", "",
          "## Control C: Combined (freq cap + cooling-off)", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = combo_df[combo_df["gap_thr"] == thr]
    b_test = BASE[thr]["test"][2]
    rows = []
    for _, r in sub.iterrows():
        delta = f"{r['test_sum'] - b_test:+.2f}% vs baseline"
        rows.append((r["combo"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], delta))
    lines.append(tbl(["Combo", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "TEST vs baseline"], rows))
    lines.append("")

lines += ["---", "", "## Summary", ""]
for thr in THRESHOLDS:
    b_test = BASE[thr]["test"][2]
    # best on TEST for each control
    fsub = freq_df[freq_df["gap_thr"] == thr]
    best_f = fsub.loc[fsub["test_sum"].idxmax()]
    csub = cool_df[cool_df["gap_thr"] == thr]
    best_c = csub.loc[csub["test_sum"].idxmax()]
    xsub = combo_df[combo_df["gap_thr"] == thr]
    best_x = xsub.loc[xsub["test_sum"].idxmax()]

    lines += [
        f"**{thr}% — baseline TEST {b_test:.2f}%**",
        f"- Best freq cap on TEST: cap={best_f['cap']} -> {best_f['test_sum']:.2f}% "
        f"({'BETTER' if best_f['test_sum'] > b_test else 'WORSE'})",
        f"- Best cooloff on TEST: {int(best_c['cooloff'])}d -> {best_c['test_sum']:.2f}% "
        f"({'BETTER' if best_c['test_sum'] > b_test else 'WORSE'})",
        f"- Best combo on TEST: '{best_x['combo']}' -> {best_x['test_sum']:.2f}% "
        f"({'BETTER' if best_x['test_sum'] > b_test else 'WORSE'})",
        "",
    ]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")

# console summary
print("\n=== QUICK RESULTS ===")
print(f"Baselines: TEST {BASE[2.5]['test'][2]:.2f}% @2.5% | {BASE[3.0]['test'][2]:.2f}% @3.0%")
print("\nControl A (rolling freq cap, TRAIN-best -> TEST):")
for thr in THRESHOLDS:
    bc = best_cap[thr]
    r = freq_df[(freq_df["gap_thr"] == thr) & (freq_df["cap"] == bc)].iloc[0]
    b_test = BASE[thr]["test"][2]
    print(f"  @{thr}% cap={bc}: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}% "
          f"({'PASS' if r['test_sum'] > b_test else 'FAIL'})")
print("\nControl B (cooling-off, TRAIN-best -> TEST):")
for thr in THRESHOLDS:
    bc = best_cooloff[thr]
    r = cool_df[(cool_df["gap_thr"] == thr) & (cool_df["cooloff"] == bc)].iloc[0]
    b_test = BASE[thr]["test"][2]
    print(f"  @{thr}% cooloff={bc}d: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}% "
          f"({'PASS' if r['test_sum'] > b_test else 'FAIL'})")
print("\nBest combo on TEST:")
for thr in THRESHOLDS:
    xsub = combo_df[combo_df["gap_thr"] == thr]
    best_x = xsub.loc[xsub["test_sum"].idxmax()]
    b_test = BASE[thr]["test"][2]
    print(f"  @{thr}% '{best_x['combo']}': TEST {best_x['test_sum']:.2f}% "
          f"({'PASS' if best_x['test_sum'] > b_test else 'FAIL'})")
