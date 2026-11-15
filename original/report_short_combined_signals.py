"""
report_short_combined_signals.py — combines the three SHORT-side levers
tested so far (Q1 drift filter, cooling-off after a loss, Polygon
short-volume-ratio signal) to see if stacking gets SHORT to a profitable
OOS result. Each lever individually reduces the loss but does not flip it
positive (see short_q1_drift_filter.md, short_portfolio_controls.md,
polygon_short_volume_signal.md).

2.5% and 3.0% gap thresholds processed and reported SEPARATELY throughout
(duplicate-trade issue - see report_short_q1_drift.py docstring).

Train/test disciplined: every cut/parameter chosen on TRAIN, evaluated
ONCE on TEST.

Output: output/short_combined_signals.md
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("<local-path>")
SV_CACHE = ROOT / "data" / "polygon_short_volume.parquet"
OUT_MD = ROOT / "output" / "short_combined_signals.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
SHORT_RULE = "failed_bounce_1000_tp_2.5"
THRESHOLDS = [2.5, 3.0]

df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["publication_date"] = pd.to_datetime(df["publication_date"])

short_all = df[(df["side"] == "SHORT") & (df["rule_name"] == SHORT_RULE)].copy()
short_all = short_all.dropna(subset=["price_drift_pct", "realized_return_pct"])

sv = pd.read_parquet(SV_CACHE)
sv["date"] = pd.to_datetime(sv["date"])
sv = sv.sort_values(["ticker", "date"]).reset_index(drop=True)
sv["svr_30d_avg"] = sv.groupby("ticker")["short_volume_ratio"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=10).mean())
sv["svr_vs_hist"] = sv["short_volume_ratio"] - sv["svr_30d_avg"]

short_all = short_all.merge(
    sv[["ticker", "date", "svr_vs_hist"]],
    left_on=["ticker", "trade_date"], right_on=["ticker", "date"], how="left"
).drop(columns=["date"])


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
    "# SHORT Side: Combined Signal Stack (Q1 drift + cooling-off + Polygon SVR)",
    "",
    "Stacks the three SHORT-side levers tested individually (each reduces "
    "the loss but doesn't flip it positive alone): Q1 drift quartile filter, "
    "cooling-off after a loss (15d), and Polygon short_volume_ratio vs its "
    "own 30-day average. 2.5%/3.0% thresholds processed separately "
    "(duplicate-trade issue).",
    "",
]

summary = []
for thr in THRESHOLDS:
    pop = short_all[short_all["gap_threshold_pct"] == thr].copy()
    pop["is_train"] = pop["trade_date"] < TRAIN_CUTOFF
    train_all = pop[pop["is_train"]]
    test_all = pop[~pop["is_train"]]
    base_train = perf(train_all)
    base_test = perf(test_all)

    q1_boundary = train_all["price_drift_pct"].quantile(0.25)

    # Step 1: Q1 drift filter
    step1 = pop[pop["price_drift_pct"] <= q1_boundary]
    # Step 2: + cooling-off 15d (chronological, on the Q1-filtered population)
    step2 = apply_cooling_off(step1, 15)
    # Step 3: + Polygon SVR vs hist >= 3pp (where available)
    step3 = step2[step2["svr_vs_hist"].fillna(-999) >= 3.0]

    def split(d):
        return d[d["trade_date"] < TRAIN_CUTOFF], d[d["trade_date"] >= TRAIN_CUTOFF]

    s1_tr, s1_te = split(step1)
    s2_tr, s2_te = split(step2)
    s3_tr, s3_te = split(step3)

    rows = [
        ("baseline", *perf(train_all), *perf(test_all)),
        ("+ Q1 drift filter", *perf(s1_tr), *perf(s1_te)),
        ("+ cooling-off 15d", *perf(s2_tr), *perf(s2_te)),
        ("+ Polygon SVR>=3pp", *perf(s3_tr), *perf(s3_te)),
    ]
    lines += [f"## Gap threshold {thr}%", "", f"Q1 boundary (TRAIN): {q1_boundary:.2f}%", "",
              tbl(["Step", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                   "TEST n", "TEST hit%", "TEST sum%"], rows), ""]
    final_test = perf(s3_te)[2]
    verdict = "PASS" if final_test > base_test[2] else "FAIL"
    lines += [f"Final stack -> TEST {final_test:.2f}% vs baseline {base_test[2]:.2f}% -> **{verdict}**",
              "", "---", ""]
    summary.append((thr, base_test[2], final_test))

    print(f"\n=== {thr}% ===")
    print(f"baseline: TEST {base_test[2]:.2f}%")
    print(f"+ Q1: TRAIN {perf(s1_tr)[2]:.2f}% -> TEST {perf(s1_te)[2]:.2f}% (n={perf(s1_te)[0]})")
    print(f"+ cooloff15: TRAIN {perf(s2_tr)[2]:.2f}% -> TEST {perf(s2_te)[2]:.2f}% (n={perf(s2_te)[0]})")
    print(f"+ SVR>=3pp: TRAIN {perf(s3_tr)[2]:.2f}% -> TEST {perf(s3_te)[2]:.2f}% (n={perf(s3_te)[0]})")

lines += ["## Conclusion", ""]
for thr, base, final in summary:
    lines.append(f"- {thr}%: baseline TEST {base:.2f}% -> full stack TEST {final:.2f}%")

lines += [
    "",
    "**Caveat — sample size is too small to trust.** The full stack leaves "
    "only 16-21 TEST trades per threshold (down from 146-197 at baseline). "
    "Three sequential filters were applied (Q1 drift, cooling-off, Polygon "
    "SVR), each individually chosen on TRAIN, but stacking them narrows the "
    "surviving population enough that the near-breakeven TEST result could "
    "easily be noise rather than a real edge — a handful of trades flipping "
    "either way would change the sign. This is a promising DIRECTION (every "
    "lever points the same way: less negative) but not a validated result. "
    "Would need a longer history or a wider universe to get a TEST sample "
    "large enough to trust before treating SHORT as solved.",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
