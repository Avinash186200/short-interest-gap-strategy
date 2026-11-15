"""
Polygon daily short-volume signal for SHORT side.
Fetches /stocks/v1/short-volume for all 284 SHORT tickers (2024-01-01 ->
2026-05-31), caches to data/polygon_short_volume.parquet, then tests whether
the short_volume_ratio at trade_date (or recent trend) is a useful filter for
the SHORT failed_bounce_1000_tp_2.5 rule.

Signal ideas tested:
  S1. short_volume_ratio on trade_date > threshold (elevated shorting = confirm SHORT)
  S2. short_volume_ratio trend: ratio on trade_date vs 5-day prior avg
      (rising = shorts accumulating = confirm SHORT)
  S3. short_volume_ratio vs ticker's own 30-day rolling avg (elevated vs history)

Train/test disciplined: threshold chosen on TRAIN (< 2026-01-22), evaluated on TEST.
Output: output/polygon_short_volume_signal.md
"""
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import time

ROOT   = Path("<local-path>")
CACHE  = ROOT / "data" / "polygon_short_volume.parquet"
OUT_MD = ROOT / "output" / "polygon_short_volume_signal.md"

KEY = "<REDACTED>"
BASE_URL = "https://api.polygon.io/stocks/v1/short-volume"

TRAIN_CUTOFF = pd.Timestamp("2026-01-22")

# ── load SHORT trades ──────────────────────────────────────────────────────
df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])

shorts = df[
    (df["side"] == "SHORT") &
    (df["rule_name"] == "failed_bounce_1000_tp_2.5") &
    (df["gap_threshold_pct"] == 2.5)
].copy()
tickers = sorted(shorts["ticker"].unique())
print(f"SHORT tickers: {len(tickers)}, trades: {len(shorts)}")

# ── fetch / load cache ─────────────────────────────────────────────────────
if CACHE.exists():
    sv = pd.read_parquet(CACHE)
    cached_tickers = set(sv["ticker"].unique())
    missing = [t for t in tickers if t not in cached_tickers]
    print(f"Cache exists: {len(cached_tickers)} tickers cached, {len(missing)} missing")
else:
    sv = pd.DataFrame()
    missing = tickers
    print(f"No cache, fetching all {len(missing)} tickers")

def fetch_ticker(ticker):
    rows = []
    url = BASE_URL
    params = {
        "ticker": ticker,
        "date.gte": "2024-01-01",
        "date.lte": "2026-05-31",
        "limit": 50000,
        "apiKey": KEY,
    }
    resp = requests.get(url, params=params, timeout=30, verify=False)
    if resp.status_code != 200:
        return pd.DataFrame()
    data = resp.json().get("results", [])
    for rec in data:
        # short_volume_ratio is returned as a PERCENTAGE (0-100), e.g. 60.19 = 60.19%
        rows.append({
            "ticker": ticker,
            "date": pd.to_datetime(rec.get("date")),
            "short_volume": rec.get("short_volume", 0),
            "total_volume": rec.get("total_volume", 0),
            "short_volume_ratio": float(rec.get("short_volume_ratio", 0)),  # 0-100 scale
        })
    return pd.DataFrame(rows)

if missing:
    print(f"Fetching {len(missing)} tickers...")
    new_frames = []
    for i, tk in enumerate(missing):
        frame = fetch_ticker(tk)
        if not frame.empty:
            new_frames.append(frame)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(missing)} done")
        time.sleep(0.05)  # gentle rate limit
    if new_frames:
        new_sv = pd.concat(new_frames, ignore_index=True)
        sv = pd.concat([sv, new_sv], ignore_index=True) if not sv.empty else new_sv
        sv.to_parquet(CACHE, index=False)
        print(f"Saved {len(sv)} rows to cache")
    else:
        print("No new data fetched")

if sv.empty:
    print("No short-volume data available - cannot run signal test")
    exit()

sv["date"] = pd.to_datetime(sv["date"])
print(f"Short-volume data: {len(sv)} rows, {sv['ticker'].nunique()} tickers, "
      f"{sv['date'].min().date()} -> {sv['date'].max().date()}")

# ── compute signal features ────────────────────────────────────────────────
sv = sv.sort_values(["ticker", "date"]).reset_index(drop=True)
sv["svr_5d_avg"]   = sv.groupby("ticker")["short_volume_ratio"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=2).mean())
sv["svr_30d_avg"]  = sv.groupby("ticker")["short_volume_ratio"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=10).mean())
sv["svr_trend"]    = sv["short_volume_ratio"] - sv["svr_5d_avg"]   # + = rising
sv["svr_vs_hist"]  = sv["short_volume_ratio"] - sv["svr_30d_avg"]  # + = elevated vs history

# ── merge onto SHORT trades ────────────────────────────────────────────────
shorts = shorts.merge(
    sv[["ticker", "date", "short_volume_ratio", "svr_5d_avg", "svr_30d_avg",
        "svr_trend", "svr_vs_hist"]],
    left_on=["ticker", "trade_date"],
    right_on=["ticker", "date"],
    how="left"
)
coverage = shorts["short_volume_ratio"].notna().mean() * 100
print(f"Signal coverage: {coverage:.1f}% of SHORT trades have short_volume_ratio on trade_date")

THRESHOLDS = [2.5, 3.0]

def perf(subset):
    n    = len(subset)
    hit  = (subset["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = subset["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)

# baselines per threshold
baselines = {}
for thr in THRESHOLDS:
    sub = df[(df["side"] == "SHORT") & (df["rule_name"] == "failed_bounce_1000_tp_2.5") &
             (df["gap_threshold_pct"] == thr)].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    tr = sub[sub["trade_date"] < TRAIN_CUTOFF]
    te = sub[sub["trade_date"] >= TRAIN_CUTOFF]
    baselines[thr] = {"train": perf(tr), "test": perf(te), "all": sub}

# ── Signal S1: short_volume_ratio > threshold ──────────────────────────────
s1_thresholds = [40, 45, 48, 50, 52, 55, 60]  # short_volume_ratio is 0-100 scale
s1_results = []

for thr in THRESHOLDS:
    sub = df[(df["side"] == "SHORT") & (df["rule_name"] == "failed_bounce_1000_tp_2.5") &
             (df["gap_threshold_pct"] == thr)].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub = sub.merge(
        sv[["ticker", "date", "short_volume_ratio", "svr_trend", "svr_vs_hist"]],
        left_on=["ticker", "trade_date"], right_on=["ticker", "date"], how="left"
    )
    for svr_thr in s1_thresholds:
        f = sub[sub["short_volume_ratio"] >= svr_thr]
        tr, te = f[f["trade_date"] < TRAIN_CUTOFF], f[f["trade_date"] >= TRAIN_CUTOFF]
        s1_results.append({"svr_min": svr_thr, "gap_thr": thr,
                           "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                           "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})
    # no filter row
    tr, te = sub[sub["trade_date"] < TRAIN_CUTOFF], sub[sub["trade_date"] >= TRAIN_CUTOFF]
    s1_results.append({"svr_min": "none", "gap_thr": thr,
                       "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                       "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

s1_df = pd.DataFrame(s1_results)

# ── Signal S2: rising SVR (trend > threshold) ─────────────────────────────
s2_thresholds = [0.0, 0.5, 1.0, 2.0, 3.0]  # trend in pct-point terms (0-100 scale)
s2_results = []

for thr in THRESHOLDS:
    sub = df[(df["side"] == "SHORT") & (df["rule_name"] == "failed_bounce_1000_tp_2.5") &
             (df["gap_threshold_pct"] == thr)].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub = sub.merge(
        sv[["ticker", "date", "svr_trend"]],
        left_on=["ticker", "trade_date"], right_on=["ticker", "date"], how="left"
    )
    for trend_thr in s2_thresholds:
        f = sub[sub["svr_trend"] >= trend_thr]
        tr, te = f[f["trade_date"] < TRAIN_CUTOFF], f[f["trade_date"] >= TRAIN_CUTOFF]
        s2_results.append({"trend_min": trend_thr, "gap_thr": thr,
                           "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                           "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

s2_df = pd.DataFrame(s2_results)

# ── Signal S3: elevated vs own 30d history ────────────────────────────────
s3_thresholds = [0.0, 0.5, 1.0, 2.0, 3.0]  # vs 30d avg, in pct-point terms
s3_results = []

for thr in THRESHOLDS:
    sub = df[(df["side"] == "SHORT") & (df["rule_name"] == "failed_bounce_1000_tp_2.5") &
             (df["gap_threshold_pct"] == thr)].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub = sub.merge(
        sv[["ticker", "date", "svr_vs_hist"]],
        left_on=["ticker", "trade_date"], right_on=["ticker", "date"], how="left"
    )
    for vs_thr in s3_thresholds:
        f = sub[sub["svr_vs_hist"] >= vs_thr]
        tr, te = f[f["trade_date"] < TRAIN_CUTOFF], f[f["trade_date"] >= TRAIN_CUTOFF]
        s3_results.append({"vs_hist_min": vs_thr, "gap_thr": thr,
                           "train_n": perf(tr)[0], "train_hit": perf(tr)[1], "train_sum": perf(tr)[2],
                           "test_n":  perf(te)[0], "test_hit":  perf(te)[1], "test_sum":  perf(te)[2]})

s3_df = pd.DataFrame(s3_results)

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
    "# Polygon Daily Short-Volume Signal — SHORT Side",
    "",
    f"Data: Polygon `/stocks/v1/short-volume` — daily FINRA short_volume_ratio "
    f"(T+1, history from ~2024-02). Coverage: **{coverage:.1f}%** of SHORT trades "
    f"have a matching short_volume_ratio on their trade_date.",
    "",
    "Three signals tested on the SHORT `failed_bounce_1000_tp_2.5` rule, "
    "train/test disciplined (TRAIN < 2026-01-22, TEST >= 2026-01-22).",
    "",
    "## Baselines (no Polygon filter)",
    "",
]
base_rows = []
for thr in THRESHOLDS:
    b = baselines[thr]
    tr, te = b["train"], b["test"]
    base_rows.append((f"{thr}%", tr[0], tr[1], tr[2], te[0], te[1], te[2]))
lines.append(tbl(["Gap thr", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                   "TEST n", "TEST hit%", "TEST sum%"], base_rows))

lines += ["", "---", "", "## Signal S1: short_volume_ratio >= threshold on trade_date", "",
          "Higher short_volume_ratio = more of the day's volume is short-selling. "
          "If elevated = confirms the failed-bounce SHORT.", ""]

for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = s1_df[s1_df["gap_thr"] == thr].copy()
    sub["_sort"] = pd.to_numeric(sub["svr_min"], errors="coerce").fillna(999)
    sub = sub.sort_values("_sort").drop(columns="_sort")
    b_test = baselines[thr]["test"][2]
    rows = []
    for _, r in sub.iterrows():
        vs = f"{r['test_sum'] - b_test:+.2f}% vs base" if r["svr_min"] != "none" else "baseline"
        rows.append((r["svr_min"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], vs))
    lines.append(tbl(["SVR min", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    # TRAIN best
    best = sub[sub["svr_min"] != "none"].loc[sub[sub["svr_min"] != "none"]["train_sum"].idxmax()]
    verdict = "PASS" if best["test_sum"] > b_test else "FAIL"
    lines += ["", f"**TRAIN-best SVR threshold {best['svr_min']}** -> "
              f"TEST {best['test_sum']:.2f}% vs baseline {b_test:.2f}% -> **{verdict}**", ""]

lines += ["---", "", "## Signal S2: short_volume_ratio TREND >= threshold (rising short pressure)", ""]
for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = s2_df[s2_df["gap_thr"] == thr]
    b_test = baselines[thr]["test"][2]
    rows = []
    for _, r in sub.iterrows():
        vs = f"{r['test_sum'] - b_test:+.2f}% vs base"
        rows.append((r["trend_min"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], vs))
    lines.append(tbl(["Trend min", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    best = sub.loc[sub["train_sum"].idxmax()]
    verdict = "PASS" if best["test_sum"] > b_test else "FAIL"
    lines += ["", f"**TRAIN-best trend threshold {best['trend_min']}** -> "
              f"TEST {best['test_sum']:.2f}% vs baseline {b_test:.2f}% -> **{verdict}**", ""]

lines += ["---", "", "## Signal S3: SVR elevated vs own 30-day rolling average", ""]
for thr in THRESHOLDS:
    lines.append(f"### Gap threshold {thr}%")
    sub = s3_df[s3_df["gap_thr"] == thr]
    b_test = baselines[thr]["test"][2]
    rows = []
    for _, r in sub.iterrows():
        vs = f"{r['test_sum'] - b_test:+.2f}% vs base"
        rows.append((r["vs_hist_min"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], vs))
    lines.append(tbl(["vs_hist min", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    best = sub.loc[sub["train_sum"].idxmax()]
    verdict = "PASS" if best["test_sum"] > b_test else "FAIL"
    lines += ["", f"**TRAIN-best vs_hist threshold {best['vs_hist_min']}** -> "
              f"TEST {best['test_sum']:.2f}% vs baseline {b_test:.2f}% -> **{verdict}**", ""]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
print("\n=== QUICK RESULTS ===")
print(f"SHORT baseline TEST: {baselines[2.5]['test'][2]:.2f}% @2.5% | {baselines[3.0]['test'][2]:.2f}% @3.0%")
for thr in [2.5]:
    print(f"\nS1 (SVR level) @{thr}%:")
    sub = s1_df[(s1_df["gap_thr"] == thr) & (s1_df["svr_min"] != "none")]
    for _, r in sub.iterrows():
        print(f"  SVR>={r['svr_min']}: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}%")
    print(f"\nS2 (SVR trend) @{thr}%:")
    for _, r in s2_df[s2_df["gap_thr"] == thr].iterrows():
        print(f"  trend>={r['trend_min']}: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}%")
    print(f"\nS3 (SVR vs 30d avg) @{thr}%:")
    for _, r in s3_df[s3_df["gap_thr"] == thr].iterrows():
        print(f"  vs_hist>={r['vs_hist_min']}: TRAIN {r['train_sum']:.2f}% -> TEST {r['test_sum']:.2f}%")
