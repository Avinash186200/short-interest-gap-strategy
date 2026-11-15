"""
report_short_portfolio_controls.py — tests the portfolio-level controls that
worked for LONG (cooling-off after a loss, rolling frequency cap) on the
SHORT side, on the raw SHORT baseline.

IMPORTANT: a trade gapping >=3.0% qualifies for BOTH the 2.5% and 3.0%
gap_threshold_pct buckets and appears as two rows with IDENTICAL
entry/exit/realized_return_pct. Applying a chronological control (cooling-
off, freq cap) across both buckets MIXED TOGETHER lets a loss recorded under
one threshold's row wrongly suppress the duplicate row under the other
threshold (cross-contamination) - this produced an inflated, invalid result
in an earlier version of this script (TEST appeared to flip to +121.79%,
which did not survive once thresholds were separated - see history).
Per project convention (filter_tests.py, portfolio_controls.py), 2.5% and
3.0% are always processed and reported SEPARATELY.

Train/test disciplined throughout.

Output: output/short_portfolio_controls.md
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("<local-path>")
OUT_MD = ROOT / "output" / "short_portfolio_controls.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
SHORT_RULE = "failed_bounce_1000_tp_2.5"
THRESHOLDS = [2.5, 3.0]

df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["publication_date"] = pd.to_datetime(df["publication_date"])

short_all = df[(df["side"] == "SHORT") & (df["rule_name"] == SHORT_RULE)].copy()
short_all = short_all.dropna(subset=["price_drift_pct", "realized_return_pct"])
short_all = short_all.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def perf(s):
    n = len(s)
    hit = (s["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = s["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)


def apply_rolling_freq_cap(data, max_trades, window_days=30):
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    ticker_dates = defaultdict(list)
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]; tk = row["ticker"]
        cutoff = td - pd.Timedelta(days=window_days)
        past = [d for d in ticker_dates[tk] if d >= cutoff]
        keep.append(len(past) < max_trades)
        ticker_dates[tk].append(td)
    data = data.copy(); data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")


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


def run_grid(pop):
    freq_caps = [1, 2, 3, 4, 5]
    cooloffs = [0, 5, 10, 15, 20, 30]

    base_tr = perf(pop[pop["trade_date"] < TRAIN_CUTOFF])
    base_te = perf(pop[pop["trade_date"] >= TRAIN_CUTOFF])

    freq_rows = []
    for cap in freq_caps:
        f = apply_rolling_freq_cap(pop, cap)
        tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
        te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])
        freq_rows.append({"cap": cap, "train_n": tr[0], "train_hit": tr[1], "train_sum": tr[2],
                           "test_n": te[0], "test_hit": te[1], "test_sum": te[2]})
    freq_df = pd.DataFrame(freq_rows)
    best_cap = freq_df.loc[freq_df["train_sum"].idxmax()]

    cool_rows = []
    for cd in cooloffs:
        f = pop if cd == 0 else apply_cooling_off(pop, cd)
        tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
        te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])
        cool_rows.append({"cooloff": cd, "train_n": tr[0], "train_hit": tr[1], "train_sum": tr[2],
                           "test_n": te[0], "test_hit": te[1], "test_sum": te[2]})
    cool_df = pd.DataFrame(cool_rows)
    best_cool = cool_df.loc[cool_df["train_sum"].idxmax()]

    f = apply_rolling_freq_cap(pop, int(best_cap["cap"]))
    if best_cool["cooloff"] > 0:
        f = apply_cooling_off(f, int(best_cool["cooloff"]))
    tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
    te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])

    return {
        "base_tr": base_tr, "base_te": base_te,
        "freq_df": freq_df, "best_cap": best_cap,
        "cool_df": cool_df, "best_cool": best_cool,
        "combo_tr": tr, "combo_te": te,
    }


def tbl(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        out.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(out)


lines = [
    "# SHORT Side: Portfolio-Level Controls (Cooling-off / Freq Cap)",
    "",
    "Tests the LONG-side controls that worked (cooling-off, rolling frequency "
    "cap) on the SHORT `failed_bounce_1000_tp_2.5` rule.",
    "",
    "**2.5% and 3.0% gap thresholds are processed and reported SEPARATELY** "
    "— a trade gapping >=3.0% qualifies for both buckets with identical P&L; "
    "running the chronological controls on both mixed together cross-"
    "contaminates them (a loss flagged under one threshold's row would "
    "incorrectly suppress the duplicate row under the other threshold).",
    "",
]

summary_rows = []
for thr in THRESHOLDS:
    pop = short_all[short_all["gap_threshold_pct"] == thr]
    res = run_grid(pop)

    lines += [f"## Gap threshold {thr}%", "",
              tbl(["", "n", "hit%", "sum%"],
                  [("TRAIN baseline", res["base_tr"][0], res["base_tr"][1], res["base_tr"][2]),
                   ("TEST baseline", res["base_te"][0], res["base_te"][1], res["base_te"][2])]),
              "",
              "### Rolling frequency cap (30d window)", ""]
    rows = []
    for _, r in res["freq_df"].iterrows():
        marker = " <- TRAIN best" if r["cap"] == res["best_cap"]["cap"] else ""
        rows.append((r["cap"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Cap", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    v = "PASS" if res["best_cap"]["test_sum"] > res["base_te"][2] else "FAIL"
    lines += ["", f"TRAIN-best cap={res['best_cap']['cap']} -> TEST {res['best_cap']['test_sum']:.2f}% "
              f"vs baseline {res['base_te'][2]:.2f}% -> **{v}**", "",
              "### Cooling-off after a loss", ""]
    rows = []
    for _, r in res["cool_df"].iterrows():
        marker = " <- TRAIN best" if r["cooloff"] == res["best_cool"]["cooloff"] else ""
        rows.append((f"{int(r['cooloff'])}d", r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Cooloff", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
    v2 = "PASS" if res["best_cool"]["test_sum"] > res["base_te"][2] else "FAIL"
    lines += ["", f"TRAIN-best cooloff={int(res['best_cool']['cooloff'])}d -> TEST "
              f"{res['best_cool']['test_sum']:.2f}% vs baseline {res['base_te'][2]:.2f}% -> **{v2}**", "",
              "### Combined (best cap + best cooloff)", "",
              tbl(["", "n", "hit%", "sum%"],
                  [("TRAIN", res["combo_tr"][0], res["combo_tr"][1], res["combo_tr"][2]),
                   ("TEST", res["combo_te"][0], res["combo_te"][1], res["combo_te"][2])]),
              ""]
    v3 = "PASS" if res["combo_te"][2] > res["base_te"][2] else "FAIL"
    lines += [f"Combined -> TEST {res['combo_te'][2]:.2f}% vs baseline {res['base_te'][2]:.2f}% -> **{v3}**",
              "", "---", ""]

    summary_rows.append((thr, res))
    print(f"\n=== {thr}% ===")
    print(f"baseline: TRAIN {res['base_tr'][2]:.2f}% -> TEST {res['base_te'][2]:.2f}%")
    print(f"  best freq cap={res['best_cap']['cap']}: TRAIN {res['best_cap']['train_sum']:.2f}% -> TEST {res['best_cap']['test_sum']:.2f}%")
    print(f"  best cooloff={int(res['best_cool']['cooloff'])}d: TRAIN {res['best_cool']['train_sum']:.2f}% -> TEST {res['best_cool']['test_sum']:.2f}%")
    print(f"  combined: TRAIN {res['combo_tr'][2]:.2f}% -> TEST {res['combo_te'][2]:.2f}%")

lines += ["## Conclusion", ""]
for thr, res in summary_rows:
    best_overall = max(
        [("freq cap", res["best_cap"]["test_sum"]),
         ("cooloff", res["best_cool"]["test_sum"]),
         ("combined", res["combo_te"][2])],
        key=lambda x: x[1])
    lines.append(f"- {thr}%: best variant '{best_overall[0]}' at TEST {best_overall[1]:.2f}% "
                 f"(baseline {res['base_te'][2]:.2f}%)")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
