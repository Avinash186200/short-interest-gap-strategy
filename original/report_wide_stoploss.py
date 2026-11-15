"""
report_wide_stoploss.py — tests Expert's exit-rule suggestion: add a wide
hard stop-loss floor to the LONG drift_low top-5 strategy to cap catastrophic
blowups (SKYQ -25.75% etc.) WITHOUT cutting into normal volatility, the way
the earlier tight stops (2-5%, report_stoploss_test.py) did when they failed
OOS by clipping winners too.

Expert's literal suggestion was -18%. Tests a range of wide floors
{-10,-12,-15,-18,-20,-25,-30}% for context, headlines the literal -18%
result regardless of TRAIN-optimality (it's a fixed SME-specified value,
not a tuned hyperparameter), and separately reports the TRAIN-best wide
floor for comparison.

Re-simulates each selected LONG hold_above_open_945_tp3 trade from cached
1-min bars, same method as report_stoploss_test.py: walk bars from 09:45
entry forward, exit at stop if low <= stop_level (checked first), else at
+3% TP if high >= entry*1.03, else EOD close.

Train/test disciplined: TRAIN-best floor chosen on TRAIN (trade_date <
2026-01-22), evaluated ONCE on TEST (last ~4 months). The literal -18%
result is reported alongside without being "selected" by TRAIN.

Output: output/wide_stoploss_test.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "wide_stoploss_test.md"

ET = ZoneInfo("America/New_York")
STOP_LEVELS = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0]
EXPERT_LEVEL = 18.0
TRAIN_CUTOFF = "2026-01-22"
THRESHOLDS = [2.5, 3.0]


def load_bars(ticker: str) -> pd.DataFrame:
    p = CACHE_DIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert(ET)
    df["session"] = df["ts_et"].dt.date
    df["minute_et"] = df["ts_et"].dt.strftime("%H:%M")
    return df


def simulate_with_stop(bars_after: pd.DataFrame, entry: float, stop_pct: float) -> float:
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[wide_stoploss] missing {TRADES_SEL}")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    results = {lvl: [] for lvl in STOP_LEVELS}
    baseline = []
    skipped = 0

    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        if bars.empty:
            skipped += 1
            baseline.append(row["realized_return_pct"])
            for lvl in STOP_LEVELS:
                results[lvl].append(row["realized_return_pct"])
            continue

        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        entry = row["entry_price"]

        if bars_after.empty:
            skipped += 1
            baseline.append(row["realized_return_pct"])
            for lvl in STOP_LEVELS:
                results[lvl].append(row["realized_return_pct"])
            continue

        baseline.append(row["realized_return_pct"])
        for lvl in STOP_LEVELS:
            ex_px = simulate_with_stop(bars_after, entry, lvl)
            ret = 100.0 * (ex_px - entry) / entry
            results[lvl].append(ret)

    sel["ret_baseline"] = baseline
    for lvl in STOP_LEVELS:
        sel[f"ret_stop{lvl}"] = results[lvl]

    train = sel[sel["trade_date"] < TRAIN_CUTOFF]
    test = sel[sel["trade_date"] >= TRAIN_CUTOFF]

    def hit(s):
        return (s > 0).mean() * 100 if len(s) else 0

    def tbl(headers, rows):
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
        for row in rows:
            out.append("|" + "|".join(str(v) for v in row) + "|")
        return "\n".join(out)

    lines = [
        "# Wide Hard Stop-Loss Test (Expert's -18% suggestion)",
        "",
        "Adds a wide hard stop-loss floor to the LONG drift_low top-5 strategy "
        "to cap catastrophic single-trade blowups (e.g. SKYQ -25.75%) without "
        "clipping normal volatility the way the earlier tight stops (2-5%, see "
        "output/stoploss_test.md, which FAILED OOS) did. Re-simulated bar-by-bar "
        "from cached 1-min data: stop checked first if both stop and TP could "
        "trigger in the same bar (conservative).",
        "",
        f"({skipped} of {len(sel)} trades had no cached bars and fall back to "
        f"the original realized_return_pct unchanged.)",
        "",
        "Train/test disciplined: TRAIN-best floor chosen on TRAIN (trade_date < "
        "2026-01-22), evaluated ONCE on TEST. The literal **-18%** value Expert "
        "suggested is reported on its own merits (not tuned).",
        "",
        "## Combined (both gap thresholds together — matches headline strategy)",
        "",
    ]

    rows = []
    base_train_sum = train["ret_baseline"].sum()
    base_test_sum = test["ret_baseline"].sum()
    rows.append(("none (baseline)", len(train), round(hit(train["ret_baseline"]), 2),
                  round(base_train_sum, 2), len(test), round(hit(test["ret_baseline"]), 2),
                  round(base_test_sum, 2), ""))
    best_lvl = max(STOP_LEVELS, key=lambda lvl: train[f"ret_stop{lvl}"].sum())
    for lvl in STOP_LEVELS:
        col = f"ret_stop{lvl}"
        marker = ""
        if lvl == best_lvl:
            marker += " <- TRAIN best"
        if lvl == EXPERT_LEVEL:
            marker += " <- Expert's suggestion"
        rows.append((f"-{lvl}%", len(train), round(hit(train[col]), 2),
                      round(train[col].sum(), 2), len(test), round(hit(test[col]), 2),
                      round(test[col].sum(), 2), marker))
    lines.append(tbl(["Stop level", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))

    expert_test_sum = test[f"ret_stop{EXPERT_LEVEL}"].sum()
    best_test_sum = test[f"ret_stop{best_lvl}"].sum()
    verdict_expert = "PASS" if expert_test_sum > base_test_sum else "FAIL"
    verdict_best = "PASS" if best_test_sum > base_test_sum else "FAIL"

    lines += [
        "",
        f"**Expert's -18% floor:** TEST sum_ret {expert_test_sum:.2f}% vs baseline "
        f"{base_test_sum:.2f}% -> **{verdict_expert}**",
        "",
        f"**TRAIN-best floor (-{best_lvl}%):** TEST sum_ret {best_test_sum:.2f}% vs "
        f"baseline {base_test_sum:.2f}% -> **{verdict_best}**",
        "",
        "---", "",
        "## By gap threshold", "",
    ]

    for thr in THRESHOLDS:
        lines.append(f"### Gap threshold {thr}%")
        tr_t = train[train["gap_threshold_pct"] == thr]
        te_t = test[test["gap_threshold_pct"] == thr]
        rows = []
        rows.append(("none (baseline)", len(tr_t), round(hit(tr_t["ret_baseline"]), 2),
                      round(tr_t["ret_baseline"].sum(), 2), len(te_t),
                      round(hit(te_t["ret_baseline"]), 2), round(te_t["ret_baseline"].sum(), 2), ""))
        best_lvl_t = max(STOP_LEVELS, key=lambda lvl: tr_t[f"ret_stop{lvl}"].sum())
        for lvl in STOP_LEVELS:
            col = f"ret_stop{lvl}"
            marker = ""
            if lvl == best_lvl_t:
                marker += " <- TRAIN best"
            if lvl == EXPERT_LEVEL:
                marker += " <- Expert's suggestion"
            rows.append((f"-{lvl}%", len(tr_t), round(hit(tr_t[col]), 2),
                          round(tr_t[col].sum(), 2), len(te_t), round(hit(te_t[col]), 2),
                          round(te_t[col].sum(), 2), marker))
        lines.append(tbl(["Stop level", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wide_stoploss] wrote {OUT_MD}", flush=True)

    print("\n=== QUICK RESULTS ===")
    print(f"Baseline: TRAIN {base_train_sum:.2f}% -> TEST {base_test_sum:.2f}%")
    print(f"Expert -18%: TRAIN {train[f'ret_stop{EXPERT_LEVEL}'].sum():.2f}% -> TEST {expert_test_sum:.2f}% ({verdict_expert})")
    print(f"TRAIN-best -{best_lvl}%: TRAIN {train[f'ret_stop{best_lvl}'].sum():.2f}% -> TEST {best_test_sum:.2f}% ({verdict_best})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
