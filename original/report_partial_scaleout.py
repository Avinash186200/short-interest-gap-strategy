"""
report_partial_scaleout.py — tests a partial scale-out exit on LONG:
book HALF the position at +3% TP (current rule), let the other half ride
with a trailing stop instead of going to EOD flat. Current rule is binary
(100% out at TP or 100% out at EOD) - this tests whether letting a runner
go captures more of the big squeeze days without giving back the TP
discipline that protects the median trade.

Re-simulates bar-by-bar from cached 1-min bars:
  - first half: exits at +3% TP (or EOD if never hit, same as today)
  - second half: if/once +3% is hit, arms a trailing stop trailing X% off
    the running high; before TP, this half has the SAME -wide_stop floor
    as the validated combined-controls test (-18%, since that's the
    current best LONG config). Falls back to EOD if neither triggers.

Tests trailing distances {1,1.5,2,3,4}%, train/test disciplined.
2.5%/3.0% reported separately.

Output: output/partial_scaleout.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "partial_scaleout.md"

ET = ZoneInfo("America/New_York")
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]
TRAIL_PCTS = [1.0, 1.5, 2.0, 3.0, 4.0]
STOP_PCT = 18.0


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


def simulate_full_exit(bars_after: pd.DataFrame, entry: float, stop_pct: float) -> float:
    """Current best-known rule: -18% hard stop, +3% TP, else EOD (both halves do this today)."""
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def simulate_runner(bars_after: pd.DataFrame, entry: float, stop_pct: float, trail_pct: float) -> float:
    """Second half: -18% hard stop before TP; once +3% TP is touched, arm a
    trailing stop trail_pct off the running high instead of taking profit."""
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    armed = False
    best = entry
    for _, b in bars_after.iterrows():
        if not armed:
            if b["l"] <= stop_level:
                return stop_level
            if b["h"] >= tp_level:
                armed = True
                best = b["h"]
        else:
            best = max(best, b["h"])
            trail_level = best * (1 - trail_pct / 100)
            if b["l"] <= trail_level:
                return trail_level
    return float(bars_after.iloc[-1]["c"])


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    full_rets = []
    runner_rets = {t: [] for t in TRAIL_PCTS}

    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        entry = row["entry_price"]
        if bars.empty:
            full_rets.append(row["realized_return_pct"])
            for t in TRAIL_PCTS:
                runner_rets[t].append(row["realized_return_pct"])
            continue
        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        if bars_after.empty:
            full_rets.append(row["realized_return_pct"])
            for t in TRAIL_PCTS:
                runner_rets[t].append(row["realized_return_pct"])
            continue

        ex_full = simulate_full_exit(bars_after, entry, STOP_PCT)
        full_rets.append(100.0 * (ex_full - entry) / entry)
        for t in TRAIL_PCTS:
            ex_runner = simulate_runner(bars_after, entry, STOP_PCT, t)
            runner_rets[t].append(100.0 * (ex_runner - entry) / entry)

    sel["ret_full"] = full_rets
    for t in TRAIL_PCTS:
        sel[f"ret_runner{t}"] = runner_rets[t]
        sel[f"ret_blend{t}"] = 0.5 * sel["ret_full"] + 0.5 * sel[f"ret_runner{t}"]

    def perf(s, col):
        n = len(s)
        hit = (s[col] > 0).mean() * 100 if n else 0
        sumr = s[col].sum()
        return n, round(hit, 2), round(sumr, 2)

    def tbl(headers, rows):
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
        for row in rows:
            out.append("|" + "|".join(str(v) for v in row) + "|")
        return "\n".join(out)

    lines = [
        "# Partial Scale-Out Exit (LONG)",
        "",
        "Tests booking HALF the LONG position at +3% TP (current rule) and "
        "letting the OTHER half ride past +3% with a trailing stop, instead "
        "of taking the full position off at +3%/EOD. Both halves share the "
        "-18% hard stop (current best-known config) before the TP is first "
        "touched. Blend = 50% current-rule + 50% runner-with-trailing-stop.",
        "",
        "Train/test disciplined: trailing distance chosen on TRAIN, "
        "evaluated ONCE on TEST. 2.5%/3.0% reported separately.",
        "",
    ]

    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr]
        train = sub[sub["trade_date"] < TRAIN_CUTOFF]
        test = sub[sub["trade_date"] >= TRAIN_CUTOFF]
        base_tr, base_te = perf(train, "ret_full"), perf(test, "ret_full")

        lines.append(f"## Gap threshold {thr}%")
        lines.append("")
        lines.append(tbl(["", "n", "hit%", "sum%"],
                          [("TRAIN baseline (100% @TP)", *base_tr),
                           ("TEST baseline (100% @TP)", *base_te)]))
        lines.append("")
        rows = []
        results = []
        for t in TRAIL_PCTS:
            col = f"ret_blend{t}"
            tr_p = perf(train, col)
            te_p = perf(test, col)
            results.append((t, tr_p, te_p))
            rows.append((f"{t}%", *tr_p, *te_p))
        lines.append(tbl(["Trail dist", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        best = max(results, key=lambda x: x[1][2])
        verdict = "PASS" if best[2][2] > base_te[2] else "FAIL"
        lines += ["", f"TRAIN-best trail={best[0]}%: TEST {best[2][2]:.2f}% vs baseline "
                  f"{base_te[2]:.2f}% -> **{verdict}**", "", "---", ""]

        print(f"\n=== {thr}% ===")
        print(f"baseline: TRAIN {base_tr[2]:.2f}% -> TEST {base_te[2]:.2f}%")
        print(f"best blend (trail={best[0]}%): TRAIN {best[1][2]:.2f}% -> TEST {best[2][2]:.2f}% ({verdict})")

    lines += [
        "## Conclusion", "",
        "Same pattern as the TP-sensitivity test: TRAIN sum_ret rises "
        "monotonically with trail distance at both thresholds (no interior "
        "optimum), so the naive TRAIN-best always picks the widest level "
        "tested (4%). Result is split: FAILS at 2.5% (35.27% vs 50.59%), "
        "marginally PASSES at 3.0% (55.16% vs 51.34%) - given the "
        "monotonic-grid selection issue, the 3.0% pass is not trustworthy "
        "as a real edge (same mechanism that produced a false pass "
        "elsewhere). **Not adopted** - current all-or-nothing +3% TP stays.",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
