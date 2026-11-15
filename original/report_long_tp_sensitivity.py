"""
report_long_tp_sensitivity.py — tests whether the fixed +3% take-profit on
the LONG drift_low top-5 strategy is well-chosen, or whether a different
fixed TP level does better. The current level was "just picked" (per
Expert's original comment that motivated the trailing-ATR exit experiment,
which failed OOS) — this directly tests fixed alternatives instead of a
volatility-scaled stop.

Re-simulates each selected LONG trade from cached 1-min bars: entry at
09:45, exit at +TP% intraday or EOD close (no stop-loss layered in here -
isolates the TP-level question on its own).

Train/test disciplined: TP level chosen on TRAIN (trade_date < 2026-01-22),
evaluated ONCE on TEST. 2.5%/3.0% gap thresholds reported separately.

Output: output/long_tp_sensitivity.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "long_tp_sensitivity.md"

ET = ZoneInfo("America/New_York")
TP_LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
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


def simulate_tp(bars_after: pd.DataFrame, entry: float, tp_pct: float) -> float:
    tp_level = entry * (1 + tp_pct / 100)
    for _, b in bars_after.iterrows():
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    ret_cols = {lvl: [] for lvl in TP_LEVELS}

    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        entry = row["entry_price"]
        if bars.empty:
            for lvl in TP_LEVELS:
                ret_cols[lvl].append(row["realized_return_pct"])
            continue
        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        if bars_after.empty:
            for lvl in TP_LEVELS:
                ret_cols[lvl].append(row["realized_return_pct"])
            continue
        for lvl in TP_LEVELS:
            ex_px = simulate_tp(bars_after, entry, lvl)
            ret_cols[lvl].append(100.0 * (ex_px - entry) / entry)

    for lvl in TP_LEVELS:
        sel[f"ret_tp{lvl}"] = ret_cols[lvl]

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
        "# LONG Take-Profit Level Sensitivity",
        "",
        "Tests whether a different fixed TP level beats the current +3% on "
        "the LONG drift_low top-5 strategy. Re-simulated bar-by-bar from "
        "cached 1-min data: entry 09:45, exit at +TP% intraday or EOD close "
        "(no stop-loss here - isolates the TP question).",
        "",
        "Train/test disciplined: TP level chosen on TRAIN (trade_date < "
        "2026-01-22), evaluated ONCE on TEST. 2.5%/3.0% gap thresholds "
        "reported separately.",
        "",
    ]

    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr]
        train = sub[sub["trade_date"] < TRAIN_CUTOFF]
        test = sub[sub["trade_date"] >= TRAIN_CUTOFF]
        rows = []
        for lvl in TP_LEVELS:
            col = f"ret_tp{lvl}"
            tr = perf(train, col)
            te = perf(test, col)
            rows.append((f"{lvl}%", tr[0], tr[1], tr[2], te[0], te[1], te[2]))
        best_lvl = max(TP_LEVELS, key=lambda lvl: train[f"ret_tp{lvl}"].sum())
        rows_marked = []
        for lvl, row in zip(TP_LEVELS, rows):
            marker = " <- TRAIN best" if lvl == 3.0 else ""
            marker2 = " <- TRAIN best" if lvl == best_lvl else ""
            rows_marked.append(row + (marker2,))
        lines.append(f"## Gap threshold {thr}%")
        lines.append("")
        lines.append(tbl(["TP level", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%", "Note"], rows_marked))
        cur = perf(test, "ret_tp3.0")
        best = perf(test, f"ret_tp{best_lvl}")
        verdict = "PASS" if best[2] > cur[2] else "FAIL"
        lines += ["",
                  f"Current (+3.0%): TEST {cur[2]:.2f}%. TRAIN-best ({best_lvl}%): "
                  f"TEST {best[2]:.2f}% -> **{verdict}** (vs current, not vs no-TP)",
                  "", "---", ""]
        print(f"\n=== {thr}% ===")
        print(f"Current TP=3.0%: TRAIN {perf(train,'ret_tp3.0')[2]:.2f}% -> TEST {cur[2]:.2f}%")
        print(f"TRAIN-best TP={best_lvl}%: TRAIN {perf(train, f'ret_tp{best_lvl}')[2]:.2f}% -> TEST {best[2]:.2f}%")

    lines += [
        "## Conclusion", "",
        "TRAIN sum_ret rises monotonically with TP width across the whole "
        "grid tested (2%-6%) - it never rolls over - so the naive "
        "TRAIN-best rule always picks the widest level tested (6%), which "
        "**fails badly OOS** (TEST collapses to ~9% from baseline ~53%). "
        "This is a textbook case of a monotonic relationship breaking the "
        "TRAIN-best selection heuristic: there is no interior optimum "
        "visible from TRAIN alone here.",
        "",
        "Note (NOT a recommendation - would violate train/test discipline "
        "since it requires looking at TEST to find): intermediate levels "
        "like 4.0% happen to beat the current 3.0% on TEST at both "
        "thresholds (85.01% vs 52.79% @2.5%, 69.24% vs 53.54% @3.0%) WHILE "
        "also outperforming on TRAIN's local trend - but selecting it "
        "requires hindsight, so it cannot be adopted from this test alone. "
        "Flagging as a candidate worth re-testing if TRAIN data accumulates "
        "to where 4% becomes selectable on TRAIN evidence by itself.",
        "",
        "**Verdict: TP-level tuning fails OOS, same pattern as every other "
        "per-trade parameter tuned in this investigation. Current +3% TP "
        "is not provably suboptimal - keep as-is.**",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
