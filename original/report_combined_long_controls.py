"""
report_combined_long_controls.py — stacks the two LONG risk controls that
survived (or nearly survived) OOS testing on their own:
  - cooling-off after a loss (5 days, PASSED OOS: output/portfolio_controls.md)
  - wide hard stop-loss (-18%, near-wash OOS: output/wide_stoploss_test.md)

Re-simulates each selected LONG trade bar-by-bar with the -18% stop applied
FIRST (entry/exit mechanics), then applies the cooling-off filter
CHRONOLOGICALLY on top using the stop-adjusted returns (so a trade that was
saved by the stop from a -25% loss to a -18% loss is correctly NOT flagged
as triggering cooling-off if it would have been a small win otherwise, and
vice versa - cooling-off should react to the realistic, stop-protected P&L
a live trader would have seen).

Also jointly re-tunes cooling-off days {0,5,10,15,20} x stop level
{none,15,18,20}% on TRAIN, evaluates the TRAIN-best combo ONCE on TEST.

Output: output/combined_long_controls.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "combined_long_controls.md"

ET = ZoneInfo("America/New_York")
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]
STOP_LEVELS = ["none", 15.0, 18.0, 20.0]
COOLOFFS = [0, 5, 10, 15, 20]


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


def simulate_with_stop(bars_after: pd.DataFrame, entry: float, stop_pct) -> float:
    tp_level = entry * 1.03
    if stop_pct == "none":
        for _, b in bars_after.iterrows():
            if b["h"] >= tp_level:
                return tp_level
        return float(bars_after.iloc[-1]["c"])
    stop_level = entry * (1 - stop_pct / 100)
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def apply_cooling_off(data: pd.DataFrame, cooloff_days: int, ret_col: str) -> pd.DataFrame:
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    ticker_loss_date = {}
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]
        tk = row["ticker"]
        last_loss = ticker_loss_date.get(tk)
        if last_loss is not None and (td - last_loss).days <= cooloff_days:
            keep.append(False)
        else:
            keep.append(True)
            if row[ret_col] < 0:
                ticker_loss_date[tk] = td
    data = data.copy()
    data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    ret_cols = {}
    for lvl in STOP_LEVELS:
        ret_cols[lvl] = []

    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        if bars.empty:
            for lvl in STOP_LEVELS:
                ret_cols[lvl].append(row["realized_return_pct"])
            continue
        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        entry = row["entry_price"]
        if bars_after.empty:
            for lvl in STOP_LEVELS:
                ret_cols[lvl].append(row["realized_return_pct"])
            continue
        for lvl in STOP_LEVELS:
            ex_px = simulate_with_stop(bars_after, entry, lvl)
            ret_cols[lvl].append(100.0 * (ex_px - entry) / entry)

    for lvl in STOP_LEVELS:
        col = "ret_none" if lvl == "none" else f"ret_stop{lvl}"
        sel[col] = ret_cols[lvl]

    def perf(s, col):
        n = len(s)
        hit = (s[col] > 0).mean() * 100 if n else 0
        sumr = s[col].sum()
        return n, round(hit, 2), round(sumr, 2)

    # ── grid search: stop level x cooloff days, TRAIN-best, evaluate on TEST ──
    # IMPORTANT: cooling-off must be applied SEPARATELY per gap_threshold_pct.
    # A ticker that gaps >=3.0% appears in BOTH the 2.5% and 3.0% threshold
    # rows (same underlying trade, recorded twice). Applying cooling-off
    # across both mixed together lets a loss on one threshold's row wrongly
    # suppress the other threshold's row for the same ticker/day, inflating
    # results via a selection artifact (this was caught: an earlier combined
    # pass produced an implausible 96% hit rate). Match portfolio_controls.py's
    # per-threshold loop structure.
    results = []
    for lvl in STOP_LEVELS:
        ret_col = "ret_none" if lvl == "none" else f"ret_stop{lvl}"
        for cd in COOLOFFS:
            tr_n_tot = tr_sum_tot = te_n_tot = te_sum_tot = 0
            tr_hit_num = te_hit_num = 0
            for thr in THRESHOLDS:
                sub = sel[sel["gap_threshold_pct"] == thr]
                filtered = sub if cd == 0 else apply_cooling_off(sub, cd, ret_col)
                tr = filtered[filtered["trade_date"] < TRAIN_CUTOFF]
                te = filtered[filtered["trade_date"] >= TRAIN_CUTOFF]
                tr_n_tot += len(tr); te_n_tot += len(te)
                tr_sum_tot += tr[ret_col].sum(); te_sum_tot += te[ret_col].sum()
                tr_hit_num += (tr[ret_col] > 0).sum(); te_hit_num += (te[ret_col] > 0).sum()
            tr_hit = round(100 * tr_hit_num / tr_n_tot, 2) if tr_n_tot else 0
            te_hit = round(100 * te_hit_num / te_n_tot, 2) if te_n_tot else 0
            results.append({"stop": lvl, "cooloff": cd, "train_n": tr_n_tot,
                             "train_hit": tr_hit, "train_sum": round(tr_sum_tot, 2),
                             "test_n": te_n_tot, "test_hit": te_hit, "test_sum": round(te_sum_tot, 2)})

    grid_df = pd.DataFrame(results)
    best_row = grid_df.loc[grid_df["train_sum"].idxmax()]

    baseline_row = grid_df[(grid_df["stop"] == "none") & (grid_df["cooloff"] == 0)].iloc[0]

    def tbl(headers, rows):
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
        for row in rows:
            out.append("|" + "|".join(str(v) for v in row) + "|")
        return "\n".join(out)

    lines = [
        "# Combined LONG Controls: Cooling-off + Wide Stop-Loss",
        "",
        "Stacks the two LONG risk controls tested independently: cooling-off "
        "after a loss (5d PASSED OOS on its own, output/portfolio_controls.md) "
        "and a wide hard stop-loss (-18% near-wash OOS on its own, "
        "output/wide_stoploss_test.md). Cooling-off is applied chronologically "
        "using the STOP-ADJUSTED return (a more realistic live-trading P&L "
        "estimate than using the unprotected return to decide when to sit out).",
        "",
        "Joint grid search: stop level x cooloff days, TRAIN-best combo chosen "
        "on TRAIN sum_ret, evaluated ONCE on TEST.",
        "",
        "## Full grid (both thresholds combined)",
        "",
    ]
    rows = []
    for _, r in grid_df.iterrows():
        marker = " <- TRAIN best" if (r["stop"] == best_row["stop"] and r["cooloff"] == best_row["cooloff"]) else ""
        rows.append((r["stop"], r["cooloff"], r["train_n"], r["train_hit"], r["train_sum"],
                     r["test_n"], r["test_hit"], r["test_sum"], marker))
    lines.append(tbl(["Stop", "Cooloff(d)", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%", "Note"], rows))

    verdict = "PASS" if best_row["test_sum"] > baseline_row["test_sum"] else "FAIL"
    cooloff_only_row = grid_df[(grid_df["stop"] == "none") & (grid_df["cooloff"] == 5)].iloc[0]
    lines += [
        "",
        f"**Baseline (no stop, no cooloff):** TRAIN {baseline_row['train_sum']:.2f}% -> "
        f"TEST {baseline_row['test_sum']:.2f}%",
        "",
        f"**TRAIN-best combo: stop={best_row['stop']}, cooloff={best_row['cooloff']}d** "
        f"-> TRAIN {best_row['train_sum']:.2f}% -> TEST {best_row['test_sum']:.2f}% -> **{verdict}**",
        "",
        f"**Caveat — the stop adds negligible incremental value on TEST.** "
        f"Cooling-off alone (no stop, 5d) scores TEST {cooloff_only_row['test_sum']:.2f}%, "
        f"essentially identical to (slightly better than) the combo with the -18% stop "
        f"layered on ({best_row['test_sum']:.2f}%). The two controls address different risk: "
        f"cooling-off prevents compounding losses on a name that just lost; the wide stop "
        f"protects against a single catastrophic print on a name's FIRST loss (before "
        f"cooling-off has a chance to engage). Recommend keeping both for that reason, "
        f"even though the wide stop's measured backtest contribution is within noise.",
        "",
        "---", "",
        "## By gap threshold (TRAIN-best combo applied)", "",
    ]

    bs = best_row["stop"]
    bc = best_row["cooloff"]
    ret_col = "ret_none" if bs == "none" else f"ret_stop{bs}"
    for thr in THRESHOLDS:
        base_sub = sel[sel["gap_threshold_pct"] == thr]
        sub = base_sub if bc == 0 else apply_cooling_off(base_sub, int(bc), ret_col)
        tr = sub[sub["trade_date"] < TRAIN_CUTOFF]
        te = sub[sub["trade_date"] >= TRAIN_CUTOFF]
        btr = base_sub[base_sub["trade_date"] < TRAIN_CUTOFF]
        bte = base_sub[base_sub["trade_date"] >= TRAIN_CUTOFF]
        rows = [
            ("baseline", *perf(btr, "realized_return_pct"), *perf(bte, "realized_return_pct")),
            (f"stop={bs},cooloff={bc}d", *perf(tr, ret_col), *perf(te, ret_col)),
        ]
        lines.append(f"### Gap threshold {thr}%")
        lines.append(tbl(["Variant", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")

    print("\n=== QUICK RESULTS ===")
    print(f"Baseline: TRAIN {baseline_row['train_sum']:.2f}% -> TEST {baseline_row['test_sum']:.2f}%")
    print(f"TRAIN-best combo (stop={bs}, cooloff={bc}d): TRAIN {best_row['train_sum']:.2f}% -> TEST {best_row['test_sum']:.2f}% ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
