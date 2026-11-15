"""
report_stoploss_test.py — tests whether adding a per-trade stop-loss to the
primary LONG drift_low top-5 strategy (select_signals.py /
results_summary_selected.md) improves the headline result, motivated by the
drawdown investigation (output/drawdown_investigation.md) which found a few
illiquid small-cap names (SKYQ -25.75%, OMER -14.71%, ...) drove most of the
2025-10->2026-03 drawdown and the current rules have no protective stop
(exit only at +3% TP or EOD).

Re-simulates each selected LONG hold_above_open_945_tp3 trade from cached
1-min bars (cache/intraday/{TICKER}.parquet), walking bars from 09:45 entry
forward and checking, bar by bar:
  - if low <= stop_level: exit at stop_level, "stop_loss" (checked first -
    conservative, i.e. if both stop and TP could trigger in the same bar we
    assume the stop wins)
  - elif high >= entry*(1+3%): exit at TP, "take_profit"
  - else continue; fall back to EOD close if neither triggers.

Tests stop levels in {2, 3, 4, 5}% on TRAIN (trade_date < 2026-01-22),
reports TEST (last 4 months) sum_ret for the TRAIN-selected level vs the
no-stop baseline. Train/test disciplined per project convention.

Writes output/stoploss_test.md.
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "stoploss_test.md"

ET = ZoneInfo("America/New_York")
STOP_LEVELS = [2.0, 3.0, 4.0, 5.0]
TRAIN_CUTOFF = "2026-01-22"


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


def simulate_with_stop(bars_after: pd.DataFrame, entry: float, stop_pct: float) -> tuple[float, str]:
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level, "stop_loss"
        if b["h"] >= tp_level:
            return tp_level, "take_profit"
    return float(bars_after.iloc[-1]["c"]), "eod"


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[stoploss] missing {TRADES_SEL} - run select_signals.py first")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[df["selected"]].copy()
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
        bars_after = rth.iloc[1:]  # match backtest.py: skip the 09:45 bar itself
        entry = row["entry_price"]

        if bars_after.empty:
            skipped += 1
            baseline.append(row["realized_return_pct"])
            for lvl in STOP_LEVELS:
                results[lvl].append(row["realized_return_pct"])
            continue

        baseline.append(row["realized_return_pct"])
        for lvl in STOP_LEVELS:
            ex_px, _ = simulate_with_stop(bars_after, entry, lvl)
            ret = 100.0 * (ex_px - entry) / entry
            results[lvl].append(ret)

    sel["ret_baseline"] = baseline
    for lvl in STOP_LEVELS:
        sel[f"ret_stop{lvl}"] = results[lvl]

    train = sel[sel["trade_date"] < TRAIN_CUTOFF]
    test = sel[sel["trade_date"] >= TRAIN_CUTOFF]

    lines = [
        "# Per-trade stop-loss test (LONG drift_low top-5)",
        "",
        "Re-simulates each selected trade from cached 1-min bars, adding a "
        "hard stop-loss at entry*(1-stop_pct%), checked bar-by-bar alongside "
        "the existing +3% TP / EOD exit (stop checked first if both trigger "
        "in the same bar - conservative). Train/test disciplined: stop level "
        "chosen on TRAIN (trade_date < 2026-01-22), evaluated once on TEST.",
        "",
        f"({skipped} of {len(sel)} trades had no cached bars for that "
        f"session and fall back to the original realized_return_pct "
        f"unchanged.)",
        "",
        "## TRAIN vs TEST sum_ret by stop level",
        "",
        "| stop_pct | TRAIN sum_ret | TEST sum_ret | TRAIN hit% | TEST hit% |",
        "|---:|---:|---:|---:|---:|",
    ]

    def hit(s):
        return (s > 0).mean() * 100

    lines.append(
        f"| none (baseline) | {train['ret_baseline'].sum():.2f} | "
        f"{test['ret_baseline'].sum():.2f} | {hit(train['ret_baseline']):.2f} | "
        f"{hit(test['ret_baseline']):.2f} |"
    )
    for lvl in STOP_LEVELS:
        col = f"ret_stop{lvl}"
        lines.append(
            f"| {lvl} | {train[col].sum():.2f} | {test[col].sum():.2f} | "
            f"{hit(train[col]):.2f} | {hit(test[col]):.2f} |"
        )
    lines.append("")

    best_lvl = max(STOP_LEVELS, key=lambda lvl: train[f"ret_stop{lvl}"].sum())
    train_best = train[f"ret_stop{best_lvl}"].sum()
    train_base = train["ret_baseline"].sum()
    test_best = test[f"ret_stop{best_lvl}"].sum()
    test_base = test["ret_baseline"].sum()
    lines.append(
        f"**TRAIN-selected stop level: {best_lvl}%** "
        f"(TRAIN sum_ret {train_base:.2f} -> {train_best:.2f}). "
        f"On TEST: baseline {test_base:.2f} -> with stop {test_best:.2f}."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stoploss] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
