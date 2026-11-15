"""
select_signals.py — pipeline step that applies the validated selection rule
to trades.parquet, producing data/trades_selected.parquet:

  - LONG (hold_above_open_945_tp3): apply `drift_low` top-5 selection -
    rank each day's candidates by settlement->publication price_drift_pct
    ASCENDING (biggest pre-publication price decline first) within each
    gap_threshold_pct group, keep only the top 5. This is the validated
    primary LONG signal (see output/rank_drift_strategy.md,
    output/ranking_refine2_corrected.md, output/rank_drift_recent.md -
    positive OOS, positive in the last 3/4 months).
  - LONG trailing_atr / SHORT (both rules): passed through UNFILTERED,
    tagged `selected=False` - no validated selection rule exists yet for
    these (see NEXT_ITERATION_PLAN.md item on SHORT). Kept in the output so
    report_selected.py can show them as "research / not yet validated"
    rather than silently dropping them.
  - Portfolio controls on the selected LONG candidates - the current
    "live" strategy spec, validated in
    output/combined_long_controls.md / output/risk_metrics.md: re-simulate
    each selected trade's exit with a wide -18% hard stop (catches
    catastrophic blowups only, e.g. SKYQ), then apply a 5-day per-ticker
    cooling-off (skip a name for 5 days after it produces a stop-adjusted
    loss). Adds `stop_adjusted_return_pct` and `live_trade` so
    report_selected.py can report the live spec without re-deriving it.

Adds columns to every row: `selected` (bool), `selection_rule` (str, e.g.
"drift_low_top5" or "none"), `stop_adjusted_return_pct` (float, NaN unless
selected LONG), `live_trade` (bool, selected candidates that also survive
the 5-day cooling-off).

Run after backtest.py and drift_signal.py. Writes data/trades_selected.parquet.
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT = ROOT / "data" / "trades_selected.parquet"

LONG_PRIMARY_RULE = "hold_above_open_945_tp3"
TOP_N = 5
ET = ZoneInfo("America/New_York")
STOP_PCT = 18.0
COOLOFF_DAYS = 5


def _load_bars(ticker: str) -> pd.DataFrame:
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


def _simulate_stop(bars_after: pd.DataFrame, entry: float, stop_pct: float) -> float:
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def _stop_adjusted_returns(rows: pd.DataFrame) -> pd.Series:
    """Re-simulate each row's exit with the -18% wide stop, from cached
    1-min bars. Falls back to realized_return_pct when bars are missing."""
    bars_cache: dict[str, pd.DataFrame] = {}
    out = []
    for _, row in rows.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = _load_bars(ticker)
        bars = bars_cache[ticker]
        entry = row["entry_price"]
        if bars.empty:
            out.append(row["realized_return_pct"])
            continue
        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        if bars_after.empty:
            out.append(row["realized_return_pct"])
            continue
        ex = _simulate_stop(bars_after, entry, STOP_PCT)
        out.append(100.0 * (ex - entry) / entry)
    return pd.Series(out, index=rows.index)


def _apply_cooling_off(rows: pd.DataFrame, cooloff_days: int, ret_col: str) -> pd.Index:
    """Returns the index of rows that survive per-ticker cooling-off,
    applied chronologically per gap_threshold_pct (thresholds must be kept
    separate - a >=3.0% gap also qualifies as a >=2.5% gap, so combining
    them would let a loss on one threshold's row wrongly suppress the
    other threshold's row for the same ticker/day)."""
    keep_idx = []
    for thr, sub in rows.groupby("gap_threshold_pct"):
        sub = sub.sort_values(["trade_date", "ticker"])
        last_loss: dict[str, pd.Timestamp] = {}
        for idx, row in sub.iterrows():
            td, tk = row["trade_date"], row["ticker"]
            ll = last_loss.get(tk)
            if ll is not None and (td - ll).days <= cooloff_days:
                continue
            keep_idx.append(idx)
            if row[ret_col] < 0:
                last_loss[tk] = td
    return pd.Index(keep_idx)


def main() -> int:
    if not TRADES.exists() or not DRIFT.exists():
        print("[select_signals] missing trades.parquet or drift_signal.parquet")
        return 1

    trades = pd.read_parquet(TRADES)
    drift = pd.read_parquet(DRIFT)

    df = trades.merge(
        drift[["settlementDate", "symbolCode", "price_drift_pct", "volume_ratio"]],
        left_on=["settlement_date", "ticker"],
        right_on=["settlementDate", "symbolCode"],
        how="left",
    ).drop(columns=["settlementDate", "symbolCode"])

    df["selected"] = False
    df["selection_rule"] = "none"

    # LONG primary rule: drift_low top-5, only where drift data is available
    is_long_primary = (df["side"] == "LONG") & (df["rule_name"] == LONG_PRIMARY_RULE)
    has_drift = df["price_drift_pct"].notna()

    candidates = df[is_long_primary & has_drift].copy()
    candidates["rank"] = candidates.groupby(["trade_date", "gap_threshold_pct"])["price_drift_pct"] \
        .rank(method="first", ascending=True)

    selected_idx = candidates[candidates["rank"] <= TOP_N].index
    df.loc[selected_idx, "selected"] = True
    df.loc[selected_idx, "selection_rule"] = "drift_low_top5"

    # Portfolio controls (live strategy spec) on top of the selected candidates:
    # -18% wide stop re-simulated from cached bars, then 5-day per-ticker cooling-off.
    df["stop_adjusted_return_pct"] = float("nan")
    df["live_trade"] = False

    selected_rows = df.loc[selected_idx].copy()
    selected_rows["trade_date"] = pd.to_datetime(selected_rows["trade_date"])
    stop_adj = _stop_adjusted_returns(selected_rows)
    df.loc[selected_idx, "stop_adjusted_return_pct"] = stop_adj

    selected_rows["stop_adjusted_return_pct"] = stop_adj
    live_idx = _apply_cooling_off(selected_rows, COOLOFF_DAYS, "stop_adjusted_return_pct")
    df.loc[live_idx, "live_trade"] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    n_long_primary = is_long_primary.sum()
    n_long_no_drift = (is_long_primary & ~has_drift).sum()
    n_selected = df["selected"].sum()
    n_live = df["live_trade"].sum()
    print(f"[select_signals] {len(df):,} total rows", flush=True)
    print(f"[select_signals] LONG {LONG_PRIMARY_RULE}: {n_long_primary:,} candidates "
          f"({n_long_no_drift:,} missing drift data, excluded from selection)", flush=True)
    print(f"[select_signals] selected (drift_low_top5): {n_selected:,} rows", flush=True)
    print(f"[select_signals] live_trade (+ cooloff{COOLOFF_DAYS}d + stop{STOP_PCT}%): {n_live:,} rows", flush=True)
    print(f"[select_signals] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
