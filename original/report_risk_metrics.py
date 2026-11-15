"""
report_risk_metrics.py — standard trading-strategy risk metrics (Sharpe,
annualized vol, max drawdown, Calmar) for the BASELINE strategy (drift_low
top-5, no extra controls) vs the NEW best config (cooling-off 5d + wide
stop -18%), so the improvement can be read in standard risk-adjusted terms
rather than just sum_ret.

Methodology: daily portfolio return = sum of realized_return_pct across
all trades entered that day (equal-weighted, consistent with this
project's "sum_ret" convention throughout - NOT compounded, summed). Daily
series built by reindexing over ALL trading days in the sample window
(including zero-return days with no trade) so vol/Sharpe reflect true
time-in-market, not just active days.

Sharpe = mean(daily_ret) / std(daily_ret) * sqrt(252), 0% risk-free.
Max drawdown = largest peak-to-trough drop in the cumulative (summed)
equity curve. Calmar = annualized return / abs(max drawdown).

2.5%/3.0% gap thresholds reported separately (see threshold_overlap.md -
they are NOT independent, so do not sum them into one portfolio).

Output: output/risk_metrics.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "risk_metrics.md"

ET = ZoneInfo("America/New_York")
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]
STOP_PCT = 18.0
COOLOFF_DAYS = 5


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


def simulate_stop(bars_after: pd.DataFrame, entry: float, stop_pct: float) -> float:
    stop_level = entry * (1 - stop_pct / 100)
    tp_level = entry * 1.03
    for _, b in bars_after.iterrows():
        if b["l"] <= stop_level:
            return stop_level
        if b["h"] >= tp_level:
            return tp_level
    return float(bars_after.iloc[-1]["c"])


def apply_cooling_off(data: pd.DataFrame, cooloff_days: int, ret_col: str) -> pd.DataFrame:
    data = data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    last_loss = {}
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]; tk = row["ticker"]
        ll = last_loss.get(tk)
        if ll is not None and (td - ll).days <= cooloff_days:
            keep.append(False)
        else:
            keep.append(True)
            if row[ret_col] < 0:
                last_loss[tk] = td
    data = data.copy(); data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")


def daily_returns(trades: pd.DataFrame, ret_col: str) -> pd.Series:
    """Daily portfolio return (sum of trade returns that day), reindexed
    over all NYSE trading days in [min,max] trade_date so vol/Sharpe
    reflect true time-in-market (zero-return days included)."""
    daily = trades.groupby("trade_date")[ret_col].sum()
    full_idx = pd.bdate_range(trades["trade_date"].min(), trades["trade_date"].max())
    return daily.reindex(full_idx, fill_value=0.0)


def risk_stats(daily: pd.Series) -> dict:
    n_days = len(daily)
    ann_ret = daily.mean() * 252
    ann_vol = daily.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    equity = daily.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")
    return {
        "n_days": n_days,
        "total_ret_pct": equity.iloc[-1],
        "ann_ret_pct": ann_ret,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "calmar": calmar,
    }


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    new_rets = []
    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        entry = row["entry_price"]
        if bars.empty:
            new_rets.append(row["realized_return_pct"]); continue
        sess = pd.Timestamp(row["trade_date"]).date()
        sess_bars = bars[bars["session"] == sess]
        rth = sess_bars[(sess_bars["minute_et"] >= "09:45") & (sess_bars["minute_et"] <= "16:00")]
        bars_after = rth.iloc[1:]
        if bars_after.empty:
            new_rets.append(row["realized_return_pct"]); continue
        ex = simulate_stop(bars_after, entry, STOP_PCT)
        new_rets.append(100.0 * (ex - entry) / entry)
    sel["ret_new"] = new_rets

    def tbl(headers, rows):
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
        for row in rows:
            out.append("|" + "|".join(str(v) for v in row) + "|")
        return "\n".join(out)

    lines = [
        "# Risk-Adjusted Metrics: Baseline vs New Combined LONG Controls",
        "",
        "Standard trading metrics (Sharpe, annualized vol, max drawdown, "
        "Calmar) comparing BASELINE (drift_low top-5, no extra controls) "
        "vs NEW (cooling-off 5d + wide stop -18%). Daily portfolio return = "
        "sum of realized_return_pct across trades entered that day, "
        "reindexed over all NYSE business days (zero-return days included) "
        "so vol/Sharpe reflect true time-in-market. Not compounded - sums, "
        "matching this project's convention throughout.",
        "",
        "**2.5%/3.0% reported separately - see threshold_overlap.md "
        "(94.7% trade overlap, not independent portfolios).**",
        "",
    ]

    summary = []
    for thr in THRESHOLDS:
        pop = sel[sel["gap_threshold_pct"] == thr]
        baseline_daily = daily_returns(pop, "realized_return_pct")
        new_filtered = apply_cooling_off(pop, COOLOFF_DAYS, "ret_new")
        new_daily = daily_returns(new_filtered, "ret_new")

        base_stats = risk_stats(baseline_daily)
        new_stats = risk_stats(new_daily)

        lines.append(f"## Gap threshold {thr}%")
        lines.append("")
        rows = [
            ("Baseline (no controls)", base_stats["n_days"],
             f"{base_stats['total_ret_pct']:.2f}%", f"{base_stats['ann_ret_pct']:.2f}%",
             f"{base_stats['ann_vol_pct']:.2f}%", f"{base_stats['sharpe']:.2f}",
             f"{base_stats['max_dd_pct']:.2f}%", f"{base_stats['calmar']:.2f}"),
            ("New (cooloff 5d + stop -18%)", new_stats["n_days"],
             f"{new_stats['total_ret_pct']:.2f}%", f"{new_stats['ann_ret_pct']:.2f}%",
             f"{new_stats['ann_vol_pct']:.2f}%", f"{new_stats['sharpe']:.2f}",
             f"{new_stats['max_dd_pct']:.2f}%", f"{new_stats['calmar']:.2f}"),
        ]
        lines.append(tbl(["Variant", "Trading days", "Total ret", "Ann. ret",
                           "Ann. vol", "Sharpe", "Max DD", "Calmar"], rows))
        lines.append("")
        summary.append((thr, base_stats, new_stats))
        print(f"\n=== {thr}% ===")
        print(f"Baseline: total={base_stats['total_ret_pct']:.2f}% ann_ret={base_stats['ann_ret_pct']:.2f}% "
              f"ann_vol={base_stats['ann_vol_pct']:.2f}% sharpe={base_stats['sharpe']:.2f} "
              f"max_dd={base_stats['max_dd_pct']:.2f}% calmar={base_stats['calmar']:.2f}")
        print(f"New:      total={new_stats['total_ret_pct']:.2f}% ann_ret={new_stats['ann_ret_pct']:.2f}% "
              f"ann_vol={new_stats['ann_vol_pct']:.2f}% sharpe={new_stats['sharpe']:.2f} "
              f"max_dd={new_stats['max_dd_pct']:.2f}% calmar={new_stats['calmar']:.2f}")

    lines += [
        "---", "",
        "## Notes / caveats", "",
        "- Returns are summed (not compounded) percentage-of-notional per "
        "trade, equal-weighted - this is NOT a real position-sized P&L "
        "(no capital constraint, no limit on concurrent positions). "
        "Treat Sharpe/vol here as a signal-quality diagnostic, not an "
        "investable-strategy risk figure.",
        "- Max drawdown is in the same summed-percent units as total "
        "return (e.g. -300% means the equity curve gave back 300 "
        "percentage-points of cumulative gain from its peak), not a "
        "fraction of capital - do not read it as '-300% of capital lost'.",
        "- 2.5% and 3.0% thresholds are NOT independent samples (94.7% "
        "trade overlap, see threshold_overlap.md) - do not add their "
        "Sharpes/drawdowns together.",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
