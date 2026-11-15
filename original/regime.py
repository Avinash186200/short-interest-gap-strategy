"""
regime.py — Item 2 helper: daily OHLC + rolling ATR% per ticker, used as the
"regime volatility" input for the trailing-stop exit rule in backtest.py.

Per Expert's comment on the call: the fixed 2.5-3% TP was "just picked" for
a 3-month window and should instead be based on how volatile the market /
name currently is.

ATR% is computed on a rolling lookback of daily bars and shifted by one
session, so the value used for a trade entered on day D only reflects
information available through day D-1 (no lookahead).
"""
from __future__ import annotations
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache" / "intraday"

ATR_LOOKBACK = 14


def load_daily_ohlc(ticker: str) -> pd.DataFrame:
    """One row per session: open, high, low, close, volume (RTH only)."""
    p = CACHE_DIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    ts_et = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(session=ts_et.dt.date, minute_et=ts_et.dt.strftime("%H:%M"))
    rth = df[(df["minute_et"] >= "09:30") & (df["minute_et"] <= "16:00")]
    if rth.empty:
        return pd.DataFrame()
    g = rth.groupby("session")
    out = pd.DataFrame({
        "open": g["o"].first(),
        "high": g["h"].max(),
        "low": g["l"].min(),
        "close": g["c"].last(),
        "volume": g["v"].sum(),
    }).reset_index().sort_values("session").reset_index(drop=True)
    return out


def atr_pct_series(daily: pd.DataFrame, lookback: int = ATR_LOOKBACK) -> pd.Series:
    """Rolling ATR as a % of close, shifted by 1 session (no lookahead)."""
    if daily.empty:
        return pd.Series(dtype=float)
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean()
    return (100.0 * atr / daily["close"]).shift(1)


def regime_vol_as_of(daily: pd.DataFrame, atr_pct: pd.Series, session: dt.date) -> float | None:
    """ATR% as known at the start of `session` (strictly prior sessions only)."""
    if daily.empty:
        return None
    idx = daily.index[daily["session"] == session]
    if len(idx) == 0:
        return None
    val = atr_pct.loc[idx[0]]
    return float(val) if pd.notna(val) else None


ATR_PCTILE_LOOKBACK = 60


def atr_percentile_series(atr_pct: pd.Series, lookback: int = ATR_PCTILE_LOOKBACK) -> pd.Series:
    """Rolling percentile rank (0-100) of atr_pct within its own trailing
    `lookback`-session history. `atr_pct` is already shifted by 1 session
    (no lookahead), so this is just a rolling rank of that series -- no
    further shift needed.

    Used to "regime-gate" the trailing-ATR exit: only treat the current
    session as a high-vol regime if its ATR% is itself elevated vs its
    recent history, per the item-5 finding that the trailing exit's edge
    is concentrated in a handful of high-vol episodes.
    """
    if atr_pct.empty or len(atr_pct) < lookback:
        return pd.Series(np.nan, index=atr_pct.index)

    values = atr_pct.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(values, lookback)  # (n-L+1, L)
    last = windows[:, -1]
    valid_mask = ~np.isnan(windows)
    valid_count = valid_mask.sum(axis=1)
    le_count = np.where(valid_mask & (windows <= last[:, None]), 1, 0).sum(axis=1)

    pct = 100.0 * le_count / np.where(valid_count > 0, valid_count, 1)
    pct = np.where((valid_count >= 2) & ~np.isnan(last), pct, np.nan)
    out[lookback - 1:] = pct
    return pd.Series(out, index=atr_pct.index)


def regime_pctile_as_of(daily: pd.DataFrame, atr_pctile: pd.Series, session: dt.date) -> float | None:
    """ATR%-percentile as known at the start of `session`."""
    if daily.empty:
        return None
    idx = daily.index[daily["session"] == session]
    if len(idx) == 0:
        return None
    val = atr_pctile.loc[idx[0]]
    return float(val) if pd.notna(val) else None
