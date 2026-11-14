"""Price-data sources.

The backtest needs two things per ticker: a daily table (open, close, volume
per session) and, for the few sessions that gap, the regular-hours 1-minute
bars. Both are served through the small `BarSource` interface so the same
pipeline runs on the synthetic market or on cached vendor bars.

On-disk format for `ParquetBarSource` (one file per ticker, `<TICKER>.parquet`):
columns `t` (bar start, unix seconds UTC), `o`, `h`, `l`, `c`, `v`.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

RTH_MINUTES = 390  # 09:30 .. 15:59 bar starts


class BarSource(Protocol):
    def daily(self, ticker: str) -> pd.DataFrame:
        """Index: session (datetime.date). Columns: open, close, volume."""

    def session_bars(self, ticker: str, session: dt.date) -> pd.DataFrame | None:
        """Regular-hours 1-minute bars for one session, or None if missing.

        Columns: minute (0 = 09:30 bar), o, h, l, c, v. Sorted by minute.
        """


class ParquetBarSource:
    """Reads `<dir>/<TICKER>.parquet` 1-minute bar files."""

    def __init__(self, bar_dir: str | Path) -> None:
        self.bar_dir = Path(bar_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _rth(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        p = self.bar_dir / f"{ticker}.parquet"
        if not p.exists():
            df = pd.DataFrame(columns=["session", "minute", "o", "h", "l", "c", "v"])
        else:
            raw = pd.read_parquet(p)
            ts = pd.to_datetime(raw["t"], unit="s", utc=True).dt.tz_convert("America/New_York")
            minute = (ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)
            df = raw.assign(session=ts.dt.date, minute=minute)
            df = df[(df["minute"] >= 0) & (df["minute"] < RTH_MINUTES)]
            df = df.sort_values(["session", "minute"]).reset_index(drop=True)
        self._cache[ticker] = df
        return df

    def daily(self, ticker: str) -> pd.DataFrame:
        rth = self._rth(ticker)
        if rth.empty:
            return pd.DataFrame(columns=["open", "close", "volume"])
        g = rth.groupby("session")
        return pd.DataFrame({"open": g["o"].first(), "close": g["c"].last(), "volume": g["v"].sum()})

    def session_bars(self, ticker: str, session: dt.date) -> pd.DataFrame | None:
        rth = self._rth(ticker)
        out = rth[rth["session"] == session]
        if out.empty:
            return None
        return out[["minute", "o", "h", "l", "c", "v"]].reset_index(drop=True)


def bars_to_frame(session: dt.date, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                  c: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """Turn RTH bar arrays into the vendor file format (unix-second `t`)."""
    start = pd.Timestamp(session).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    t = start + pd.to_timedelta(np.arange(len(o)), unit="min")
    secs = (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return pd.DataFrame({"t": np.asarray(secs, dtype="int64"), "o": o, "h": h, "l": l, "c": c, "v": v})
