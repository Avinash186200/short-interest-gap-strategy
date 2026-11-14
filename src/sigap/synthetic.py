"""Synthetic small-cap market with FINRA-style short-interest data.

Everything the pipeline reads is generated here with the same columns as the
real inputs:

* a FINRA consolidated short-interest table (`settlementDate`, `symbolCode`,
  `currentShortPositionQuantity`, `averageDailyVolumeQuantity`,
  `daysToCoverQuantity`, `marketClassCode`),
* a shares-outstanding table with the period end AND the filing date,
* daily open/close/volume and on-demand regular-hours 1-minute bars.

A known effect is planted so the pipeline has something real to find: in the
10 sessions after a heavily shorted name's data is published, names whose
price fell the most between settlement and publication drift up intraday.
`signal_strength=0` removes it, which the tests use as a no-edge control.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import first_tradable_session, trading_sessions
from .sources import RTH_MINUTES

HOLD_DAYS = 10
MARKET_CLASSES = np.array(["NNM", "NYSE", "SC", "OTC"])


def settlement_dates(sessions: pd.DatetimeIndex) -> list[dt.date]:
    """Mid-month and month-end settlement dates, rolled back onto a session."""
    days = [d.date() for d in sessions]
    s = pd.Series(days, index=sessions)
    out: list[dt.date] = []
    for _, month in s.groupby([sessions.year, sessions.month]):
        mid = month[month.index.day <= 15]
        if len(mid):
            out.append(mid.iloc[-1])
        out.append(month.iloc[-1])
    return sorted(set(out))


@dataclass
class SyntheticMarket:
    """Generates and serves a reproducible synthetic market (implements BarSource)."""

    n_tickers: int = 150
    start: str = "2024-01-02"
    end: str = "2026-05-29"
    seed: int = 7
    signal_strength: float = 1.0
    finra: pd.DataFrame = field(init=False, repr=False)
    shares: pd.DataFrame = field(init=False, repr=False)
    sessions: list[dt.date] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._generate()

    # ------------------------------------------------------------------ build
    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        idx = trading_sessions(self.start, self.end)
        self.sessions = [d.date() for d in idx]
        self._pos = {d: k for k, d in enumerate(self.sessions)}
        n, T = self.n_tickers, len(self.sessions)
        self.tickers = [f"SYN{i:03d}" for i in range(n)]
        self._tix = {t: i for i, t in enumerate(self.tickers)}

        price0 = np.exp(rng.uniform(np.log(1.5), np.log(40.0), n))
        self._sig_on = rng.uniform(0.02, 0.045, n)
        self._sig_oc = rng.uniform(0.03, 0.06, n)
        beta = rng.uniform(0.5, 1.5, n)
        turnover = rng.uniform(0.005, 0.03, n)
        mkt_class = rng.choice(MARKET_CLASSES, n, p=[0.45, 0.25, 0.2, 0.1])

        # shares outstanding: step changes (dilution events) over time
        shares = np.empty((T, n))
        shares[0] = np.exp(rng.uniform(np.log(20e6), np.log(300e6), n))
        for t in range(1, T):
            event = rng.random(n) < 0.15 / 21
            shares[t] = shares[t - 1] * np.where(event, 1 + rng.uniform(0.05, 0.3, n), 1.0)

        settles = settlement_dates(idx)
        pos = {d: i for i, d in enumerate(self.sessions)}
        si_frac = np.clip(rng.normal(0.12, 0.06, n), 0.01, 0.45)
        si_by_settle: dict[dt.date, np.ndarray] = {}
        for s in settles:
            si_frac = np.clip(si_frac + rng.normal(0, 0.02, n), 0.01, 0.45)
            si_by_settle[s] = si_frac.copy()

        # post-publication windows: session index -> list of (settle index, eligible mask)
        windows: dict[int, tuple[int, int, np.ndarray]] = {}
        for s in settles:
            first = first_tradable_session(s, self.sessions)
            if first is None:
                continue
            w0 = pos[first]
            eligible = si_by_settle[s] >= 0.10
            for k in range(w0, min(w0 + HOLD_DAYS, T)):
                windows[k] = (pos[s], w0, eligible)  # later settlement overwrites earlier

        opens = np.empty((T, n))
        closes = np.empty((T, n))
        vols = np.empty((T, n))
        prev = price0
        for t in range(T):
            m_on, m_oc = 0.007 * rng.standard_t(3), rng.normal(0, 0.008)
            gap = beta * m_on + self._sig_on * rng.standard_t(4, n) / np.sqrt(2)
            alpha = np.zeros(n)
            if t in windows and self.signal_strength:
                si_idx, w0, eligible = windows[t]
                drift = closes[w0 - 1] / closes[si_idx] - 1.0
                alpha = self.signal_strength * 0.03 * np.clip(-drift / 0.15, 0.0, 1.0) * eligible
            oc = alpha + beta * m_oc + self._sig_oc * rng.standard_normal(n)
            opens[t] = np.maximum(prev * np.exp(gap), 0.05)
            closes[t] = np.maximum(opens[t] * np.exp(oc), 0.05)
            vols[t] = shares[t] * turnover * rng.lognormal(0, 0.4, n) * (1 + 10 * np.abs(gap))
            prev = closes[t]
        self._open, self._close, self._vol = opens, closes, vols

        rows = []
        for s in settles:
            k = pos[s]
            adv = vols[max(0, k - 19):k + 1].mean(axis=0)
            si = si_by_settle[s] * shares[k]
            for i in range(n):
                rows.append((s.isoformat(), self.tickers[i], round(si[i]), round(adv[i]),
                             round(si[i] / adv[i], 2), mkt_class[i]))
        self.finra = pd.DataFrame(rows, columns=[
            "settlementDate", "symbolCode", "currentShortPositionQuantity",
            "averageDailyVolumeQuantity", "daysToCoverQuantity", "marketClassCode"])

        q_ends = pd.date_range(pd.Timestamp(self.start) - pd.offsets.QuarterEnd(1), self.end, freq="QE")
        srows = []
        for qe in q_ends:
            k = max(0, np.searchsorted(idx, qe, side="right") - 1)
            filed = (qe + pd.Timedelta(days=40)).date()
            for i in range(n):
                srows.append((self.tickers[i], qe.date().isoformat(), filed.isoformat(), float(shares[k, i])))
        self.shares = pd.DataFrame(srows, columns=["symbol", "period_end", "filed_date", "shares"])

    # -------------------------------------------------------------- BarSource
    def daily(self, ticker: str) -> pd.DataFrame:
        i = self._tix.get(ticker)
        if i is None:
            return pd.DataFrame(columns=["open", "close", "volume"])
        return pd.DataFrame({"open": self._open[:, i], "close": self._close[:, i],
                             "volume": self._vol[:, i]}, index=pd.Index(self.sessions, name="session"))

    def session_bars(self, ticker: str, session: dt.date) -> pd.DataFrame | None:
        i = self._tix.get(ticker)
        t = self._pos.get(session)
        if i is None or t is None:
            return None
        o, h, l, c, v = self.bar_arrays(i, t)
        return pd.DataFrame({"minute": np.arange(RTH_MINUTES), "o": o, "h": h, "l": l, "c": c, "v": v})

    def bar_arrays(self, i: int, t: int) -> tuple[np.ndarray, ...]:
        """Deterministic Brownian-bridge 1-minute path from the day's open to its close."""
        rng = np.random.default_rng([self.seed, i, t])
        lo, lc = np.log(self._open[t, i]), np.log(self._close[t, i])
        step = self._sig_oc[i] / np.sqrt(RTH_MINUTES)
        w = np.concatenate([[0.0], np.cumsum(rng.normal(0, step, RTH_MINUTES))])
        frac = np.arange(RTH_MINUTES + 1) / RTH_MINUTES
        path = np.exp(lo + frac * (lc - lo) + w - frac * w[-1])
        o, c = path[:-1], path[1:]
        wick = np.exp(np.abs(rng.normal(0, step * 0.5, (2, RTH_MINUTES))))
        h = np.maximum(o, c) * wick[0]
        l = np.minimum(o, c) / wick[1]
        v = self._vol[t, i] / RTH_MINUTES * rng.lognormal(0, 0.5, RTH_MINUTES)
        return o, h, l, c, v


def write_dataset(mkt: SyntheticMarket, out_dir, tickers: list[str] | None = None) -> None:
    """Write the market in the on-disk format `run_backtest.py --data-dir` reads.

    Files: finra_short_interest.csv, shares_outstanding.csv and bars/<TICKER>.parquet
    (1-minute regular-hours bars). Writing every ticker's bars is large, so
    `tickers` limits the output to a subset (the FINRA and shares tables are
    filtered to match).
    """
    from pathlib import Path

    from .sources import bars_to_frame

    out = Path(out_dir)
    (out / "bars").mkdir(parents=True, exist_ok=True)
    keep = tickers or mkt.tickers
    mkt.finra[mkt.finra["symbolCode"].isin(keep)].to_csv(out / "finra_short_interest.csv", index=False)
    mkt.shares[mkt.shares["symbol"].isin(keep)].to_csv(out / "shares_outstanding.csv", index=False)
    for tkr in keep:
        i = mkt.tickers.index(tkr)
        frames = [bars_to_frame(s, *mkt.bar_arrays(i, t)) for t, s in enumerate(mkt.sessions)]
        pd.concat(frames, ignore_index=True).to_parquet(out / "bars" / f"{tkr}.parquet", index=False)
