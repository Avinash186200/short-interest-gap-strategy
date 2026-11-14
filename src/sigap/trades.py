"""Candidate trades: the gap rules applied in each post-publication window.

For every screened (settlement date, ticker), look at the first `hold_days`
sessions after the data becomes public. On a session where the stock gaps up
by at least `gap_threshold` percent:

* LONG  "hold above open": if the 09:45 bar closes at or above the 09:30 open,
  buy at the open of the next bar (09:46). Exit at +tp_long % or the close.
* SHORT "failed bounce": if the 10:00 bar closes below both the open and the
  09:45 close, sell at the open of the next bar (10:01). Exit at -tp_short %
  or the close.

Every candidate also carries the pre-publication features used for ranking:
`price_drift_pct` (settlement close -> last close before the first tradable
session) and `volume_ratio` (average volume over the same span / FINRA ADV).
Both use only closes that are known before the first trade in the window.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import PUBLICATION_LAG_BUSINESS_DAYS, publication_date
from .sources import BarSource

LONG_SIGNAL_BAR = 15   # 09:45 bar
SHORT_SIGNAL_BAR = 30  # 10:00 bar


@dataclass(frozen=True)
class RuleParams:
    gap_threshold: float = 2.5
    hold_days: int = 10
    tp_long: float = 3.0
    tp_short: float = 2.5
    stop_pct: float | None = None   # optional hard stop, in percent
    cost_bps: float = 15.0          # per side; charged twice per round trip
    pub_lag: int = PUBLICATION_LAG_BUSINESS_DAYS


def simulate_exit(h: np.ndarray, l: np.ndarray, c: np.ndarray, entry: float, side: str,
                  tp_pct: float, stop_pct: float | None = None) -> tuple[float, int, str]:
    """Walk bars from the entry bar on. Returns (exit price, bar offset, reason).

    If the stop and the target are both inside one bar, the stop is assumed
    to fill first (conservative).
    """
    n = len(c)
    if side == "LONG":
        tp_px = entry * (1 + tp_pct / 100)
        tp_hit = np.flatnonzero(h >= tp_px)
        sl_px = entry * (1 - stop_pct / 100) if stop_pct else None
        sl_hit = np.flatnonzero(l <= sl_px) if sl_px else np.array([], dtype=int)
    else:
        tp_px = entry * (1 - tp_pct / 100)
        tp_hit = np.flatnonzero(l <= tp_px)
        sl_px = entry * (1 + stop_pct / 100) if stop_pct else None
        sl_hit = np.flatnonzero(h >= sl_px) if sl_px else np.array([], dtype=int)
    i_tp = tp_hit[0] if len(tp_hit) else n
    i_sl = sl_hit[0] if len(sl_hit) else n
    if i_sl < n and i_sl <= i_tp:
        return float(sl_px), int(i_sl), "stop"
    if i_tp < n:
        return float(tp_px), int(i_tp), "take_profit"
    return float(c[-1]), n - 1, "close"


@dataclass
class DailyArrays:
    """One ticker's daily table as sorted numpy arrays (fast window lookups)."""

    dates: np.ndarray   # datetime64[D]
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_frame(cls, daily: pd.DataFrame) -> "DailyArrays":
        d = pd.to_datetime(pd.Index(daily.index)).values.astype("datetime64[D]")
        return cls(d, daily["open"].to_numpy(float), daily["close"].to_numpy(float),
                   daily["volume"].to_numpy(float))


def window_features(daily: DailyArrays, settlement: dt.date, first_idx: int, adv: float) -> dict | None:
    """Pre-publication drift and volume features, from closes before session `first_idx`."""
    k_s = int(np.searchsorted(daily.dates, np.datetime64(settlement, "D"), side="right")) - 1
    k_e = first_idx - 1
    if k_s < 0 or k_e < k_s or daily.close[k_s] <= 0:
        return None
    return {
        "pub_eve_date": daily.dates[k_e].astype(object),
        "price_drift_pct": 100.0 * (daily.close[k_e] / daily.close[k_s] - 1.0),
        "volume_ratio": float(daily.volume[k_s:k_e + 1].mean()) / adv if adv > 0 else np.nan,
    }


def build_candidates(screened: pd.DataFrame, source: BarSource, params: RuleParams = RuleParams()
                     ) -> tuple[pd.DataFrame, dict]:
    """All LONG and SHORT candidate trades plus data-coverage counts."""
    rows: list[dict] = []
    cov = {"windows": 0, "no_price_data": 0, "gap_sessions": 0, "missing_bars": 0, "stale_prev_close": 0}
    cost = 2 * params.cost_bps / 1e4
    daily_cache: dict[str, DailyArrays] = {}
    for rec in screened.itertuples(index=False):
        cov["windows"] += 1
        tkr = rec.symbolCode
        settlement = pd.Timestamp(rec.settlementDate).date()
        if tkr not in daily_cache:
            daily_cache[tkr] = DailyArrays.from_frame(source.daily(tkr))
        daily = daily_cache[tkr]
        pub = publication_date(settlement, params.pub_lag)
        first = int(np.searchsorted(daily.dates, np.datetime64(pub, "D"), side="right"))
        feats = (window_features(daily, settlement, first, float(rec.averageDailyVolumeQuantity))
                 if first < len(daily.dates) else None)
        if feats is None:
            cov["no_price_data"] += 1
            continue
        for k in range(first, min(first + params.hold_days, len(daily.dates))):
            if k == 0:
                continue
            if (daily.dates[k] - daily.dates[k - 1]).astype(int) > 5:
                cov["stale_prev_close"] += 1  # a hole in the data; the gap would be wrong
                continue
            gap = 100.0 * (daily.open[k] / daily.close[k - 1] - 1.0)
            if gap < params.gap_threshold:
                continue
            cov["gap_sessions"] += 1
            sess = daily.dates[k].astype(object)
            bars = source.session_bars(tkr, sess)
            if bars is None or len(bars) <= SHORT_SIGNAL_BAR + 1:
                cov["missing_bars"] += 1
                continue
            o, h, l, c = (bars[x].to_numpy(float) for x in ("o", "h", "l", "c"))
            base = {"trade_date": sess, "ticker": tkr, "settlement_date": settlement,
                    "publication_date": pub, "gap_pct": float(gap), **feats}
            open_px, p945, p1000 = o[0], c[LONG_SIGNAL_BAR], c[SHORT_SIGNAL_BAR]
            if p945 >= open_px:
                e = LONG_SIGNAL_BAR + 1
                px, off, why = simulate_exit(h[e:], l[e:], c[e:], o[e], "LONG",
                                             params.tp_long, params.stop_pct)
                gross = px / o[e] - 1.0
                rows.append({**base, "side": "LONG", "entry_bar": e, "entry_px": o[e], "exit_px": px,
                             "exit_bar": e + off, "exit_reason": why,
                             "gross_ret": gross, "net_ret": gross - cost})
            if p1000 < open_px and p1000 < p945:
                e = SHORT_SIGNAL_BAR + 1
                px, off, why = simulate_exit(h[e:], l[e:], c[e:], o[e], "SHORT",
                                             params.tp_short, params.stop_pct)
                gross = 1.0 - px / o[e]
                rows.append({**base, "side": "SHORT", "entry_bar": e, "entry_px": o[e], "exit_px": px,
                             "exit_bar": e + off, "exit_reason": why,
                             "gross_ret": gross, "net_ret": gross - cost})
    cands = pd.DataFrame(rows)
    if cands.empty:
        return cands, cov
    # a name in two overlapping windows would otherwise be traded twice on one day
    cands = (cands.sort_values(["trade_date", "ticker", "side", "settlement_date"])
             .drop_duplicates(["trade_date", "ticker", "side"], keep="last")
             .reset_index(drop=True))
    return cands, cov
