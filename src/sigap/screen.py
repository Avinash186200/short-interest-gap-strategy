"""Short-interest screen, run once per FINRA settlement date.

Filters: days-to-cover >= 5, short interest >= 1M shares, ADV >= 100k,
listed (non-OTC) names, and short interest >= 10% of shares outstanding.
Survivors are ranked by 0.6 * pct-of-shares rank + 0.4 * days-to-cover rank
and the top N per date are kept.

Point-in-time shares: a shares-outstanding figure is only used once its
filing date is on or before the settlement date. The original code matched on
the reporting *period end* (a filing lag of weeks of look-ahead) and, when no
earlier figure existed, fell back to the first *future* one.
"""
from __future__ import annotations

import pandas as pd

MIN_DTC = 5.0
MIN_SI = 1_000_000
MIN_ADV = 100_000
MIN_SI_PCT_SHARES = 10.0
TOP_N = 150
OTC_CLASSES = {"OTCE", "OTC", "OTHER", "OTHERS"}


def shares_asof(shares: pd.DataFrame, asof: str | pd.Timestamp) -> pd.Series:
    """Latest *filed* shares-outstanding figure per symbol as of `asof` (no backfill)."""
    asof = pd.Timestamp(asof)
    known = shares[pd.to_datetime(shares["filed_date"]) <= asof]
    if known.empty:
        return pd.Series(dtype=float)
    known = known.sort_values(["symbol", "period_end", "filed_date"])
    return known.groupby("symbol")["shares"].last()


def screen_one_date(finra_date: pd.DataFrame, shares: pd.DataFrame, top_n: int = TOP_N,
                    include_otc: bool = False) -> pd.DataFrame:
    """Apply the screen to one settlement date's FINRA rows."""
    g = finra_date.copy()
    settlement = str(g["settlementDate"].iloc[0])
    num = ["currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"]
    g[num] = g[num].apply(pd.to_numeric, errors="coerce")
    g = g.dropna(subset=num)
    g = g[(g["daysToCoverQuantity"] >= MIN_DTC) & (g["currentShortPositionQuantity"] >= MIN_SI)
          & (g["averageDailyVolumeQuantity"] >= MIN_ADV)]
    if not include_otc:
        g = g[~g["marketClassCode"].isin(OTC_CLASSES)]
    g = g.assign(shares_out=g["symbolCode"].map(shares_asof(shares, settlement)))
    g = g[g["shares_out"] > 0]
    g = g.assign(si_pct_shares=100.0 * g["currentShortPositionQuantity"] / g["shares_out"])
    g = g[g["si_pct_shares"] >= MIN_SI_PCT_SHARES]
    if g.empty:
        return g
    score = 0.6 * g["si_pct_shares"].rank(pct=True) + 0.4 * g["daysToCoverQuantity"].rank(pct=True)
    g = g.assign(combined_score=100 * score).sort_values(["combined_score", "symbolCode"],
                                                         ascending=[False, True])
    return g.head(top_n).reset_index(drop=True)


def screen(finra: pd.DataFrame, shares: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Screen every settlement date; one row per (settlementDate, symbolCode)."""
    frames = [screen_one_date(g, shares, top_n) for _, g in finra.groupby("settlementDate")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(finra.columns) + ["shares_out", "si_pct_shares", "combined_score"])
    return pd.concat(frames, ignore_index=True)
