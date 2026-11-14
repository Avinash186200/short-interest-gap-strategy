"""Daily selection: cooling-off, then rank the day's eligible candidates and take the top N.

The strategy is run as a causal day-by-day loop, the way it would be traded:

1. Drop names that are cooling off (a losing trade on an *earlier* session
   within `cooloff_days` calendar days).
2. Rank what is left by the chosen key and take the top N.
3. Record today's losses so they affect only later sessions.

Because every decision uses only earlier sessions, one run over the full
sample can be sliced into any train/test window without leaking.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ranking name -> (column, ascending). None = take every candidate.
RANKINGS: dict[str, tuple[str, bool] | None] = {
    "drift_low": ("price_drift_pct", True),
    "drift_high": ("price_drift_pct", False),
    "volume_low": ("volume_ratio", True),
    "volume_high": ("volume_ratio", False),
    "random": None,
}


@dataclass(frozen=True)
class Config:
    side: str = "LONG"
    ranking: str = "drift_low"
    top_n: int | None = 5          # None = take all eligible candidates
    cooloff_days: int = 0

    @property
    def label(self) -> str:
        n = "all" if self.top_n is None else self.top_n
        return f"{self.side}:{self.ranking}:top{n}:cool{self.cooloff_days}"


def run_selection(cands: pd.DataFrame, cfg: Config, seed: int = 0) -> pd.DataFrame:
    """Return the trades actually taken under `cfg` (subset of `cands`)."""
    if cands.empty:
        return cands
    side = cands[cands["side"] == cfg.side].sort_values(["trade_date", "ticker"])
    if side.empty:
        return side
    rng = np.random.default_rng(seed)
    spec = RANKINGS[cfg.ranking]
    days = side["trade_date"].to_numpy()
    tickers = side["ticker"].to_numpy()
    net = side["net_ret"].to_numpy(float)
    key = side[spec[0]].to_numpy(float) * (1 if spec[1] else -1) if spec else None
    bounds = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    last_loss: dict[str, dt.date] = {}
    taken: list[int] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        day = days[a]
        rows = np.arange(a, b)
        if cfg.cooloff_days:
            rows = np.array([r for r in rows if tickers[r] not in last_loss
                             or (day - last_loss[tickers[r]]).days > cfg.cooloff_days], dtype=int)
        if len(rows) == 0:
            continue
        if cfg.top_n is not None and len(rows) > cfg.top_n:
            if key is None:
                rows = rows[rng.permutation(len(rows))[: cfg.top_n]]
            else:  # rows are already in ticker order, so a stable sort breaks ties by ticker
                rows = rows[np.argsort(key[rows], kind="stable")[: cfg.top_n]]
        taken.extend(rows.tolist())
        for r in rows:
            if net[r] < 0:
                last_loss[tickers[r]] = day
    return side.iloc[sorted(taken)]
