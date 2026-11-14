"""Portfolio returns and statistics.

The original research summed per-trade percentage returns ("sum_ret") with no
capital constraint, which produced drawdowns like -300% and Sharpe ratios that
do not correspond to any tradable account. Here capital is split into a fixed
number of slots: each trade gets 1 / max(slots, trades that day) of equity,
returns compound, and every session in the calendar counts (flat days are 0).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

ANN = 252


def daily_returns(trades: pd.DataFrame, sessions: list, slots: int = 5) -> pd.Series:
    """Daily portfolio return over `sessions` from intraday trades (net of costs)."""
    idx = pd.Index(sessions, name="session")
    if trades.empty:
        return pd.Series(0.0, index=idx)
    g = trades.groupby("trade_date")["net_ret"]
    day = g.sum() / np.maximum(slots, g.count())
    return day.reindex(idx, fill_value=0.0)


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(ANN)) if sd > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def summarize(r: pd.Series, trades: pd.DataFrame) -> dict:
    """Headline statistics for one daily-return series and its trades."""
    n = len(trades)
    years = max(len(r) / ANN, 1e-9)
    total = float((1 + r).prod() - 1)
    return {
        "n_trades": n,
        "hit_rate": float((trades["net_ret"] > 0).mean()) if n else float("nan"),
        "avg_net_ret_pct": float(100 * trades["net_ret"].mean()) if n else float("nan"),
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
    }


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes: list[float]) -> float:
    """Probability that the true Sharpe exceeds the best-of-`n_trials` noise level.

    Bailey and Lopez de Prado (2014). Sharpe values are per period (not
    annualised) inside the formula; `trial_sharpes` are annualised and are
    converted.
    """
    T = len(r)
    sd = r.std(ddof=1)
    if T < 3 or sd == 0 or n_trials < 1:
        return float("nan")
    sr = r.mean() / sd
    var_trials = float(np.var(np.asarray(trial_sharpes) / math.sqrt(ANN), ddof=1)) if len(trial_sharpes) > 1 else 0.0
    nd = NormalDist()
    gamma = 0.5772156649
    if n_trials > 1:
        sr0 = math.sqrt(var_trials) * ((1 - gamma) * nd.inv_cdf(1 - 1 / n_trials)
                                       + gamma * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    skew = float(((r - r.mean()) ** 3).mean() / sd ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / sd ** 4)
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(nd.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
