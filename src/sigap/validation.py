"""Selection-bias-aware evaluation.

The original research tried dozens of variants (ranking formula and direction,
top-N, cooling-off length, stops, filters, exit rules) and judged each one on
the same 4-month "TEST" window. Once a hold-out has been looked at that many
times it is no longer a hold-out. This module replaces that with:

1. **Naive in-sample** (for contrast only): the best grid config on the full
   sample, plus its deflated Sharpe given how many configs were tried.
2. **Walk-forward**: an expanding window over the research period. At each
   step the config with the best *training* Sharpe is chosen and traded on
   the next block. The out-of-sample blocks are stitched together.
3. **Final hold-out**: the last `holdout_months`, never used for any choice.
   It is scored once, for the config chosen on the whole research period and
   for the pre-registered spec.
4. **Random top-N control**: the same top-N and cooling-off, but picking the
   day's names at random. If the ranking key is real, it should beat most
   random draws. This separates "drift ranking works" from "trading fewer
   names on busy days works".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import daily_returns, deflated_sharpe, sharpe, summarize
from .selection import Config, run_selection

RANKS = ["drift_low", "drift_high", "volume_low", "volume_high"]
TOP_NS: list[int | None] = [1, 2, 3, 5, None]
COOLOFFS = [0, 5]
PRE_REGISTERED = Config("LONG", "drift_low", 5, 5)


def grid(side: str) -> list[Config]:
    """Every config the research considered, for one side."""
    out = [Config(side, r, n, c) for r in RANKS for n in TOP_NS if n is not None for c in COOLOFFS]
    out += [Config(side, "drift_low", None, c) for c in COOLOFFS]  # "take all" (ranking irrelevant)
    return out


@dataclass
class Run:
    cfg: Config
    trades: pd.DataFrame
    daily: pd.Series


def run_config(cands: pd.DataFrame, cfg: Config, sessions: list, slots: int, seed: int = 0) -> Run:
    tr = run_selection(cands, cfg, seed=seed)
    return Run(cfg, tr, daily_returns(tr, sessions, slots))


def run_grid(cands: pd.DataFrame, side: str, sessions: list, slots: int) -> dict[Config, Run]:
    return {c: run_config(cands, c, sessions, slots) for c in grid(side)}


def split_holdout(sessions: list, holdout_months: int) -> tuple[list, list]:
    cut = (pd.Timestamp(sessions[-1]) - pd.DateOffset(months=holdout_months)).date()
    return [s for s in sessions if s < cut], [s for s in sessions if s >= cut]


def choose(runs: dict[Config, Run], window: list) -> Config:
    """Best Sharpe over `window`; ties broken by label so the choice is deterministic."""
    scored = [(sharpe(r.daily.loc[window]), c.label, c) for c, r in runs.items()]
    return max(scored, key=lambda x: (x[0], x[1]))[2]


def slice_run(run: Run, window: list) -> tuple[pd.Series, pd.DataFrame]:
    lo, hi = window[0], window[-1]
    tr = run.trades[(run.trades["trade_date"] >= lo) & (run.trades["trade_date"] <= hi)] \
        if not run.trades.empty else run.trades
    return run.daily.loc[window], tr


def walk_forward(runs: dict[Config, Run], research: list, min_train: int = 252,
                 step: int = 63) -> dict:
    """Expanding-window selection; returns stitched OOS returns and the per-step choices."""
    oos_r, oos_t, steps = [], [], []
    for start in range(min_train, len(research), step):
        train, test = research[:start], research[start:start + step]
        cfg = choose(runs, train)
        r, t = slice_run(runs[cfg], test)
        oos_r.append(r)
        oos_t.append(t)
        steps.append({"train_end": str(train[-1]), "test_start": str(test[0]), "test_end": str(test[-1]),
                      "chosen": cfg.label, "train_sharpe": round(sharpe(runs[cfg].daily.loc[train]), 3),
                      "test_sharpe": round(sharpe(r), 3)})
    if not oos_r:
        return {"oos": pd.Series(dtype=float), "trades": pd.DataFrame(), "steps": []}
    return {"oos": pd.concat(oos_r), "trades": pd.concat(oos_t), "steps": steps}


def random_control(cands: pd.DataFrame, cfg: Config, sessions: list, window: list,
                   slots: int, n_draws: int = 200, seed: int = 0) -> np.ndarray:
    """Sharpe over `window` of `n_draws` random-pick runs with cfg's top-N and cooling-off."""
    rnd = Config(cfg.side, "random", cfg.top_n, cfg.cooloff_days)
    out = []
    for k in range(n_draws):
        run = run_config(cands, rnd, sessions, slots, seed=seed + k)
        out.append(sharpe(run.daily.loc[window]))
    return np.asarray(out)


def evaluate_side(cands: pd.DataFrame, side: str, sessions: list, holdout_months: int = 4,
                  slots: int = 5, n_random: int = 200, seed: int = 0) -> dict:
    """Full honest evaluation for one side. Returns a JSON-friendly dict."""
    runs = run_grid(cands, side, sessions, slots)
    research, holdout = split_holdout(sessions, holdout_months)

    # 1. naive: best config on everything, as the original reports effectively did
    naive = choose(runs, sessions)
    trial_sr = [sharpe(r.daily) for r in runs.values()]
    naive_stats = summarize(runs[naive].daily, runs[naive].trades)
    naive_stats["deflated_sharpe_prob"] = deflated_sharpe(runs[naive].daily, len(runs), trial_sr)

    # 2. walk-forward inside the research period
    wf = walk_forward(runs, research)
    wf_stats = summarize(wf["oos"], wf["trades"]) if len(wf["oos"]) else {}

    # 3. final hold-out, scored once
    chosen = choose(runs, research)
    r_h, t_h = slice_run(runs[chosen], holdout)
    hold_stats = summarize(r_h, t_h)
    take_all = Config(side, "drift_low", None, 0)
    r_a, t_a = slice_run(runs[take_all], holdout)

    # 4. random top-N control on the hold-out
    rnd = random_control(cands, chosen, sessions, holdout, slots, n_random, seed) \
        if chosen.top_n is not None and n_random else np.array([])
    pct = float((rnd < hold_stats["sharpe"]).mean()) if len(rnd) else float("nan")

    out = {
        "side": side,
        "n_candidates": int((cands["side"] == side).sum()) if not cands.empty else 0,
        "n_configs": len(runs),
        "research_period": [str(research[0]), str(research[-1])],
        "holdout_period": [str(holdout[0]), str(holdout[-1])],
        "naive_in_sample": {"config": naive.label, **naive_stats},
        "walk_forward_oos": {**wf_stats, "steps": wf["steps"]},
        "holdout": {"config": chosen.label, **hold_stats,
                    "random_control_median_sharpe": float(np.median(rnd)) if len(rnd) else float("nan"),
                    "pct_of_random_draws_beaten": pct},
        "holdout_take_all": summarize(r_a, t_a),
    }
    if side == PRE_REGISTERED.side:
        r_p, t_p = slice_run(runs[PRE_REGISTERED], holdout)
        out["holdout_pre_registered"] = {"config": PRE_REGISTERED.label, **summarize(r_p, t_p)}
    return out
