"""End-to-end run: short-interest screen -> gap candidates -> walk-forward + hold-out evaluation.

    python scripts/run_backtest.py                         # synthetic market (seed 7)
    python scripts/run_backtest.py --signal-strength 0     # no planted effect: should show no edge
    python scripts/run_backtest.py --data-dir my_data      # your own files (see README, "Data format")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sigap.screen import screen  # noqa: E402
from sigap.sources import ParquetBarSource  # noqa: E402
from sigap.synthetic import SyntheticMarket  # noqa: E402
from sigap.trades import RuleParams, build_candidates  # noqa: E402
from sigap.validation import evaluate_side  # noqa: E402


def load_inputs(args):
    if args.data_dir:
        d = Path(args.data_dir)
        finra = pd.read_csv(d / "finra_short_interest.csv")
        shares = pd.read_csv(d / "shares_outstanding.csv")
        source = ParquetBarSource(d / "bars")
        sessions = sorted({s for t in finra["symbolCode"].unique() for s in source.daily(t).index})
        label = str(d)
    else:
        mkt = SyntheticMarket(n_tickers=args.n_tickers, seed=args.seed, signal_strength=args.signal_strength)
        finra, shares, source, sessions = mkt.finra, mkt.shares, mkt, mkt.sessions
        label = f"synthetic (seed {args.seed}, signal_strength {args.signal_strength})"
    return finra, shares, source, sessions, label


def fmt(s: dict) -> str:
    return (f"trades {s['n_trades']:>5}  hit {s['hit_rate']:.1%}  avg {s['avg_net_ret_pct']:+.2f}%  "
            f"total {s['total_return']:+.1%}  Sharpe {s['sharpe']:+.2f}  maxDD {s['max_drawdown']:.1%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder with finra_short_interest.csv, shares_outstanding.csv, bars/")
    ap.add_argument("--n-tickers", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--signal-strength", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=2.5, help="gap threshold, percent (fixed in advance)")
    ap.add_argument("--cost-bps", type=float, default=15.0, help="cost per side, basis points")
    ap.add_argument("--holdout-months", type=int, default=4)
    ap.add_argument("--slots", type=int, default=5, help="capital is split into this many slots per day")
    ap.add_argument("--n-random", type=int, default=200, help="random top-N draws for the control")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    finra, shares, source, sessions, label = load_inputs(args)
    scr = screen(finra, shares)
    cands, cov = build_candidates(scr, source, RuleParams(gap_threshold=args.gap, cost_bps=args.cost_bps))
    print(f"input: {label}")
    print(f"screened windows {len(scr)}, coverage {cov}")
    print(f"candidates: {cands['side'].value_counts().to_dict() if not cands.empty else {}}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"input": label, "gap_threshold": args.gap, "cost_bps_per_side": args.cost_bps,
               "coverage": cov, "sides": {}}
    for side in ("LONG", "SHORT"):
        res = evaluate_side(cands, side, sessions, args.holdout_months, args.slots, args.n_random, args.seed)
        summary["sides"][side] = res
        print(f"\n== {side}  ({res['n_candidates']} candidates, {res['n_configs']} configs) ==")
        print(f"naive in-sample best  {res['naive_in_sample']['config']:<28} {fmt(res['naive_in_sample'])}"
              f"  DSR p={res['naive_in_sample']['deflated_sharpe_prob']:.2f}")
        if res["walk_forward_oos"].get("n_trades") is not None:
            print(f"walk-forward OOS      {'(re-chosen each quarter)':<28} {fmt(res['walk_forward_oos'])}")
        h = res["holdout"]
        print(f"hold-out, chosen      {h['config']:<28} {fmt(h)}")
        print(f"   beats {h['pct_of_random_draws_beaten']:.0%} of random top-N draws "
              f"(random median Sharpe {h['random_control_median_sharpe']:+.2f})")
        print(f"hold-out, take all    {'':<28} {fmt(res['holdout_take_all'])}")
        if "holdout_pre_registered" in res:
            p = res["holdout_pre_registered"]
            print(f"hold-out, pre-reg     {p['config']:<28} {fmt(p)}")
    cands.to_csv(out / "candidates.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
