"""Write a small synthetic dataset in the on-disk format the backtest reads.

    python scripts/make_synthetic_data.py --n-tickers 20 --out data/synthetic
    python scripts/run_backtest.py --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sigap.synthetic import SyntheticMarket, write_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    a = ap.parse_args()
    mkt = SyntheticMarket(n_tickers=a.n_tickers, start=a.start, end=a.end, seed=a.seed)
    write_dataset(mkt, a.out)
    print(f"wrote {a.n_tickers} tickers, {len(mkt.sessions)} sessions to {a.out}")
