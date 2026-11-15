"""
report_liquidity_floor_test.py — tests whether excluding low-priced /
illiquid names from the primary LONG drift_low top-5 strategy improves the
result, motivated by the drawdown investigation (output/drawdown_investigation.md)
which found illiquid small/micro-cap names (SKYQ at $0.45, SBH, OMER, ...)
drove most of the 2025-10->2026-03 drawdown.

Tests a minimum entry_price floor in {1, 1.5, 2, 3, 5} dollars on TRAIN
(trade_date < 2026-01-22), reports TEST (last 4 months) sum_ret for the
TRAIN-selected floor vs the no-floor baseline. Train/test disciplined per
project convention.

Writes output/liquidity_floor_test.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
OUT_MD = ROOT / "output" / "liquidity_floor_test.md"

FLOORS = [1.0, 1.5, 2.0, 3.0, 5.0]
TRAIN_CUTOFF = "2026-01-22"


def hit(s):
    return (s > 0).mean() * 100


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[liqfloor] missing {TRADES_SEL} - run select_signals.py first")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[df["selected"]].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    train = sel[sel["trade_date"] < TRAIN_CUTOFF]
    test = sel[sel["trade_date"] >= TRAIN_CUTOFF]

    lines = [
        "# Minimum entry-price (liquidity) floor test (LONG drift_low top-5)",
        "",
        "Excludes selected trades with entry_price below a floor, motivated "
        "by the drawdown investigation finding that illiquid small/micro-cap "
        "names drove most of the 2025-10->2026-03 drawdown. Train/test "
        "disciplined: floor chosen on TRAIN (trade_date < 2026-01-22), "
        "evaluated once on TEST.",
        "",
        "## TRAIN vs TEST sum_ret by floor",
        "",
        "| floor ($) | TRAIN n | TRAIN sum_ret | TRAIN hit% | TEST n | TEST sum_ret | TEST hit% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    tr0 = train["realized_return_pct"]
    te0 = test["realized_return_pct"]
    lines.append(
        f"| none (baseline) | {len(train)} | {tr0.sum():.2f} | {hit(tr0):.2f} | "
        f"{len(test)} | {te0.sum():.2f} | {hit(te0):.2f} |"
    )
    for floor in FLOORS:
        tr = train[train["entry_price"] >= floor]["realized_return_pct"]
        te = test[test["entry_price"] >= floor]["realized_return_pct"]
        lines.append(
            f"| {floor} | {len(tr)} | {tr.sum():.2f} | {hit(tr):.2f} | "
            f"{len(te)} | {te.sum():.2f} | {hit(te):.2f} |"
        )
    lines.append("")

    best_floor = max(FLOORS, key=lambda f: train[train["entry_price"] >= f]["realized_return_pct"].sum())
    train_best = train[train["entry_price"] >= best_floor]["realized_return_pct"].sum()
    test_best = test[test["entry_price"] >= best_floor]["realized_return_pct"].sum()
    lines.append(
        f"**TRAIN-selected floor: \\${best_floor}** "
        f"(TRAIN sum_ret {tr0.sum():.2f} -> {train_best:.2f}). "
        f"On TEST: baseline {te0.sum():.2f} -> with floor {test_best:.2f}."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[liqfloor] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
