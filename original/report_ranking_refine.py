"""
report_ranking_refine.py — refine the ranking/selection score from
report_ranking_selection.py.

The original score (signed_drift_pct x volume_ratio) was a positive OOS
finding for LONG, but price_drift_pct has extreme outliers (max ~14,000%),
which could distort a raw product. This script tests alternative score
formulas, all computed from the same drift_signal.parquet features:

  A. product        : signed_drift_pct * max(volume_ratio, 0)          (original)
  B. product_clip    : signed_drift_pct clipped to [-50, 50] * max(volume_ratio, 0)
  C. rank_drift      : rank of signed_drift_pct alone (within day/side/threshold group)
  D. rank_volume     : rank of volume_ratio alone
  E. rank_combo      : average of rank_drift and rank_volume (robust to outliers)

For each side, for each formula, for each top_n in [1,2,3,5,all], compute
TRAIN sum_ret_pct (trade_date < cutoff). Select the single best
(formula, top_n) combo per side by TRAIN sum_ret_pct (n_trades >= 10), then
report its TEST (held-out, last 4 months) performance, plus TEST for the
other formulas at the same top_n for context.

Only applied to the original fixed-TP rules (hold_above_open_945_tp3 /
failed_bounce_1000_tp_2.5), matching Expert's setup.

Writes output/ranking_refine.md.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "ranking_refine.md"

FIXED_RULE = {"LONG": "hold_above_open_945_tp3", "SHORT": "failed_bounce_1000_tp_2.5"}
TOP_NS = [1, 2, 3, 5, 999]  # 999 = "all"
TEST_MONTHS = 4
MIN_TRAIN_N = 10


def agg(g: pd.DataFrame) -> pd.Series:
    if len(g) == 0:
        return pd.Series({"n_trades": 0, "hit_rate_pct": np.nan, "avg_ret_pct": np.nan, "sum_ret_pct": 0.0})
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    if not TRADES.exists() or not DRIFT.exists():
        print("[report_ranking_refine] missing trades.parquet or drift_signal.parquet")
        return 1

    trades = pd.read_parquet(TRADES)
    drift = pd.read_parquet(DRIFT)

    df = trades.merge(
        drift,
        left_on=["settlement_date", "ticker"],
        right_on=["settlementDate", "symbolCode"],
        how="left",
    )
    df = df.dropna(subset=["price_drift_pct", "volume_ratio", "realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    max_date = df["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=TEST_MONTHS)

    df["signed_drift"] = np.where(df["side"] == "LONG", df["price_drift_pct"], -df["price_drift_pct"])
    df["signed_drift_clip"] = df["signed_drift"].clip(-50, 50)
    df["volume_ratio_pos"] = df["volume_ratio"].clip(lower=0)

    lines = [
        "# Ranking/Selection Score Refinement",
        "",
        f"- Full sample: {df['trade_date'].min().date()} -> {max_date.date()}",
        f"- TRAIN window: trade_date < {cutoff.date()}",
        f"- TEST window (held out): trade_date >= {cutoff.date()}",
        "",
        "Score formulas (all ranked descending within "
        "(trade_date, gap_threshold_pct) groups, per side):",
        "",
        "- **product** (original): `signed_drift_pct x max(volume_ratio, 0)`",
        "- **product_clip**: same, but `signed_drift_pct` clipped to [-50, 50] "
        "first (price_drift_pct has extreme outliers up to ~14,000%)",
        "- **rank_drift**: rank by `signed_drift_pct` alone",
        "- **rank_volume**: rank by `volume_ratio` alone",
        "- **rank_combo**: average of the drift-rank and volume-rank "
        "(robust, no outlier-sensitive arithmetic)",
        "",
        "Selection: for each side, pick the single best "
        "(formula, top_n) combo by TRAIN sum_ret_pct (n_trades >= 10), "
        "report its TEST result plus TEST for the other formulas at the "
        "same top_n for context.",
        "",
    ]

    for side in ["LONG", "SHORT"]:
        rule = FIXED_RULE[side]
        sub = df[(df["side"] == side) & (df["rule_name"] == rule)].copy()

        group_cols = ["trade_date", "gap_threshold_pct"]
        scores = {
            "product": sub["signed_drift"] * sub["volume_ratio_pos"],
            "product_clip": sub["signed_drift_clip"] * sub["volume_ratio_pos"],
            "rank_drift": sub.groupby(group_cols)["signed_drift"].rank(method="first", ascending=False),
            "rank_volume": sub.groupby(group_cols)["volume_ratio"].rank(method="first", ascending=False),
        }
        # rank_combo built from the two component ranks (lower avg rank = better)
        rank_drift_full = sub.groupby(group_cols)["signed_drift"].rank(method="average", ascending=False)
        rank_volume_full = sub.groupby(group_cols)["volume_ratio"].rank(method="average", ascending=False)
        scores["rank_combo"] = -(rank_drift_full + rank_volume_full)  # higher = better, for consistent "rank ascending=False" below

        records = []
        for fname, score in scores.items():
            sub["_score"] = score
            sub["_rank"] = sub.groupby(group_cols)["_score"].rank(method="first", ascending=False)
            for n in TOP_NS:
                label = "all" if n == 999 else str(n)
                for thr in [2.5, 3.0]:
                    mask_thr = sub["gap_threshold_pct"] == thr
                    tr = sub[(sub["trade_date"] < cutoff) & (sub["_rank"] <= n) & mask_thr]
                    te = sub[(sub["trade_date"] >= cutoff) & (sub["_rank"] <= n) & mask_thr]
                    tr_m = agg(tr)
                    te_m = agg(te)
                    records.append({
                        "formula": fname, "top_n": label, "gap_threshold_pct": thr,
                        "train_n": int(tr_m["n_trades"]), "train_sum_ret": tr_m["sum_ret_pct"],
                        "train_hit_rate": tr_m["hit_rate_pct"],
                        "test_n": int(te_m["n_trades"]), "test_sum_ret": te_m["sum_ret_pct"],
                        "test_hit_rate": te_m["hit_rate_pct"],
                    })

        rec_df = pd.DataFrame(records)

        lines.append(f"## {side} ({rule})")
        lines.append("")

        for thr in [2.5, 3.0]:
            tdf = rec_df[(rec_df["gap_threshold_pct"] == thr) & (rec_df["train_n"] >= MIN_TRAIN_N)]
            if tdf.empty:
                continue
            best = tdf.loc[tdf["train_sum_ret"].idxmax()]
            lines.append(f"### @{thr:.1f}%")
            lines.append("")
            lines.append(
                f"**Best on TRAIN:** formula=`{best['formula']}`, top_n={best['top_n']} "
                f"(train sum_ret={best['train_sum_ret']:.2f}%, n={best['train_n']}, "
                f"hit_rate={best['train_hit_rate']:.2f}%) "
                f"-> **TEST: sum_ret={best['test_sum_ret']:.2f}%, n={best['test_n']}, "
                f"hit_rate={best['test_hit_rate']:.2f}%**"
            )
            lines.append("")
            lines.append("All formulas at this top_n, TEST performance (context):")
            lines.append("")
            ctx = rec_df[(rec_df["gap_threshold_pct"] == thr) & (rec_df["top_n"] == best["top_n"])]
            ctx = ctx[["formula", "train_sum_ret", "train_n", "test_sum_ret", "test_n", "test_hit_rate"]]
            lines.append(ctx.to_markdown(index=False, floatfmt=".2f"))
            lines.append("")

        # Full table for reference
        lines.append("### Full TRAIN/TEST grid")
        lines.append("")
        full = rec_df[rec_df["train_n"] >= MIN_TRAIN_N].sort_values(["gap_threshold_pct", "formula", "top_n"])
        lines.append(full.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_ranking_refine] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
