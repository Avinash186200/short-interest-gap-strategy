"""
report_ranking_refine2.py — CORRECTED re-run of report_ranking_refine.py.

report_ranking_refine.py had a bug: its "rank_drift" and "rank_volume"
formulas computed a rank-of-the-rank (double rank), which inverted the
selection — "top_n" by that formula actually selected the BOTTOM-N
candidates (worst drift / worst volume), not the top-N. This script
recomputes both directions explicitly and unambiguously:

  - drift_high : top-N by signed_drift_pct, HIGHEST first (the originally
                 intended "rank_drift" - drift confirms LONG direction)
  - drift_low  : top-N by signed_drift_pct, LOWEST first (what
                 report_ranking_refine.py's buggy "rank_drift" actually
                 computed)
  - volume_high: top-N by volume_ratio, HIGHEST first (originally intended
                 "rank_volume")
  - volume_low : top-N by volume_ratio, LOWEST first (what the buggy
                 "rank_volume" actually computed)
  - product / product_clip: unchanged from report_ranking_selection.py /
                 report_ranking_refine.py (these did NOT have the
                 double-rank bug - product is a single rank step)

Same train/test discipline as before: for each side/threshold, pick the
single best (formula, top_n) combo by TRAIN sum_ret_pct (n_trades >= 10),
report its TEST result.

Writes output/ranking_refine2_corrected.md.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "ranking_refine2_corrected.md"

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
        "# Ranking/Selection Score Refinement — CORRECTED",
        "",
        "**This corrects a bug in report_ranking_refine.py**: its "
        "`rank_drift`/`rank_volume` formulas computed a rank-of-the-rank, "
        "which inverted the selection (top_n actually picked the "
        "BOTTOM-N candidates). This script computes both directions "
        "explicitly and unambiguously - see formula descriptions below.",
        "",
        f"- Full sample: {df['trade_date'].min().date()} -> {max_date.date()}",
        f"- TRAIN window: trade_date < {cutoff.date()}",
        f"- TEST window (held out): trade_date >= {cutoff.date()}",
        "",
        "Formulas (ranked within (trade_date, gap_threshold_pct) groups, per side):",
        "",
        "- **product**: `signed_drift_pct x max(volume_ratio, 0)`, highest first (unchanged, no bug)",
        "- **drift_high**: signed_drift_pct, HIGHEST first (drift confirms direction)",
        "- **drift_low**: signed_drift_pct, LOWEST first (drift opposes direction - this is "
        "what report_ranking_refine.py's buggy 'rank_drift' actually selected)",
        "- **volume_high**: volume_ratio, HIGHEST first",
        "- **volume_low**: volume_ratio, LOWEST first (what the buggy 'rank_volume' actually selected)",
        "",
    ]

    for side in ["LONG", "SHORT"]:
        rule = FIXED_RULE[side]
        sub = df[(df["side"] == side) & (df["rule_name"] == rule)].copy()
        group_cols = ["trade_date", "gap_threshold_pct"]

        rank_fns = {
            "product": sub["signed_drift"] * sub["volume_ratio_pos"],
            "drift_high": -sub["signed_drift"],
            "drift_low": sub["signed_drift"],
            "volume_high": -sub["volume_ratio"],
            "volume_low": sub["volume_ratio"],
        }
        # for product, higher score = better -> rank ascending=False
        # for drift_high/drift_low/volume_high/volume_low, we negate so that
        # "ascending=True rank" on the negated value == descending rank on the
        # original value -> rank 1 = the desired "first" element in each case.

        records = []
        for fname, raw in rank_fns.items():
            sub["_val"] = raw
            if fname == "product":
                sub["_rank"] = sub.groupby(group_cols)["_val"].rank(method="first", ascending=False)
            else:
                sub["_rank"] = sub.groupby(group_cols)["_val"].rank(method="first", ascending=True)

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
            ctx = rec_df[(rec_df["gap_threshold_pct"] == thr) & (rec_df["top_n"] == best["top_n"])]
            ctx = ctx[["formula", "train_sum_ret", "train_n", "test_sum_ret", "test_n", "test_hit_rate"]]
            lines.append("All formulas at this top_n, TEST performance (context):")
            lines.append("")
            lines.append(ctx.to_markdown(index=False, floatfmt=".2f"))
            lines.append("")

        lines.append("### Full TRAIN/TEST grid")
        lines.append("")
        full = rec_df[rec_df["train_n"] >= MIN_TRAIN_N].sort_values(["gap_threshold_pct", "formula", "top_n"])
        lines.append(full.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_ranking_refine2] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
