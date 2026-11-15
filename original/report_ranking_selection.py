"""
report_ranking_selection.py — test Expert's actual mechanism: rank same-day
candidates by a composite signal (drift + volume) and take only the TOP-N
per day, instead of taking every trade that passes the gap/SI filter.

Composite score (per trade):
  signed_drift = price_drift_pct        if side == LONG
               = -price_drift_pct       if side == SHORT
  score = signed_drift * max(volume_ratio, 0)

Higher score = drift confirms the trade direction AND volume supports it
(Expert's two stated ranking signals). Within each
(trade_date, side, gap_threshold_pct) group, rank trades by score descending
and keep only the top N.

Only applied to the original fixed-TP rules (hold_above_open_945_tp3 /
failed_bounce_1000_tp_2.5) - this mirrors Expert's setup, which had no
trailing-ATR exit.

Per the train/test discipline established in report_train_test.py: pick N
based on TRAIN-window performance, then report the TEST-window (held-out,
last 4 months) result for that N, alongside TEST results for other N as
context.

Writes output/ranking_selection.md.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "ranking_selection.md"

FIXED_RULE = {"LONG": "hold_above_open_945_tp3", "SHORT": "failed_bounce_1000_tp_2.5"}
TOP_NS = [1, 2, 3, 5, 999]  # 999 = "all" (no selection)
TEST_MONTHS = 4


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    if not TRADES.exists() or not DRIFT.exists():
        print("[report_ranking_selection] missing trades.parquet or drift_signal.parquet")
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
    df["score"] = df["signed_drift"] * df["volume_ratio"].clip(lower=0)

    lines = [
        "# Ranking/Selection Test (top-N candidates per day by drift+volume score)",
        "",
        f"- Full sample: {df['trade_date'].min().date()} -> {max_date.date()}",
        f"- TRAIN window: trade_date < {cutoff.date()}",
        f"- TEST window (held out): trade_date >= {cutoff.date()}",
        "",
        "Score = signed_drift_pct (price_drift_pct for LONG, -price_drift_pct "
        "for SHORT) x max(volume_ratio, 0). Within each "
        "(trade_date, side, gap_threshold_pct) group, rank candidates by "
        "score descending and keep only the top N. N=999 = no selection "
        "(all candidates, current baseline).",
        "",
        "Only applied to the original fixed-TP rules (no trailing-ATR exit), "
        "matching Expert's setup.",
        "",
    ]

    for side in ["LONG", "SHORT"]:
        rule = FIXED_RULE[side]
        sub = df[(df["side"] == side) & (df["rule_name"] == rule)].copy()

        sub["rank"] = sub.groupby(["trade_date", "gap_threshold_pct"])["score"] \
            .rank(method="first", ascending=False)

        train = sub[sub["trade_date"] < cutoff]
        test = sub[sub["trade_date"] >= cutoff]

        lines.append(f"## {side} ({rule})")
        lines.append("")

        train_rows, test_rows = [], []
        for n in TOP_NS:
            label = "all" if n == 999 else str(n)
            for thr in [2.5, 3.0]:
                tr = train[(train["rank"] <= n) & (train["gap_threshold_pct"] == thr)]
                te = test[(test["rank"] <= n) & (test["gap_threshold_pct"] == thr)]
                tr_m = agg(tr) if len(tr) else pd.Series({"n_trades": 0, "hit_rate_pct": np.nan, "avg_ret_pct": np.nan, "sum_ret_pct": 0.0})
                te_m = agg(te) if len(te) else pd.Series({"n_trades": 0, "hit_rate_pct": np.nan, "avg_ret_pct": np.nan, "sum_ret_pct": 0.0})
                train_rows.append({"top_n": label, "gap_threshold_pct": thr, **tr_m.to_dict()})
                test_rows.append({"top_n": label, "gap_threshold_pct": thr, **te_m.to_dict()})

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)

        lines.append("### TRAIN (selection basis)")
        lines.append("")
        lines.append(train_df.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")
        lines.append("### TEST (held-out, last 4 months)")
        lines.append("")
        lines.append(test_df.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

        # Selection: best TRAIN sum_ret_pct (the actual P&L metric) among
        # non-trivial sample sizes (n_trades >= 10), per threshold
        for thr in [2.5, 3.0]:
            tdf = train_df[(train_df["gap_threshold_pct"] == thr) & (train_df["n_trades"] >= 10)]
            if tdf.empty:
                continue
            best = tdf.loc[tdf["sum_ret_pct"].idxmax()]
            best_n = best["top_n"]
            sel_test = test_df[(test_df["top_n"] == best_n) & (test_df["gap_threshold_pct"] == thr)]
            if not sel_test.empty:
                st = sel_test.iloc[0]
                lines.append(
                    f"- @{thr:.1f}%: best TRAIN sum_ret at top_n={best_n} "
                    f"(train sum_ret={best['sum_ret_pct']:.2f}%, n={int(best['n_trades'])}) "
                    f"-> TEST: n={int(st['n_trades'])}, hit_rate={st['hit_rate_pct']:.2f}%, "
                    f"sum_ret={st['sum_ret_pct']:.2f}%"
                )
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_ranking_selection] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
