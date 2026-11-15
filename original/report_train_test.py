"""
report_train_test.py — train/test split re-validation.

Per the user's concern (2026-06-12): any re-tuning of which exit
rule/gap-threshold to use must be selected on a TRAIN window only, then
evaluated ONCE on a held-out TEST window, to avoid overfitting to the recent
data we keep checking.

Split:
  TRAIN: trade_date <  cutoff  (everything except the last ~4 months)
  TEST:  trade_date >= cutoff  (held-out, last ~4 months)

For each side (LONG / SHORT), candidates are the (rule_name,
gap_threshold_pct) combinations available in trades.parquet:
  LONG:  (hold_above_open_945_tp3, 2.5/3.0), (trailing_atr, 2.5/3.0)
  SHORT: (failed_bounce_1000_tp_2.5, 2.5/3.0), (trailing_atr, 2.5/3.0)

Selection: pick the candidate with the best TRAIN sum_ret_pct. Report that
candidate's TEST performance, alongside TEST performance of all candidates
for context (to show whether the selection was robust or "lucky").

Writes output/train_test_selection.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
OUT_MD = ROOT / "output" / "train_test_selection.md"

TEST_MONTHS = 4

FIXED_RULE = {"LONG": "hold_above_open_945_tp3", "SHORT": "failed_bounce_1000_tp_2.5"}


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    if not TRADES.exists():
        print(f"[report_train_test] missing {TRADES}")
        return 1

    df = pd.read_parquet(TRADES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=["realized_return_pct"])

    max_date = df["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=TEST_MONTHS)
    train = df[df["trade_date"] < cutoff]
    test = df[df["trade_date"] >= cutoff]

    lines = [
        "# Train/Test Split Re-validation",
        "",
        f"- Full sample: {df['trade_date'].min().date()} -> {max_date.date()}",
        f"- TRAIN window: trade_date < {cutoff.date()} ({len(train):,} rows)",
        f"- TEST window (held out): trade_date >= {cutoff.date()} ({len(test):,} rows)",
        "",
        "Selection rule: for each side, pick the (rule_name, gap_threshold_pct) "
        "combo with the best TRAIN sum_ret_pct. Report its TEST performance, "
        "plus TEST performance of all candidates for context.",
        "",
    ]

    for side in ["LONG", "SHORT"]:
        candidates = [
            (FIXED_RULE[side], 2.5), (FIXED_RULE[side], 3.0),
            ("trailing_atr", 2.5), ("trailing_atr", 3.0),
        ]
        lines.append(f"## {side}")
        lines.append("")

        train_rows = []
        test_rows = []
        for rule, thr in candidates:
            tr = train[(train["side"] == side) & (train["rule_name"] == rule) & (train["gap_threshold_pct"] == thr)]
            te = test[(test["side"] == side) & (test["rule_name"] == rule) & (test["gap_threshold_pct"] == thr)]
            tr_m = agg(tr) if len(tr) else pd.Series({"n_trades": 0, "hit_rate_pct": float("nan"), "avg_ret_pct": float("nan"), "sum_ret_pct": 0.0})
            te_m = agg(te) if len(te) else pd.Series({"n_trades": 0, "hit_rate_pct": float("nan"), "avg_ret_pct": float("nan"), "sum_ret_pct": 0.0})
            train_rows.append({"rule_name": rule, "gap_threshold_pct": thr, **tr_m.to_dict()})
            test_rows.append({"rule_name": rule, "gap_threshold_pct": thr, **te_m.to_dict()})

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)

        best_idx = train_df["sum_ret_pct"].idxmax()
        best_rule, best_thr = train_df.loc[best_idx, "rule_name"], train_df.loc[best_idx, "gap_threshold_pct"]

        lines.append("### TRAIN (selection basis)")
        lines.append("")
        lines.append(train_df.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")
        lines.append(f"**Selected on TRAIN:** `{best_rule}` @ {best_thr:.1f}% "
                      f"(train sum_ret_pct = {train_df.loc[best_idx, 'sum_ret_pct']:.2f})")
        lines.append("")
        lines.append("### TEST (held-out, last 4 months) — all candidates for context")
        lines.append("")
        lines.append(test_df.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

        sel_test = test_df[(test_df["rule_name"] == best_rule) & (test_df["gap_threshold_pct"] == best_thr)].iloc[0]
        lines.append(f"**Selected combo's TEST result:** `{best_rule}` @ {best_thr:.1f}% -> "
                      f"n={int(sel_test['n_trades'])}, hit_rate={sel_test['hit_rate_pct']:.2f}%, "
                      f"sum_ret={sel_test['sum_ret_pct']:.2f}%")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_train_test] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
