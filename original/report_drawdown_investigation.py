"""
report_drawdown_investigation.py — digs into WHY the drift_low top-5 LONG
equity curve drew down -323.77% between its 2025-10-13 peak (+446.64%) and
2026-03-31 trough (~+122.9%) (see output/rank_drift_strategy.md,
output/results_summary_selected.md).

Looks at the selected (drift_low_top5) trades with trade_date in
[2025-10-14, 2026-03-31] and asks:
  - Is the loss concentrated in a handful of tickers/days (a few bad
    episodes), or broad-based (a regime where the edge itself flipped)?
  - Does hit rate / avg return inside the drawdown window differ materially
    from the full-sample numbers?
  - Per-ticker and per-day contribution to the drawdown-window loss.

Writes output/drawdown_investigation.md.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
OUT_MD = ROOT / "output" / "drawdown_investigation.md"

DD_START = "2025-10-14"
DD_END = "2026-03-31"


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_trades": len(g),
        "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
        "avg_ret_pct": g["realized_return_pct"].mean(),
        "sum_ret_pct": g["realized_return_pct"].sum(),
    })


def main() -> int:
    if not TRADES_SEL.exists():
        print(f"[drawdown] missing {TRADES_SEL} - run select_signals.py first")
        return 1

    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    sel = df[df["selected"]].copy()

    dd = sel[(sel["trade_date"] >= DD_START) & (sel["trade_date"] <= DD_END)].copy()
    rest = sel[(sel["trade_date"] < DD_START) | (sel["trade_date"] > DD_END)].copy()

    lines = [
        "# Drawdown investigation: drift_low top-5, 2025-10-14 -> 2026-03-31",
        "",
        f"Window covers the drawdown from the equity-curve peak (+446.64% on "
        f"2025-10-13) to the trough (~+122.9% on 2026-03-31), per "
        f"output/results_summary_selected.md.",
        "",
        "## Overall: drawdown window vs rest of sample",
        "",
    ]

    overall = pd.DataFrame([
        {"window": "drawdown (2025-10-14 .. 2026-03-31)", **agg(dd).to_dict()},
        {"window": "rest of sample", **agg(rest).to_dict()},
        {"window": "full sample", **agg(sel).to_dict()},
    ])
    lines.append(overall.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # By gap threshold within drawdown window
    lines.append("## Drawdown window, by gap_threshold_pct")
    lines.append("")
    by_thresh = dd.groupby("gap_threshold_pct").apply(agg, include_groups=False).reset_index()
    lines.append(by_thresh.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # Monthly breakdown within drawdown window
    lines.append("## Drawdown window, by month")
    lines.append("")
    dd["month"] = dd["trade_date"].dt.to_period("M").astype(str)
    by_month = dd.groupby("month").apply(agg, include_groups=False).reset_index()
    lines.append(by_month.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # Per-ticker contribution within drawdown window
    lines.append("## Drawdown window: per-ticker contribution to sum_ret")
    lines.append("")
    by_ticker = dd.groupby("ticker").apply(agg, include_groups=False).reset_index()
    by_ticker = by_ticker.sort_values("sum_ret_pct")
    n_tickers = len(by_ticker)
    total_loss = dd["realized_return_pct"].sum()
    worst10 = by_ticker.head(10)
    worst10_sum = worst10["sum_ret_pct"].sum()
    lines.append(f"{n_tickers} distinct tickers traded in the drawdown window. "
                  f"Total sum_ret in window: {total_loss:.2f}%. "
                  f"Worst 10 tickers account for {worst10_sum:.2f}% "
                  f"({100 * worst10_sum / total_loss:.1f}% of the total).")
    lines.append("")
    lines.append("### Worst 10 tickers (by sum_ret_pct)")
    lines.append("")
    lines.append(worst10.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("### Best 10 tickers (by sum_ret_pct)")
    lines.append("")
    lines.append(by_ticker.tail(10).sort_values("sum_ret_pct", ascending=False).to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # Worst single trades
    lines.append("## Worst 15 single trades in the drawdown window")
    lines.append("")
    cols = ["trade_date", "ticker", "gap_threshold_pct", "gap_pct", "price_drift_pct",
            "volume_ratio", "entry_price", "exit_price", "exit_rule", "realized_return_pct"]
    worst_trades = dd.sort_values("realized_return_pct").head(15)[cols]
    lines.append(worst_trades.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # Distribution of price_drift_pct in drawdown window vs rest (is the
    # selection picking more extreme / different candidates in this window?)
    lines.append("## price_drift_pct of selected trades: drawdown window vs rest")
    lines.append("")
    drift_stats = pd.DataFrame([
        {"window": "drawdown", **dd["price_drift_pct"].describe().to_dict()},
        {"window": "rest", **rest["price_drift_pct"].describe().to_dict()},
    ])
    lines.append(drift_stats.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[drawdown] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
