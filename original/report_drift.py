"""
report_drift.py — Item 1 validation: does the settlement->publication price
drift signal (drift_signal.py) actually improve hit rate / returns, as
Expert claimed on the call?

For each trade:
  - LONG trades:  "confirmed" if price_drift_pct > 0 (price kept rising into
                   publication -> squeeze continuing)
  - SHORT trades: "confirmed" if price_drift_pct < 0 (price kept falling into
                   publication -> shorts piling in further)

Also splits "confirmed" by volume_ratio >= 1 (avg volume in the drift window
at/above the FINRA ADV) vs < 1 ("thin volume", per Expert's caveat).

Writes output/drift_signal_breakdown.md.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
TRADES = ROOT / "data" / "trades.parquet"
DRIFT = ROOT / "data" / "drift_signal.parquet"
OUT_MD = ROOT / "output" / "drift_signal_breakdown.md"


def main() -> int:
    if not TRADES.exists():
        print(f"[report_drift] missing {TRADES}")
        return 1
    if not DRIFT.exists():
        print(f"[report_drift] missing {DRIFT} -- run drift_signal.py first")
        return 1

    trades = pd.read_parquet(TRADES)
    drift = pd.read_parquet(DRIFT)

    df = trades.merge(
        drift,
        left_on=["settlement_date", "ticker"],
        right_on=["settlementDate", "symbolCode"],
        how="left",
    )
    n_unmatched = df["price_drift_pct"].isna().sum()
    df = df.dropna(subset=["price_drift_pct", "realized_return_pct"])
    print(f"[report_drift] {len(trades):,} trades -> {len(df):,} with drift signal "
          f"({n_unmatched:,} unmatched)", flush=True)

    df["confirmed"] = np.where(
        df["side"] == "LONG", df["drift_supports_long"], df["drift_supports_short"]
    )
    df["volume_confirms"] = df["volume_ratio"] >= 1.0

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_trades": len(g),
            "hit_rate_pct": (g["realized_return_pct"] > 0).mean() * 100,
            "avg_ret_pct": g["realized_return_pct"].mean(),
            "sum_ret_pct": g["realized_return_pct"].sum(),
        })

    by_confirm = df.groupby(["rule_name", "side", "gap_threshold_pct", "confirmed"]).apply(agg, include_groups=False).reset_index()

    by_confirm_vol = (
        df[df["confirmed"]]
        .groupby(["rule_name", "side", "gap_threshold_pct", "volume_confirms"])
        .apply(agg, include_groups=False)
        .reset_index()
    )

    lines = ["# Drift Signal Breakdown (settlement -> publication price drift)", "",
             f"- Trades with drift signal matched: {len(df):,} / {len(trades):,}",
             "",
             "## Confirmed vs not confirmed (drift direction matches trade side)",
             "",
             by_confirm.to_markdown(index=False, floatfmt=".2f"),
             "",
             "## Within 'confirmed' trades: volume_ratio >= 1 vs thin volume",
             "",
             by_confirm_vol.to_markdown(index=False, floatfmt=".2f"),
             ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_drift] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
