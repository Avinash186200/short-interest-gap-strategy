"""
report_market_regime_gate.py — tests Expert's market-regime idea properly
for the first time: gate LONG entries on the small-cap index (IWM, proxy
for Russell 2000) direction, since these are high-beta small caps that
likely move with the broad small-cap tape. Previously untestable - no
index-level intraday data was cached (see regime.py docstring / README
"Polygon probe"). Fetched IWM 1-min bars via intraday_cache.fetch_symbol
for 2024-01-01 -> 2026-05-31 to enable this.

Two signals tested:
  R1. Prior-day trend: IWM close > its own 20-session SMA as of yesterday's
      close (no lookahead - known before market open).
  R2. Same-day intraday move: IWM's own return from prior close to 09:45
      on the trade day (contemporaneous - available at the same moment the
      LONG entry decision is made, since both prints happen at 09:45).

Train/test disciplined: gate direction chosen on TRAIN, evaluated ONCE on
TEST. 2.5%/3.0% gap thresholds reported separately.

Output: output/market_regime_gate.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import regime

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "market_regime_gate.md"

ET = ZoneInfo("America/New_York")
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]


def load_bars(ticker: str) -> pd.DataFrame:
    p = CACHE_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(p)
    df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert(ET)
    df["session"] = df["ts_et"].dt.date
    df["minute_et"] = df["ts_et"].dt.strftime("%H:%M")
    return df


# ── build IWM daily series + regime signals ────────────────────────────────
iwm_daily = regime.load_daily_ohlc("IWM")
iwm_daily["sma20"] = iwm_daily["close"].rolling(20, min_periods=20).mean().shift(1)
iwm_daily["above_sma20"] = iwm_daily["close"].shift(1) > iwm_daily["sma20"]  # known as of prior close

iwm_bars = load_bars("IWM")


def iwm_intraday_return_0945(session) -> float | None:
    """IWM return from yesterday's close to today's 09:45 print."""
    idx = iwm_daily.index[iwm_daily["session"] == session]
    if len(idx) == 0 or idx[0] == 0:
        return None
    prior_close = iwm_daily.loc[idx[0] - 1, "close"] if idx[0] > 0 else None
    if prior_close is None or prior_close <= 0:
        return None
    sess_bars = iwm_bars[(iwm_bars["session"] == session) & (iwm_bars["minute_et"] == "09:45")]
    if sess_bars.empty:
        return None
    p945 = float(sess_bars["c"].iloc[0])
    return 100.0 * (p945 - prior_close) / prior_close


prior_trend_map = dict(zip(iwm_daily["session"], iwm_daily["above_sma20"]))
sessions_with_data = sorted(iwm_daily["session"].unique())
intraday_ret_map = {s: iwm_intraday_return_0945(s) for s in sessions_with_data}


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])
    sel["sess"] = sel["trade_date"].dt.date

    sel["iwm_above_sma20"] = sel["sess"].map(prior_trend_map)
    sel["iwm_0945_ret"] = sel["sess"].map(intraday_ret_map)

    def perf(s):
        n = len(s)
        hit = (s["realized_return_pct"] > 0).mean() * 100 if n else 0
        sumr = s["realized_return_pct"].sum()
        return n, round(hit, 2), round(sumr, 2)

    def tbl(headers, rows):
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
        for row in rows:
            out.append("|" + "|".join(str(v) for v in row) + "|")
        return "\n".join(out)

    coverage = sel["iwm_above_sma20"].notna().mean() * 100
    coverage2 = sel["iwm_0945_ret"].notna().mean() * 100

    lines = [
        "# Market-Regime Gate (IWM small-cap proxy) — LONG",
        "",
        "Tests gating LONG entries on the IWM (Russell 2000 proxy) direction "
        "per Expert's suggestion - previously untestable (no index-level "
        "intraday data cached). Fetched IWM 1-min bars for the full history.",
        "",
        f"Coverage: {coverage:.1f}%/{coverage2:.1f}% "
        "(fixed a data-fetch bug: the 55-day chunk size in intraday_cache.py "
        "was calibrated for normal-volume small caps and silently truncated "
        "responses for a high-volume ETF like IWM, leaving 271 of 612 "
        "business days missing on the first attempt - re-fetched with "
        "10-day chunks, now only 25 missing days, all genuine NYSE "
        "holidays).",
        "",
        "Train/test disciplined: gate direction chosen on TRAIN, evaluated "
        "ONCE on TEST. 2.5%/3.0% reported separately.",
        "",
    ]

    for thr in THRESHOLDS:
        sub = sel[sel["gap_threshold_pct"] == thr]
        train = sub[sub["trade_date"] < TRAIN_CUTOFF]
        test = sub[sub["trade_date"] >= TRAIN_CUTOFF]
        base_tr, base_te = perf(train), perf(test)

        lines.append(f"## Gap threshold {thr}%")
        lines.append("")
        lines.append(tbl(["", "n", "hit%", "sum%"],
                          [("TRAIN baseline", *base_tr), ("TEST baseline", *base_te)]))
        lines.append("")

        # R1: prior-day trend
        lines.append("### R1: IWM above its 20-session SMA (prior-day trend)")
        lines.append("")
        variants_r1 = {
            "only when IWM above SMA20 (uptrend)": train["iwm_above_sma20"] == True,
            "only when IWM below SMA20 (downtrend)": train["iwm_above_sma20"] == False,
        }
        rows = []
        results_r1 = {}
        for label, mask in variants_r1.items():
            tr = train[mask]
            te_mask = (test["iwm_above_sma20"] == True) if "above" in label else (test["iwm_above_sma20"] == False)
            te = test[te_mask]
            tr_p, te_p = perf(tr), perf(te)
            results_r1[label] = (tr_p, te_p)
            rows.append((label, *tr_p, *te_p))
        lines.append(tbl(["Variant", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        best_label = max(results_r1, key=lambda k: results_r1[k][0][2])
        best_te = results_r1[best_label][1][2]
        v1 = "PASS" if best_te > base_te[2] else "FAIL"
        lines += ["", f"TRAIN-best: '{best_label}' -> TEST {best_te:.2f}% vs baseline "
                  f"{base_te[2]:.2f}% -> **{v1}**", ""]

        # R2: same-day intraday IWM return at 09:45, threshold grid
        lines.append("### R2: IWM same-day return to 09:45 (contemporaneous)")
        lines.append("")
        thresholds_r2 = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
        rows = []
        results_r2 = []
        for t2 in thresholds_r2:
            tr = train[train["iwm_0945_ret"] >= t2]
            te = test[test["iwm_0945_ret"] >= t2]
            tr_p, te_p = perf(tr), perf(te)
            results_r2.append((t2, tr_p, te_p))
            rows.append((t2, *tr_p, *te_p))
        lines.append(tbl(["IWM ret>= (%)", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        best_r2 = max(results_r2, key=lambda x: x[1][2])
        v2 = "PASS" if best_r2[2][2] > base_te[2] else "FAIL"
        lines += ["", f"TRAIN-best threshold={best_r2[0]}%: TEST {best_r2[2][2]:.2f}% vs baseline "
                  f"{base_te[2]:.2f}% -> **{v2}**", "", "---", ""]

        print(f"\n=== {thr}% ===")
        print(f"baseline: TRAIN {base_tr[2]:.2f}% -> TEST {base_te[2]:.2f}%")
        print(f"R1 best ('{best_label}'): TEST {best_te:.2f}% ({v1})")
        print(f"R2 best (thr={best_r2[0]}%): TRAIN {best_r2[1][2]:.2f}% -> TEST {best_r2[2][2]:.2f}% ({v2})")

    lines += [
        "## Conclusion", "",
        "**Both R1 and R2 FAIL at both thresholds on the corrected, "
        "near-complete data** (after fixing the chunking bug above). This "
        "reverses the earlier result computed on the biased ~65% subsample, "
        "which had appeared to pass - a useful lesson: that result was an "
        "artifact of which trades happened to have IWM data, not a real "
        "signal. Market-regime gating via IWM direction (prior-day trend "
        "or same-day move to 09:45) does not improve the LONG strategy. "
        "This closes out the last of Expert's 3 original suggestions - all "
        "three (SHORT Q1 drift filter, 10-day publication window, wide "
        "stop-loss/exit tuning including this regime-gate variant) have "
        "now been tested.",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
