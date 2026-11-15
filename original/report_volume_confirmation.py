"""
report_volume_confirmation.py — tests whether the trading volume in the
09:30-09:45 entry window (before LONG entry) predicts trade quality.
Hypothesis: blowup names (SKYQ etc.) might trade on abnormally THIN volume
relative to their own history (no real buying interest behind the gap),
while genuine squeezes show abnormally HIGH relative volume (real demand).
Tests both directions.

Computes, for each selected LONG trade: dollar volume (price*volume) summed
over the 09:30-09:45 bars on the entry day, vs. that ticker's own trailing
20-session average dollar volume in the same 09:30-09:45 window (shifted,
no lookahead). Ratio = today's window volume / 20-day avg window volume.

Train/test disciplined: ratio threshold/direction chosen on TRAIN, evaluated
ONCE on TEST. 2.5%/3.0% gap thresholds reported separately.

Output: output/volume_confirmation.md
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
TRADES_SEL = ROOT / "data" / "trades_selected.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT_MD = ROOT / "output" / "volume_confirmation.md"

ET = ZoneInfo("America/New_York")
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]


def load_bars(ticker: str) -> pd.DataFrame:
    p = CACHE_DIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert(ET)
    df["session"] = df["ts_et"].dt.date
    df["minute_et"] = df["ts_et"].dt.strftime("%H:%M")
    return df


def window_dollar_vol(bars: pd.DataFrame, session) -> float | None:
    sess_bars = bars[(bars["session"] == session) &
                      (bars["minute_et"] >= "09:30") & (bars["minute_et"] < "09:45")]
    if sess_bars.empty:
        return None
    return float((sess_bars["c"] * sess_bars["v"]).sum())


def main() -> int:
    df = pd.read_parquet(TRADES_SEL)
    df = df.dropna(subset=["realized_return_pct"])
    sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    bars_cache: dict[str, pd.DataFrame] = {}
    daily_dv_cache: dict[str, pd.DataFrame] = {}  # ticker -> per-session window dollar vol series

    ratios = []
    for _, row in sel.iterrows():
        ticker = row["ticker"]
        if ticker not in bars_cache:
            bars_cache[ticker] = load_bars(ticker)
        bars = bars_cache[ticker]
        if bars.empty:
            ratios.append(None)
            continue

        if ticker not in daily_dv_cache:
            sessions = sorted(bars["session"].unique())
            dv = pd.Series({s: window_dollar_vol(bars, s) for s in sessions}).dropna()
            avg20 = dv.shift(1).rolling(20, min_periods=10).mean()
            daily_dv_cache[ticker] = avg20

        sess = pd.Timestamp(row["trade_date"]).date()
        today_dv = window_dollar_vol(bars, sess)
        avg20_series = daily_dv_cache[ticker]
        avg20_val = avg20_series.get(sess)
        if today_dv is None or avg20_val is None or pd.isna(avg20_val) or avg20_val <= 0:
            ratios.append(None)
        else:
            ratios.append(today_dv / avg20_val)

    sel["vol_ratio"] = ratios
    sel = sel.dropna(subset=["vol_ratio"])

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

    lines = [
        "# Entry-Bar Volume Confirmation (LONG)",
        "",
        "Tests whether relative volume in the 09:30-09:45 window (vs the "
        "ticker's own trailing 20-session average in that window, no "
        "lookahead) predicts trade quality. Tests both directions: thin "
        "volume = blowup risk (filter OUT low ratio) and high volume = "
        "real demand (filter IN high ratio).",
        "",
        f"({len(df[(df['selected']) & (df['side']=='LONG')]) - len(sel)} of "
        f"{len(df[(df['selected']) & (df['side']=='LONG')])} selected LONG trades "
        "dropped for missing volume history, e.g. <20 sessions of prior data.)",
        "",
        "Train/test disciplined: threshold chosen on TRAIN, evaluated ONCE "
        "on TEST. 2.5%/3.0% reported separately.",
        "",
    ]

    floor_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # keep if ratio >= floor (high-volume confirm)
    ceil_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]     # keep if ratio <= ceiling (avoid extreme spikes/thin-vol opposite)

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
        lines.append("### Direction A: keep only HIGH relative volume (ratio >= floor)")
        lines.append("")
        rows = []
        floor_results = []
        for f in floor_grid:
            tr = train[train["vol_ratio"] >= f]
            te = test[test["vol_ratio"] >= f]
            tr_p, te_p = perf(tr), perf(te)
            floor_results.append((f, tr_p, te_p))
            rows.append((f, *tr_p, *te_p))
        lines.append(tbl(["Floor", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        best_f = max(floor_results, key=lambda x: x[1][2])
        verdict_a = "PASS" if best_f[2][2] > base_te[2] else "FAIL"
        lines += ["", f"TRAIN-best floor={best_f[0]}: TEST {best_f[2][2]:.2f}% vs baseline "
                  f"{base_te[2]:.2f}% -> **{verdict_a}**", ""]

        lines.append("### Direction B: keep only LOW/normal relative volume (ratio <= ceiling)")
        lines.append("")
        rows = []
        ceil_results = []
        for c in ceil_grid:
            tr = train[train["vol_ratio"] <= c]
            te = test[test["vol_ratio"] <= c]
            tr_p, te_p = perf(tr), perf(te)
            ceil_results.append((c, tr_p, te_p))
            rows.append((c, *tr_p, *te_p))
        lines.append(tbl(["Ceiling", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                           "TEST n", "TEST hit%", "TEST sum%"], rows))
        best_c = max(ceil_results, key=lambda x: x[1][2])
        verdict_b = "PASS" if best_c[2][2] > base_te[2] else "FAIL"
        lines += ["", f"TRAIN-best ceiling={best_c[0]}: TEST {best_c[2][2]:.2f}% vs baseline "
                  f"{base_te[2]:.2f}% -> **{verdict_b}**", "", "---", ""]

        print(f"\n=== {thr}% ===")
        print(f"baseline: TRAIN {base_tr[2]:.2f}% -> TEST {base_te[2]:.2f}%")
        print(f"Direction A (high-vol floor={best_f[0]}): TRAIN {best_f[1][2]:.2f}% -> TEST {best_f[2][2]:.2f}% ({verdict_a})")
        print(f"Direction B (low-vol ceiling={best_c[0]}): TRAIN {best_c[1][2]:.2f}% -> TEST {best_c[2][2]:.2f}% ({verdict_b})")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
