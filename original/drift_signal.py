"""
drift_signal.py — Item 1 of NEXT_ITERATION_PLAN.md.

For each (settlementDate, ticker) in the screened universe, compute the price
(and volume) drift between the FINRA settlement date and the date the short
-interest data actually becomes visible to us (publication_date, ~settlement
+ 8 business days, same definition as backtest.py).

  settlement_close -> close on/before settlementDate
  pub_eve_close    -> close on the session immediately before the first
                       trade_date (= prev_close used for that trade's gap_pct
                       in backtest.py), so this is point-in-time, no lookahead.

  price_drift_pct  = 100 * (pub_eve_close - settlement_close) / settlement_close
  avg_vol_window   = mean daily RTH volume over [settlementDate, pub_eve]
  volume_ratio     = avg_vol_window / averageDailyVolumeQuantity (FINRA ADV)

Interpretation (per Expert):
  - price_drift_pct > 0  -> price kept rising since settlement -> supports LONG/squeeze
  - price_drift_pct < 0  -> price kept falling since settlement -> supports SHORT (more shorting)
  - volume_ratio low     -> "thin volume" move, weaker confirmation

Writes data/drift_signal.parquet, one row per (settlementDate, symbolCode).
"""
from __future__ import annotations
from pathlib import Path
import datetime as dt

import pandas as pd

ROOT = Path(__file__).parent
SCR = ROOT / "data" / "screened_universe.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT = ROOT / "data" / "drift_signal.parquet"

PUBLICATION_LAG_BUSINESS_DAYS = 8


def publication_date(settlement: dt.date) -> dt.date:
    d = settlement
    added = 0
    while added < PUBLICATION_LAG_BUSINESS_DAYS:
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def load_daily(ticker: str) -> pd.DataFrame:
    """One row per session: date, close (last RTH bar), volume (sum of RTH bars)."""
    p = CACHE_DIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    ts_et = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(session=ts_et.dt.date, minute_et=ts_et.dt.strftime("%H:%M"))
    rth = df[(df["minute_et"] >= "09:30") & (df["minute_et"] <= "16:00")]
    if rth.empty:
        return pd.DataFrame()
    g = rth.groupby("session")
    out = pd.DataFrame({
        "close": g["c"].last(),
        "volume": g["v"].sum(),
    }).reset_index().sort_values("session").reset_index(drop=True)
    return out


def compute_one(daily: pd.DataFrame, settlement: dt.date, adv: float) -> dict | None:
    if daily.empty:
        return None

    on_or_before = daily[daily["session"] <= settlement]
    if on_or_before.empty:
        return None
    settlement_close = float(on_or_before.iloc[-1]["close"])
    settlement_sess = on_or_before.iloc[-1]["session"]

    pub = publication_date(settlement)
    after_pub = daily[daily["session"] >= pub]
    if after_pub.empty:
        return None
    first_trade_sess = after_pub.iloc[0]["session"]

    before_first_trade = daily[daily["session"] < first_trade_sess]
    if before_first_trade.empty:
        return None
    pub_eve_close = float(before_first_trade.iloc[-1]["close"])
    pub_eve_sess = before_first_trade.iloc[-1]["session"]

    if settlement_close <= 0:
        return None

    window = daily[(daily["session"] >= settlement_sess) & (daily["session"] <= pub_eve_sess)]
    avg_vol = float(window["volume"].mean()) if not window.empty else float("nan")
    volume_ratio = avg_vol / adv if adv and adv > 0 else float("nan")

    price_drift_pct = 100.0 * (pub_eve_close - settlement_close) / settlement_close

    return {
        "settlement_close": settlement_close,
        "settlement_close_date": settlement_sess.isoformat(),
        "pub_eve_close": pub_eve_close,
        "pub_eve_date": pub_eve_sess.isoformat(),
        "first_trade_date": first_trade_sess.isoformat(),
        "price_drift_pct": price_drift_pct,
        "avg_vol_window": avg_vol,
        "volume_ratio": volume_ratio,
        "drift_supports_long": price_drift_pct > 0,
        "drift_supports_short": price_drift_pct < 0,
    }


def main() -> int:
    if not SCR.exists():
        print(f"[drift] missing {SCR}")
        return 1
    scr = pd.read_parquet(SCR)
    scr["settlementDate"] = pd.to_datetime(scr["settlementDate"]).dt.date
    print(f"[drift] universe rows: {len(scr):,}", flush=True)

    daily_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    n_missing = 0

    for i, row in enumerate(scr.itertuples(index=False), 1):
        tkr = row.symbolCode
        d = row.settlementDate
        adv = float(getattr(row, "averageDailyVolumeQuantity", float("nan")))
        if tkr not in daily_cache:
            daily_cache[tkr] = load_daily(tkr)
        res = compute_one(daily_cache[tkr], d, adv)
        if res is None:
            n_missing += 1
            continue
        res["settlementDate"] = d.isoformat()
        res["symbolCode"] = tkr
        rows.append(res)
        if i % 1000 == 0:
            print(f"  [{i}/{len(scr)}] computed={len(rows)} missing={n_missing}", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        print("[drift] zero rows computed")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[drift] wrote {len(out):,} rows ({n_missing} missing/skipped) -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
