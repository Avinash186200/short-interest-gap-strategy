"""
intraday_cache.py — pull Finnhub 1-min bars for the screened universe.

Speed optimizations:
  - Finnhub paid tier: 150 req/min (probed)
  - Max ~15,666 bars per call ≈ 60 calendar days for a normal-volume stock
  - Per ticker, we determine the FULL date range covering all windows,
    then chunk it in 55-day pieces (buffer below the 60d cap).
  - Threaded with a token-bucket rate limiter (~140 req/min, leaves slack).

Cache layout:
  cache/intraday/{TICKER}.parquet  -> columns: t, o, h, l, c, v
"""
from __future__ import annotations
import time, urllib3, datetime as dt, threading, queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

urllib3.disable_warnings()

ROOT = Path(__file__).parent
SCR = ROOT / "data" / "screened_universe.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
KEY = "<REDACTED>"
API = "https://finnhub.io/api/v1/stock/candle"

WIN_BEFORE_DAYS = 3
WIN_AFTER_DAYS = 28   # cover settlement + 8bd publication lag + 10 trading-day hold + buffer
CHUNK_DAYS = 55
RATE_PER_MIN = 140
WORKERS = 12


class TokenBucket:
    """Sliding-window rate limiter."""
    def __init__(self, max_per_min: int):
        self.max = max_per_min
        self.events: list[float] = []
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.events = [t for t in self.events if t > now - 60]
                if len(self.events) < self.max:
                    self.events.append(now)
                    return
                wait = 60 - (now - self.events[0]) + 0.05
            time.sleep(max(0.01, wait))


BUCKET = TokenBucket(RATE_PER_MIN)


def _candle_request(symbol: str, start_ts: int, end_ts: int, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        BUCKET.acquire()
        try:
            r = requests.get(
                API,
                params={"symbol": symbol, "resolution": "1", "from": start_ts, "to": end_ts, "token": KEY},
                timeout=30, verify=False,
            )
        except Exception:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 429:
            time.sleep(2 + attempt * 2)
            continue
        if r.status_code != 200:
            return None
        try:
            j = r.json()
        except Exception:
            return None
        if j.get("s") not in ("ok", "no_data"):
            return None
        return j
    return None


def merge_ranges(ranges: list[tuple[dt.date, dt.date]]) -> list[tuple[dt.date, dt.date]]:
    """Merge overlapping/adjacent date ranges."""
    if not ranges:
        return []
    rs = sorted(ranges)
    out = [rs[0]]
    for s, e in rs[1:]:
        if s <= out[-1][1] + dt.timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def chunks_55d(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=CHUNK_DAYS - 1), end)
        yield cur, nxt
        cur = nxt + dt.timedelta(days=1)


def load_cached(symbol: str) -> pd.DataFrame:
    p = CACHE_DIR / f"{symbol}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])


def save_cached(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_DIR / f"{symbol}.parquet", index=False)


def covered_days(df: pd.DataFrame) -> set[dt.date]:
    if df.empty:
        return set()
    return set(pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.date.unique())


def fetch_symbol(symbol: str, ranges: list[tuple[dt.date, dt.date]]) -> tuple[str, int, int, float]:
    """Returns (symbol, new_bars, total_bars, elapsed_s)."""
    t0 = time.time()
    cached = load_cached(symbol)
    have_days = covered_days(cached)

    # for each merged range, sub-chunk into 55-day calls, skip if every day already cached
    merged = merge_ranges(ranges)
    new_rows: list[dict] = []
    for s, e in merged:
        for cs, ce in chunks_55d(s, e):
            need_days = {cs + dt.timedelta(days=i) for i in range((ce - cs).days + 1)}
            # if every weekday in this chunk is already cached, skip
            weekdays = {d for d in need_days if d.weekday() < 5}
            if weekdays and weekdays.issubset(have_days):
                continue
            start_ts = int(dt.datetime(cs.year, cs.month, cs.day, 0, 0).timestamp())
            end_ts = int(dt.datetime(ce.year, ce.month, ce.day, 23, 59).timestamp())
            j = _candle_request(symbol, start_ts, end_ts)
            if not j or j.get("s") == "no_data":
                continue
            for t, o, h, l, c, v in zip(j["t"], j["o"], j["h"], j["l"], j["c"], j["v"]):
                new_rows.append({"t": int(t), "o": float(o), "h": float(h),
                                 "l": float(l), "c": float(c), "v": float(v)})

    if new_rows:
        fresh = pd.DataFrame(new_rows)
        combined = pd.concat([cached, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        save_cached(symbol, combined)
        return symbol, len(fresh), len(combined), time.time() - t0
    return symbol, 0, len(cached), time.time() - t0


def main(*, limit: int | None = None) -> int:
    if not SCR.exists():
        print(f"[intraday] missing {SCR} -- run screen.py first")
        return 1
    scr = pd.read_parquet(SCR)
    scr["settlementDate"] = pd.to_datetime(scr["settlementDate"]).dt.date
    print(f"[intraday] universe rows: {len(scr):,}, unique tickers: {scr['symbolCode'].nunique()}", flush=True)

    per: dict[str, list[tuple[dt.date, dt.date]]] = {}
    for tkr, g in scr.groupby("symbolCode"):
        rngs = []
        for d in g["settlementDate"].unique():
            s = d - dt.timedelta(days=WIN_BEFORE_DAYS + 4)
            e = d + dt.timedelta(days=WIN_AFTER_DAYS + 4)
            rngs.append((s, e))
        per[tkr] = rngs

    syms = sorted(per.keys())
    if limit:
        syms = syms[:limit]
    print(f"[intraday] will fetch {len(syms)} tickers with {WORKERS} workers @ {RATE_PER_MIN} req/min", flush=True)

    t_start = time.time()
    done = 0
    new_bars_total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_symbol, s, per[s]): s for s in syms}
        for fut in as_completed(futs):
            sym, new_n, total_n, dt_s = fut.result()
            done += 1
            new_bars_total += new_n
            if done % 10 == 0 or done == len(syms) or new_n > 0:
                eta = (len(syms) - done) * (time.time() - t_start) / max(done, 1)
                print(f"  [{done:>4}/{len(syms)}] {sym:<8} +{new_n:>6} (cum_new={new_bars_total:,}) eta={eta/60:.1f}m", flush=True)
    print(f"[intraday] DONE in {(time.time()-t_start)/60:.1f}m, +{new_bars_total:,} bars total", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(limit=lim))
