"""
screen.py — replicate the FINRA short-interest screen from STRATEGY.md.

For each settlement date in the FINRA dataset:
  - filter: DTC >= 5, SI >= 1M, ADV >= 100k, exclude OTC unless include_otc
  - enrich: shares outstanding (Finnhub /stock/profile2; cached)
  - filter: SI/SharesOut >= 10%   (proxy for short_interest_pct_float)
  - rank:   combined_score = 0.6 * pct_float_rank + 0.4 * dtc_rank
  - keep:   top N (default 150) per settlement date

Writes data/screened_universe.parquet with one row per (settlementDate, symbolCode).

Float caveat: Finnhub profile2 returns CURRENT shareOutstanding — point-in-time
float would require a paid vendor with historical fundamentals. For a 2-year
backtest on small-caps this introduces some look-ahead noise but is standard.
"""
from __future__ import annotations
import time, urllib3, json
from pathlib import Path

import pandas as pd
import numpy as np
import requests

urllib3.disable_warnings()

FINNHUB_KEY = "<REDACTED>"
PROFILE_API = "https://finnhub.io/api/v1/stock/profile2"
SEC_HEADERS = {"User-Agent": "Research-Backtest <email>",
               "Accept-Encoding": "gzip, deflate"}
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SHARES_TS_CACHE = Path(__file__).parent / "cache" / "sec_shares_ts.json"

ROOT = Path(__file__).parent
FINRA = ROOT / "data" / "finra_short_interest.parquet"
FLOAT_CACHE = ROOT / "cache" / "float_shares.json"
OUT = ROOT / "data" / "screened_universe.parquet"

MIN_DTC = 5.0
MIN_SI = 1_000_000
MIN_ADV = 100_000
MIN_SI_PCT_FLOAT = 10.0
TOP_N = 150

# FINRA marketClassCode: NYSE, NNM (Nasdaq), SC (Nasdaq SmallCap), and others.
# 'OTC' / 'OTCE' are off-exchange — exclude unless include_otc.
OTC_CLASSES = {"OTCE", "OTC", "OTHER", "OTHERS"}


def load_float_cache() -> dict[str, float]:
    if FLOAT_CACHE.exists():
        return json.loads(FLOAT_CACHE.read_text())
    return {}


def save_float_cache(d: dict) -> None:
    FLOAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FLOAT_CACHE.write_text(json.dumps(d, indent=0))


_T2CIK: dict[str, str] | None = None


def _load_ticker_cik_map() -> dict[str, str]:
    global _T2CIK
    if _T2CIK is not None:
        return _T2CIK
    r = requests.get(SEC_TICKERS, headers=SEC_HEADERS, timeout=30, verify=False)
    r.raise_for_status()
    j = r.json()
    _T2CIK = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in j.values()}
    return _T2CIK


def fetch_shares_timeseries(ticker: str) -> list[tuple[str, float]] | None:
    """Return [(end_date_str, shares_outstanding), ...] sorted by date. None on failure."""
    t2cik = _load_ticker_cik_map()
    cik = t2cik.get(ticker.upper())
    if not cik:
        return None
    try:
        r = requests.get(SEC_FACTS.format(cik=cik), headers=SEC_HEADERS, timeout=30, verify=False)
        if r.status_code != 200:
            return None
        j = r.json()
        # Try us-gaap CommonStockSharesOutstanding, then dei EntityCommonStockSharesOutstanding
        for ns, key in (("us-gaap", "CommonStockSharesOutstanding"),
                        ("dei", "EntityCommonStockSharesOutstanding")):
            units = j.get("facts", {}).get(ns, {}).get(key, {}).get("units", {}).get("shares", [])
            if units:
                pts = sorted({(u["end"], float(u["val"])) for u in units if "end" in u and "val" in u})
                return pts
        return None
    except Exception:
        return None


def fetch_float_finnhub(ticker: str) -> float | None:
    """Finnhub /stock/profile2 fallback — shareOutstanding in millions."""
    try:
        r = requests.get(PROFILE_API, params={"symbol": ticker, "token": FINNHUB_KEY},
                         timeout=15, verify=False)
        if r.status_code != 200:
            return None
        j = r.json()
        so = j.get("shareOutstanding")
        return float(so) * 1_000_000 if so else None
    except Exception:
        return None


def load_shares_ts_cache() -> dict[str, list[tuple[str, float]] | None]:
    if SHARES_TS_CACHE.exists():
        d = json.loads(SHARES_TS_CACHE.read_text())
        # convert nested lists back to list-of-tuples
        return {k: [tuple(p) for p in v] if v else None for k, v in d.items()}
    return {}


def save_shares_ts_cache(d: dict) -> None:
    SHARES_TS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SHARES_TS_CACHE.write_text(json.dumps(d, default=list))


def shares_at_date(ts: list[tuple[str, float]] | None, asof: str) -> float | None:
    """Most recent shares-outstanding entry with end <= asof (YYYY-MM-DD)."""
    if not ts:
        return None
    valid = [v for d, v in ts if d <= asof]
    return valid[-1] if valid else (ts[0][1] if ts else None)


def fetch_float(ticker: str) -> float | None:
    """Legacy single-value fetcher (kept for compat — uses Finnhub fallback)."""
    return fetch_float_finnhub(ticker)


def enrich_shares_ts(tickers: list[str]) -> dict[str, list[tuple[str, float]] | None]:
    """Per-ticker shares-outstanding TIME SERIES from SEC EDGAR (with Finnhub fallback)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = load_shares_ts_cache()
    todo = [t for t in tickers if t not in cache]
    print(f"[screen] shares-ts cache: {len(cache)} hit, {len(todo)} miss", flush=True)
    # Pre-load CIK map once (single-threaded)
    if todo:
        _load_ticker_cik_map()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:  # SEC: 10 req/s polite limit
        futs = {ex.submit(fetch_shares_timeseries, t): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                ts = fut.result()
            except Exception:
                ts = None
            # Finnhub fallback for tickers SEC has nothing on (e.g. ADRs, recent IPOs)
            if not ts:
                fh = fetch_float_finnhub(t)
                ts = [("2020-01-01", fh)] if fh else None
            cache[t] = ts
            done += 1
            if done % 50 == 0:
                save_shares_ts_cache(cache)
                got = sum(1 for v in cache.values() if v)
                print(f"  shares fetch {done}/{len(todo)} ({got} resolved)", flush=True)
    save_shares_ts_cache(cache)
    return cache


def screen_one_date(df_date: pd.DataFrame, shares_ts: dict, *, include_otc: bool = False) -> pd.DataFrame:
    g = df_date.copy()
    settlement = str(g["settlementDate"].iloc[0])
    # numeric coerce
    for col in ("currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"):
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"])

    # basic FINRA filters
    g = g[g["daysToCoverQuantity"] >= MIN_DTC]
    g = g[g["currentShortPositionQuantity"] >= MIN_SI]
    g = g[g["averageDailyVolumeQuantity"] >= MIN_ADV]
    if not include_otc:
        g = g[~g["marketClassCode"].isin(OTC_CLASSES)]

    if g.empty:
        return g

    # attach point-in-time shares outstanding (most recent <= settlement_date)
    g["float_shares"] = g["symbolCode"].map(lambda t: shares_at_date(shares_ts.get(t), settlement))
    g = g.dropna(subset=["float_shares"])
    g = g[g["float_shares"] > 0]
    g["short_interest_pct_float"] = 100.0 * g["currentShortPositionQuantity"] / g["float_shares"]
    g = g[g["short_interest_pct_float"] >= MIN_SI_PCT_FLOAT]

    if g.empty:
        return g

    # rank within this settlement date (percentile, 0–100)
    g["dtc_rank"] = g["daysToCoverQuantity"].rank(pct=True) * 100.0
    g["pct_float_rank"] = g["short_interest_pct_float"].rank(pct=True) * 100.0
    g["combined_score"] = 0.6 * g["pct_float_rank"] + 0.4 * g["dtc_rank"]
    g = g.sort_values("combined_score", ascending=False).reset_index(drop=True)
    g["combined_rank"] = g.index + 1
    return g.head(TOP_N)


def main() -> int:
    if not FINRA.exists():
        print(f"[screen] missing {FINRA} — run finra_history.py first")
        return 1
    src = pd.read_parquet(FINRA)
    print(f"[screen] loaded FINRA: {len(src):,} rows, {src['settlementDate'].nunique()} dates", flush=True)

    # candidate-symbol union across all dates, BEFORE we filter
    # but to keep float fetch sane, only fetch for symbols that pass the
    # pre-float filters at least once.
    pre = src.copy()
    for col in ("currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"):
        pre[col] = pd.to_numeric(pre[col], errors="coerce")
    pre = pre.dropna(subset=["currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"])
    pre = pre[
        (pre["daysToCoverQuantity"] >= MIN_DTC)
        & (pre["currentShortPositionQuantity"] >= MIN_SI)
        & (pre["averageDailyVolumeQuantity"] >= MIN_ADV)
        & (~pre["marketClassCode"].isin(OTC_CLASSES))
    ]
    cand_syms = sorted(pre["symbolCode"].astype(str).unique())
    print(f"[screen] pre-float candidate symbols: {len(cand_syms):,}", flush=True)

    shares_ts = enrich_shares_ts(cand_syms)
    have = sum(1 for v in shares_ts.values() if v)
    print(f"[screen] shares-ts resolved for {have}/{len(cand_syms)}", flush=True)

    frames: list[pd.DataFrame] = []
    for d, g in src.groupby("settlementDate"):
        s = screen_one_date(g, shares_ts)
        if not s.empty:
            frames.append(s)
            print(f"  {d}: kept {len(s)} of {len(g)}", flush=True)

    if not frames:
        print("[screen] empty universe")
        return 1
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[screen] wrote {len(out):,} screened rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
