"""
finra_history.py — pull FINRA consolidated short-interest, all available
settlement dates from MIN_DATE onward, write one parquet keyed by
(settlementDate, symbolCode).

API:
  GET  /partitions/group/otcMarket/name/consolidatedShortInterest
       -> list of all settlement dates available
  POST /data/group/otcMarket/name/consolidatedShortInterest
       body: {limit, offset, compareFilters: [{fieldName, fieldValue, compareType: 'EQUAL'}]}
       -> 5000-row pages

Partition fields only support EQUAL — date range queries are not allowed.
"""
from __future__ import annotations
import io, time, urllib3
from pathlib import Path

import pandas as pd
import requests

urllib3.disable_warnings()

BASE = "https://api.finra.org"
PART_URL = f"{BASE}/partitions/group/otcMarket/name/consolidatedShortInterest"
DATA_URL = f"{BASE}/data/group/otcMarket/name/consolidatedShortInterest"
OUT = Path(__file__).parent / "data" / "finra_short_interest.parquet"
MIN_DATE = "2024-01-01"
PAGE = 5000


def list_partitions() -> list[str]:
    r = requests.get(PART_URL, timeout=30, verify=False)
    r.raise_for_status()
    j = r.json()
    return [p["partitions"][0] for p in j.get("availablePartitions", [])]


def fetch_one_date(date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        body = {
            "limit": PAGE,
            "offset": offset,
            "compareFilters": [
                {"fieldName": "settlementDate", "fieldValue": date, "compareType": "EQUAL"}
            ],
        }
        r = requests.post(DATA_URL, json=body, timeout=120, verify=False)
        r.raise_for_status()
        if not r.text.strip():
            break
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
        time.sleep(0.15)
        if offset > 500_000:
            raise RuntimeError(f"runaway on {date}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[finra] listing partitions ...", flush=True)
    all_dates = list_partitions()
    dates = sorted(d for d in all_dates if d >= MIN_DATE)
    print(f"[finra] {len(dates)} partitions >= {MIN_DATE} (out of {len(all_dates)} total)", flush=True)

    existing = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(existing["settlementDate"].astype(str).unique()) if not existing.empty else set()

    new_frames = [existing] if not existing.empty else []
    for i, d in enumerate(dates, 1):
        if d in have:
            print(f"  [{i:>3}/{len(dates)}] {d}  CACHED", flush=True)
            continue
        t0 = time.time()
        try:
            df = fetch_one_date(d)
        except Exception as e:
            print(f"  [{i:>3}/{len(dates)}] {d}  ERROR {e!r}", flush=True)
            continue
        print(f"  [{i:>3}/{len(dates)}] {d}  rows={len(df):>6}  {time.time()-t0:.1f}s", flush=True)
        if not df.empty:
            new_frames.append(df)
        if i % 5 == 0 and new_frames:
            pd.concat(new_frames, ignore_index=True).to_parquet(OUT, index=False)

    if new_frames:
        out = pd.concat(new_frames, ignore_index=True)
        out = out.drop_duplicates(subset=["settlementDate", "symbolCode"], keep="last")
        out.to_parquet(OUT, index=False)
        print(f"[finra] wrote {len(out):,} rows ({out['settlementDate'].nunique()} dates) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
