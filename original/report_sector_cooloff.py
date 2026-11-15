"""
report_sector_cooloff.py — tests whether a SECTOR-level cooling-off
(instead of/in addition to the validated per-TICKER cooling-off) helps the
LONG drift_low top-5 strategy. Motivation: if blowup names cluster by
sector (e.g. small-cap biotech), a sector-level "sit out" after a sector
loss might generalize better than waiting for the exact same ticker to
reappear.

Fetches Finnhub company profile (finnhubIndustry) for every traded ticker,
caches to data/ticker_sectors.parquet.

Train/test disciplined: cooloff days chosen on TRAIN, evaluated ONCE on
TEST. Tested standalone and stacked on top of the validated per-ticker
cooling-off (5d). 2.5%/3.0% reported separately.

Output: output/sector_cooloff.md
"""
from pathlib import Path
import time
import requests
import pandas as pd

ROOT = Path("<local-path>")
SECTOR_CACHE = ROOT / "data" / "ticker_sectors.parquet"
OUT_MD = ROOT / "output" / "sector_cooloff.md"

KEY = "<REDACTED>"
TRAIN_CUTOFF = pd.Timestamp("2026-01-22")
THRESHOLDS = [2.5, 3.0]

df = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
sel = df[(df["selected"]) & (df["side"] == "LONG")].copy()
tickers = sorted(sel["ticker"].unique())
print(f"tickers to fetch sector for: {len(tickers)}")

if SECTOR_CACHE.exists():
    sec = pd.read_parquet(SECTOR_CACHE)
    have = set(sec["ticker"])
else:
    sec = pd.DataFrame(columns=["ticker", "industry"])
    have = set()

missing = [t for t in tickers if t not in have]
print(f"missing: {len(missing)}")
new_rows = []
for i, t in enumerate(missing):
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/profile2",
                          params={"symbol": t, "token": KEY}, timeout=20, verify=False)
        ind = r.json().get("finnhubIndustry") if r.status_code == 200 else None
    except Exception:
        ind = None
    new_rows.append({"ticker": t, "industry": ind or "Unknown"})
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(missing)}")
    time.sleep(0.05)

if new_rows:
    sec = pd.concat([sec, pd.DataFrame(new_rows)], ignore_index=True)
    sec.to_parquet(SECTOR_CACHE, index=False)
    print(f"saved {len(sec)} ticker->industry rows")

sel = sel.merge(sec, on="ticker", how="left")
sel["industry"] = sel["industry"].fillna("Unknown")
print(sel["industry"].value_counts().head(15))


def perf(s):
    n = len(s)
    hit = (s["realized_return_pct"] > 0).mean() * 100 if n else 0
    sumr = s["realized_return_pct"].sum()
    return n, round(hit, 2), round(sumr, 2)


def apply_cooling_off(data, cooloff_days, key_col):
    data = data.sort_values(["trade_date", key_col]).reset_index(drop=True)
    last_loss = {}
    keep = []
    for _, row in data.iterrows():
        td = row["trade_date"]; k = row[key_col]
        ll = last_loss.get(k)
        if ll is not None and (td - ll).days <= cooloff_days:
            keep.append(False)
        else:
            keep.append(True)
            if row["realized_return_pct"] < 0:
                last_loss[k] = td
    data = data.copy(); data["keep"] = keep
    return data[data["keep"]].drop(columns="keep")


def tbl(headers, rows):
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["|" + "|".join(str(h) for h in headers) + "|", sep]
    for row in rows:
        out.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(out)


lines = [
    "# Sector-Level Cooling-Off (LONG)",
    "",
    "Tests whether a SECTOR-level cooling-off (Finnhub finnhubIndustry) "
    "improves on / adds to the validated per-ticker cooling-off (5d). "
    "Motivation: if blowup names cluster by sector, sector-level cooling-"
    "off could catch correlated risk that per-ticker cooling-off misses "
    "(a DIFFERENT ticker in the same sector having problems).",
    "",
    "Train/test disciplined: cooloff days chosen on TRAIN, evaluated ONCE "
    "on TEST. 2.5%/3.0% reported separately.",
    "",
]

cooloffs = [0, 5, 10, 15, 20]
summary = []

for thr in THRESHOLDS:
    pop = sel[sel["gap_threshold_pct"] == thr]
    base_tr = perf(pop[pop["trade_date"] < TRAIN_CUTOFF])
    base_te = perf(pop[pop["trade_date"] >= TRAIN_CUTOFF])

    lines.append(f"## Gap threshold {thr}%")
    lines.append("")
    lines.append(tbl(["", "n", "hit%", "sum%"],
                      [("TRAIN baseline", *base_tr), ("TEST baseline", *base_te)]))
    lines.append("")
    lines.append("### Sector-only cooling-off")
    lines.append("")
    rows = []
    sector_results = []
    for cd in cooloffs:
        f = pop if cd == 0 else apply_cooling_off(pop, cd, "industry")
        tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
        te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])
        sector_results.append((cd, tr, te))
        rows.append((f"{cd}d", *tr, *te))
    lines.append(tbl(["Cooloff", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%"], rows))
    best_sector = max(sector_results, key=lambda x: x[1][2])
    v1 = "PASS" if best_sector[2][2] > base_te[2] else "FAIL"
    lines += ["", f"TRAIN-best sector cooloff={best_sector[0]}d -> TEST {best_sector[2][2]:.2f}% "
              f"vs baseline {base_te[2]:.2f}% -> **{v1}**", ""]

    lines.append("### Ticker cooling-off (5d, validated) + sector cooling-off stacked")
    lines.append("")
    pop_tk5 = apply_cooling_off(pop, 5, "ticker")
    rows = []
    stack_results = []
    for cd in cooloffs:
        f = pop_tk5 if cd == 0 else apply_cooling_off(pop_tk5, cd, "industry")
        tr = perf(f[f["trade_date"] < TRAIN_CUTOFF])
        te = perf(f[f["trade_date"] >= TRAIN_CUTOFF])
        stack_results.append((cd, tr, te))
        rows.append((f"{cd}d", *tr, *te))
    lines.append(tbl(["+sector cooloff", "TRAIN n", "TRAIN hit%", "TRAIN sum%",
                       "TEST n", "TEST hit%", "TEST sum%"], rows))
    ref = stack_results[0]  # ticker-only (sector cooloff=0)
    best_stack = max(stack_results, key=lambda x: x[1][2])
    v2 = "PASS" if best_stack[2][2] > ref[2][2] else "FAIL"
    lines += ["", f"Ticker-only (5d): TEST {ref[2][2]:.2f}%. TRAIN-best +sector "
              f"cooloff={best_stack[0]}d: TEST {best_stack[2][2]:.2f}% -> **{v2}** "
              "(vs ticker-only, not vs no-control baseline)", "", "---", ""]

    summary.append((thr, base_te, best_sector, ref, best_stack))
    print(f"\n=== {thr}% ===")
    print(f"baseline TEST: {base_te[2]:.2f}%")
    print(f"sector-only best (cd={best_sector[0]}): TEST {best_sector[2][2]:.2f}% ({v1})")
    print(f"ticker(5d) only: TEST {ref[2][2]:.2f}%")
    print(f"ticker(5d)+sector best (cd={best_stack[0]}): TEST {best_stack[2][2]:.2f}% ({v2})")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT_MD}")
