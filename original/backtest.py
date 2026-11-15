"""
backtest.py — replicate the post-publication gap strategy:

For each (settlementDate, ticker) in the screened universe, scan the
[settlementDate, settlementDate + 10 trading days] window. On each session:

  gap_pct = (open_09:30 - prev_close) / prev_close

  LONG rule  ("hold_above_open_945_tpN"):
    if gap_pct >= threshold AND price@09:45 >= open_09:30:
      entry: price@09:45
      exit:  hit +TP% intraday  OR  EOD close

  SHORT rule ("failed_bounce_1000_tpM"):
    if gap_pct >= threshold AND failed_bounce_at_1000:
      entry: price@10:00
      exit:  hit -TP% intraday  OR  EOD close

failed_bounce_at_1000 (our spec, flagged): price@10:00 < open_09:30 AND
price@10:00 < price@09:45  (i.e. rolled back below open after the 09:30-09:45
rally).

Writes data/trades.parquet — one row per (date, ticker, threshold, side).
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime as dt

import numpy as np
import pandas as pd

import regime

ROOT = Path(__file__).parent
SCR = ROOT / "data" / "screened_universe.parquet"
CACHE_DIR = ROOT / "cache" / "intraday"
OUT = ROOT / "data" / "trades.parquet"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

GAP_THRESHOLDS = [2.5, 3.0]
LONG_TP_PCTS = [3.0]
SHORT_TP_PCTS = [2.5]
HOLD_DAYS = 10
PUBLICATION_LAG_BUSINESS_DAYS = 8   # FINRA publishes ~8 business days after settlement

# ---------- Item 2: trailing-ATR exit (regime-based, replaces fixed TP) ----------
# activation: favorable move (as % of ATR) before the trailing stop arms
# trail: trailing-stop distance from the best price seen (as % of ATR)
# hard_stop: stop loss before activation (as % of ATR)
TRAIL_ACT_MULT = 1.0
TRAIL_TRAIL_MULT = 0.5
TRAIL_STOP_MULT = 1.0


def publication_date(settlement: dt.date) -> dt.date:
    """Add ~8 business days to settlement to approximate FINRA publication date."""
    d = settlement
    added = 0
    while added < PUBLICATION_LAG_BUSINESS_DAYS:
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


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


def session_metrics(bars_session: pd.DataFrame) -> dict | None:
    """Return open_0930, p_0945, p_1000, hi_after_open, etc., or None if RTH bars missing."""
    rth = bars_session[(bars_session["minute_et"] >= "09:30") & (bars_session["minute_et"] <= "16:00")]
    if rth.empty:
        return None
    b0930 = rth[rth["minute_et"] == "09:30"]
    b0945 = rth[rth["minute_et"] == "09:45"]
    b1000 = rth[rth["minute_et"] == "10:00"]
    if b0930.empty or b0945.empty or b1000.empty:
        return None
    return {
        "open_0930": float(b0930["o"].iloc[0]),
        "price_0945": float(b0945["c"].iloc[0]),
        "price_1000": float(b1000["c"].iloc[0]),
        "rth_bars": rth,
    }


def find_exit(bars_after: pd.DataFrame, entry_price: float, side: str, tp_pct: float) -> tuple[float, str, str]:
    """Walk bars after entry; return (exit_price, exit_minute_et, exit_rule)."""
    tp_mul = 1 + tp_pct / 100 if side == "LONG" else 1 - tp_pct / 100
    for _, b in bars_after.iterrows():
        if side == "LONG" and b["h"] >= entry_price * tp_mul:
            return entry_price * tp_mul, b["minute_et"], "take_profit"
        if side == "SHORT" and b["l"] <= entry_price * tp_mul:
            return entry_price * tp_mul, b["minute_et"], "take_profit"
    last = bars_after.iloc[-1]
    return float(last["c"]), last["minute_et"], "eod"


def find_exit_trailing(bars_after: pd.DataFrame, entry_price: float, side: str,
                        regime_vol_pct: float) -> tuple[float, str, str]:
    """Regime-sized trailing stop: arm a trailing stop once the trade has moved
    favorably by TRAIL_ACT_MULT x ATR%; trail it TRAIL_TRAIL_MULT x ATR% behind
    the best price seen. Before activation, a hard stop at TRAIL_STOP_MULT x
    ATR% protects the trade. Falls back to EOD close if neither triggers."""
    activation_pct = regime_vol_pct * TRAIL_ACT_MULT
    trail_pct = regime_vol_pct * TRAIL_TRAIL_MULT
    stop_pct = regime_vol_pct * TRAIL_STOP_MULT

    best = entry_price
    activated = False
    if side == "LONG":
        hard_stop = entry_price * (1 - stop_pct / 100)
        for _, b in bars_after.iterrows():
            best = max(best, b["h"])
            if not activated:
                if 100.0 * (best - entry_price) / entry_price >= activation_pct:
                    activated = True
                elif b["l"] <= hard_stop:
                    return hard_stop, b["minute_et"], "hard_stop"
            if activated:
                trail_level = best * (1 - trail_pct / 100)
                if b["l"] <= trail_level:
                    return trail_level, b["minute_et"], "trailing_stop"
    else:  # SHORT
        hard_stop = entry_price * (1 + stop_pct / 100)
        for _, b in bars_after.iterrows():
            best = min(best, b["l"])
            if not activated:
                if 100.0 * (entry_price - best) / entry_price >= activation_pct:
                    activated = True
                elif b["h"] >= hard_stop:
                    return hard_stop, b["minute_et"], "hard_stop"
            if activated:
                trail_level = best * (1 + trail_pct / 100)
                if b["h"] >= trail_level:
                    return trail_level, b["minute_et"], "trailing_stop"

    last = bars_after.iloc[-1]
    return float(last["c"]), last["minute_et"], "eod"


def prev_close(bars: pd.DataFrame, session: dt.date) -> float | None:
    prior = bars[bars["session"] < session]
    if prior.empty:
        return None
    last_sess = prior["session"].max()
    last_rth = prior[(prior["session"] == last_sess) & (prior["minute_et"] <= "16:00")]
    if last_rth.empty:
        return None
    return float(last_rth.iloc[-1]["c"])


def backtest_one_ticker_window(ticker: str, settlement_date: dt.date, bars: pd.DataFrame,
                                daily: pd.DataFrame, atr_pct: pd.Series, atr_pctile: pd.Series) -> list[dict]:
    """Scan HOLD_DAYS sessions starting at PUBLICATION_DATE (~settlement+8bd)."""
    if bars.empty:
        return []
    pub = publication_date(settlement_date)
    sessions = sorted({s for s in bars["session"].unique() if s >= pub})
    sessions = sessions[:HOLD_DAYS]

    out: list[dict] = []
    for sess in sessions:
        sess_bars = bars[bars["session"] == sess]
        pc = prev_close(bars, sess)
        if pc is None or pc <= 0:
            continue
        m = session_metrics(sess_bars)
        if m is None:
            continue
        gap_pct = 100.0 * (m["open_0930"] - pc) / pc
        regime_vol = regime.regime_vol_as_of(daily, atr_pct, sess)
        regime_pctile = regime.regime_pctile_as_of(daily, atr_pctile, sess)

        for thr in GAP_THRESHOLDS:
            if gap_pct < thr:
                continue

            # ---------- LONG rule ----------
            if m["price_0945"] >= m["open_0930"]:
                entry = m["price_0945"]
                bars_after = m["rth_bars"][m["rth_bars"]["minute_et"] >= "09:45"].iloc[1:]
                if not bars_after.empty:
                    for tp in LONG_TP_PCTS:
                        ex_px, ex_min, ex_rule = find_exit(bars_after, entry, "LONG", tp)
                        out.append({
                            "trade_date": sess.isoformat(),
                            "ticker": ticker,
                            "settlement_date": settlement_date.isoformat(),
                            "publication_date": pub.isoformat(),
                            "side": "LONG",
                            "gap_threshold_pct": thr,
                            "gap_pct": gap_pct,
                            "rule_name": f"hold_above_open_945_tp{int(tp)}",
                            "tp_pct": tp,
                            "signal_time_et": "09:45",
                            "entry_price": entry,
                            "exit_price": ex_px,
                            "exit_time_et": ex_min,
                            "exit_rule": ex_rule,
                            "realized_return_pct": 100.0 * (ex_px - entry) / entry,
                        })

                    # ---- trailing-ATR exit variant (item 2) ----
                    if regime_vol is not None:
                        ex_px, ex_min, ex_rule = find_exit_trailing(bars_after, entry, "LONG", regime_vol)
                        out.append({
                            "trade_date": sess.isoformat(),
                            "ticker": ticker,
                            "settlement_date": settlement_date.isoformat(),
                            "publication_date": pub.isoformat(),
                            "side": "LONG",
                            "gap_threshold_pct": thr,
                            "gap_pct": gap_pct,
                            "rule_name": "trailing_atr",
                            "regime_vol_pct": regime_vol,
                            "atr_pctile": regime_pctile,
                            "signal_time_et": "09:45",
                            "entry_price": entry,
                            "exit_price": ex_px,
                            "exit_time_et": ex_min,
                            "exit_rule": ex_rule,
                            "realized_return_pct": 100.0 * (ex_px - entry) / entry,
                        })

            # ---------- SHORT rule (failed bounce by 10:00) ----------
            failed_bounce = (m["price_1000"] < m["open_0930"]) and (m["price_1000"] < m["price_0945"])
            if failed_bounce:
                entry = m["price_1000"]
                bars_after = m["rth_bars"][m["rth_bars"]["minute_et"] >= "10:00"].iloc[1:]
                if not bars_after.empty:
                    for tp in SHORT_TP_PCTS:
                        ex_px, ex_min, ex_rule = find_exit(bars_after, entry, "SHORT", tp)
                        out.append({
                            "trade_date": sess.isoformat(),
                            "ticker": ticker,
                            "settlement_date": settlement_date.isoformat(),
                            "publication_date": pub.isoformat(),
                            "side": "SHORT",
                            "gap_threshold_pct": thr,
                            "gap_pct": gap_pct,
                            "rule_name": f"failed_bounce_1000_tp_{tp}",
                            "tp_pct": tp,
                            "signal_time_et": "10:00",
                            "entry_price": entry,
                            "exit_price": ex_px,
                            "exit_time_et": ex_min,
                            "exit_rule": ex_rule,
                            "realized_return_pct": 100.0 * (entry - ex_px) / entry,
                        })

                    # ---- trailing-ATR exit variant (item 2) ----
                    if regime_vol is not None:
                        ex_px, ex_min, ex_rule = find_exit_trailing(bars_after, entry, "SHORT", regime_vol)
                        out.append({
                            "trade_date": sess.isoformat(),
                            "ticker": ticker,
                            "settlement_date": settlement_date.isoformat(),
                            "publication_date": pub.isoformat(),
                            "side": "SHORT",
                            "gap_threshold_pct": thr,
                            "gap_pct": gap_pct,
                            "rule_name": "trailing_atr",
                            "regime_vol_pct": regime_vol,
                            "atr_pctile": regime_pctile,
                            "signal_time_et": "10:00",
                            "entry_price": entry,
                            "exit_price": ex_px,
                            "exit_time_et": ex_min,
                            "exit_rule": ex_rule,
                            "realized_return_pct": 100.0 * (entry - ex_px) / entry,
                        })
    return out


def main() -> int:
    if not SCR.exists():
        print(f"[backtest] missing {SCR}")
        return 1
    scr = pd.read_parquet(SCR)
    scr["settlementDate"] = pd.to_datetime(scr["settlementDate"]).dt.date
    print(f"[backtest] universe rows: {len(scr):,}", flush=True)

    all_trades: list[dict] = []
    last_ticker_bars: dict[str, pd.DataFrame] = {}
    last_ticker_regime: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}

    n_total = len(scr)
    for i, row in enumerate(scr.itertuples(index=False), 1):
        tkr = row.symbolCode
        d = row.settlementDate
        if tkr not in last_ticker_bars:
            last_ticker_bars[tkr] = load_bars(tkr)
        if tkr not in last_ticker_regime:
            daily = regime.load_daily_ohlc(tkr)
            atr_pct = regime.atr_pct_series(daily)
            atr_pctile = regime.atr_percentile_series(atr_pct)
            last_ticker_regime[tkr] = (daily, atr_pct, atr_pctile)
        bars = last_ticker_bars[tkr]
        daily, atr_pct, atr_pctile = last_ticker_regime[tkr]
        trades = backtest_one_ticker_window(tkr, d, bars, daily, atr_pct, atr_pctile)
        all_trades.extend(trades)
        if i % 200 == 0:
            print(f"  [{i}/{n_total}] {tkr} {d}: cumulative trades={len(all_trades)}", flush=True)

    out = pd.DataFrame(all_trades)
    if out.empty:
        print("[backtest] zero trades")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[backtest] wrote {len(out):,} trade rows -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
