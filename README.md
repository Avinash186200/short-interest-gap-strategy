# Short-Interest Gap Strategy

An intraday backtest of a simple idea: heavily shorted small caps that gap up in the days after FINRA publishes their short interest may keep running (go long) or may fail and fade (go short). I screen FINRA's bi-monthly short-interest data, trade the gap rules on 1-minute bars, and then test whether a daily ranking of the candidates adds anything. The strategy idea was shared by a third party; this is my independent replication and audit of it, and the code is mine.

The original research used vendor price data I can't share, so this repo runs on a **synthetic market with the same inputs**: a FINRA-style short-interest table, a shares-outstanding table with filing dates, and 1-minute bars. You can clone it and reproduce every number below.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## At a glance

| | |
|---|---|
| **Input** | FINRA consolidated short interest (`settlementDate`, `symbolCode`, `currentShortPositionQuantity`, `averageDailyVolumeQuantity`, `daysToCoverQuantity`, `marketClassCode`), shares outstanding with filing dates, 1-minute regular-hours bars |
| **Method** | Short-interest screen per settlement date, then LONG "hold above open" and SHORT "failed bounce" gap rules in the 10 sessions after publication. Candidates are ranked daily by pre-publication price drift or volume, with optional cooling-off. Configs are chosen by expanding walk-forward and checked on a final hold-out and against random picks |
| **Output** | Candidate trade list, per-side summary (naive in-sample, walk-forward out-of-sample, hold-out, take-all, random control), deflated Sharpe for the best-of-grid config |
| **Leakage controls** | Holiday-aware publication date and first trade on the session *after* it, point-in-time shares (filing date), entry on the bar after the signal, costs on every trade, selection on training data only, random-pick control, noise test in CI |
| **Status** | Research complete. SHORT side: no edge. LONG side: one candidate rule that was never validated on a clean hold-out. The numbers below are synthetic and only illustrate the method |

## How it works

```
FINRA short interest ──► screen ──► post-publication window ──► gap rules on 1-min bars ──► candidates
  + shares (filed)      screen.py     calendar.py                trades.py                       │
                                                                                                  ▼
             hold-out + random control ◄── walk-forward choice ◄── daily rank + cooling-off ◄─────┘
                    validation.py              validation.py           selection.py, metrics.py
```

1. **Screen** each settlement date: days-to-cover >= 5, short interest >= 1M shares, ADV >= 100k, listed names only, short interest >= 10% of shares outstanding. Rank by 0.6 x short-%-of-shares rank + 0.4 x days-to-cover rank and keep the top 150.
2. **Window**: FINRA publishes 7 exchange business days after settlement. The first tradable session is the one *after* the publication date, and the window runs for 10 sessions.
3. **Gap rules**, on a session where the open is at least 2.5% above the prior close:
   - LONG: the 09:45 bar closes at or above the open, so buy at 09:46. Exit at +3% or the close.
   - SHORT: the 10:00 bar closes below the open and below the 09:45 close, so sell at 10:01. Exit at -2.5% or the close.
   - Each round trip pays 15 bps per side.
4. **Daily selection**: drop names still cooling off after a recent loss, then rank the day's candidates by `price_drift_pct` (settlement close to the last close before the window) or by `volume_ratio`, and take the top N.
5. **Portfolio**: capital is split into 5 slots, so each trade gets 1/max(5, trades that day) of equity. Returns compound, and flat days count as zero.
6. **Evaluation**: I score 34 configs (4 rankings x top-N {1, 2, 3, 5} x cooling-off {0, 5}, plus "take all"). The config is chosen on training data only, re-chosen each quarter, and then checked once on the last 4 months.

## Results on synthetic data

`python scripts/run_backtest.py` (seed 7, 150 tickers, Jan 2024 to May 2026, 15 bps per side). The planted effect is deliberately strong, so the mechanics are easy to see:

| LONG | Config | Trades | Hit rate | Avg net / trade | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| Naive: best of 34 on the full sample | drift_low, top 2 | 1,030 | 71.7% | +0.75% | 4.66 | -7.2% |
| Walk-forward out-of-sample | re-chosen each quarter | 717 | 70.2% | +0.63% | 5.02 | -6.9% |
| Hold-out, config chosen on research period | drift_low, top 3 | 227 | 70.0% | +0.49% | 3.56 | -8.3% |
| Hold-out, take every candidate | all | 578 | 60.9% | -0.08% | -1.06 | -16.0% |
| Hold-out, pre-registered spec | drift_low, top 5, cool 5 | 341 | 66.3% | +0.26% | 2.05 | -12.0% |

On the hold-out, the chosen config beats 100% of 200 random top-3 draws (random median Sharpe -0.98). The SHORT side has no planted effect, and it loses money in every view: walk-forward Sharpe -2.87, hold-out -2.02.

**The same run with the effect switched off** (`--signal-strength 0`):

| LONG, no planted effect | Config | Avg net / trade | Sharpe | Deflated Sharpe p |
|---|---|---|---|---|
| Naive: best of 34 on the full sample | volume_low, top 2, cool 5 | +0.08% | 0.44 | 0.01 |
| Walk-forward out-of-sample | re-chosen each quarter | -0.16% | -0.69 | |
| Hold-out, pre-registered spec | drift_low, top 5, cool 5 | -0.44% | -3.68 | |

The naive best-of-34 still shows a positive Sharpe on pure noise. The deflated Sharpe (p = 0.01) and the walk-forward correctly reject it. One more thing shows up here. On noise, random top-2 picks have a "better" Sharpe than taking everything (-2.74 vs -4.48), only because they trade less. So "any top-N beats take-all" is not evidence that a ranking works. That is what the random control is for.

**With a weaker effect** (`--signal-strength 0.6`), the walk-forward still finds it (Sharpe 1.52), but the single config chosen for the hold-out is the wrong one (volume_low, Sharpe -0.47). Picking the best of 34 configs with a weak effect is fragile, and one 4-month hold-out cannot tell you much.

> **About the real-data result.** On the 2024-2026 FINRA universe, the SHORT side lost money under almost every variant I tried. The one exception, three filters stacked together, was near break-even on about 20 test trades before costs, too few to tell apart from noise. I treat the SHORT side as no edge. On the LONG side, the original research ended on "take the 5 names whose price fell most before publication". It was positive in the original 4-month test window for every top-N from 1 to 5, but it was not the only positive variant there. And that window had been looked at for many decisions (ranking formula and direction, top-N, cooling-off, stops, filters), the direction was the opposite of the starting hypothesis, and there was no random-pick control. Every top-5 ranking in that test window, including the opposite direction, beat taking all candidates. So a large part of the apparent edge may simply be trading fewer names on busy days. I treat the LONG rule as an unvalidated candidate and don't publish its numbers. It needs a walk-forward run with costs, the random control, and fresh data.

## Bias and leakage audit

I found these when I re-audited the original research code. The new code fixes each one. The details are in [docs/methodology.md](docs/methodology.md).

| Issue | Effect | Fix |
|---|---|---|
| One 4-month TEST window reused for 20+ decisions (formula, direction, top-N, cooling-off, stop, TP, filters) | The "held-out" result was really in-sample, so the best variant looked better than it is | Walk-forward selection inside the research period, a final hold-out scored once, and a deflated Sharpe over all 34 configs |
| Ranking direction adopted after seeing results (the opposite of the hypothesis, found through a bug) | Post-hoc hypothesis, so a selection effect counted as a signal | Both directions sit in the grid, and only training data chooses between them |
| No random-selection control | The benefit of trading fewer names was credited to the ranking signal | Random top-N control, with the same N and cooling-off |
| Returns summed per trade, with no capital limit | Drawdowns of more than -100% and a Sharpe that no account could earn | Slot-based, compounding portfolio. Flat days count |
| No transaction costs | Small-cap intraday edges overstated | 15 bps per side by default (`--cost-bps`) |
| Entered at the close of the same bar that triggered the signal | Zero-latency fill | Entry at the open of the next bar |
| Publication date was "settlement + 8 weekdays" with no holidays, and trading was allowed on that day | When a holiday fell in the lag, trades landed on the actual publication day, possibly before the data was out | Holiday-aware 7 business days, with the first trade on the next session |
| Shares outstanding matched by reporting period end, falling back to the first *future* value or a current vendor figure | Look-ahead in the short-%-of-shares screen | Only figures filed on or before the settlement date, with no backfill |
| 2.5% and 3.0% gap thresholds (about 95% the same trades) reported as two confirmations | One piece of evidence counted twice | One threshold, fixed in advance |
| Nothing stopped one name being traded twice on a day when two publication windows overlap | Double-counted trades | De-duplicated per (day, ticker, side) |
| Price history only for names the vendor still serves | Survivorship: blow-ups that delisted can drop out | Coverage counts are reported (`no_price_data`, `missing_bars`). Delisted data is needed to close this fully |

## Repo layout

```
src/sigap/   calendar.py · screen.py · trades.py · selection.py · metrics.py · validation.py · sources.py · synthetic.py
scripts/     run_backtest.py · make_synthetic_data.py
tests/       publication timing, point-in-time shares, entry timing, costs, cooling-off, walk-forward boundaries,
             planted-signal recovery, noise test, file-format round trip
docs/        methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_backtest.py                        # synthetic, planted effect
python scripts/run_backtest.py --signal-strength 0    # no effect: should show no edge
```

### Data format (for your own data)

```bash
python scripts/make_synthetic_data.py --n-tickers 20 --out data/synthetic   # writes an example of the format
python scripts/run_backtest.py --data-dir data/synthetic
```

`--data-dir` expects `finra_short_interest.csv` (FINRA column names), `shares_outstanding.csv` (`symbol, period_end, filed_date, shares`) and `bars/<TICKER>.parquet` with columns `t` (bar start, unix seconds UTC), `o, h, l, c, v`.

## License

MIT
