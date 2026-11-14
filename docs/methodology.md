# Methodology and bias audit

## Idea

Short interest in US equities is reported for a settlement date twice a month and published by FINRA about a week and a half later. The hypothesis was that when a heavily shorted small cap gaps up shortly after its short interest becomes public, either:

- shorts are squeezed and the move continues (LONG "hold above open"), or
- the gap fails early and the stock fades (SHORT "failed bounce").

A later refinement ranked each day's LONG candidates by how far the price had moved between settlement and publication, and traded only the top N.

## Data contract

| Input | Columns | Notes |
|---|---|---|
| FINRA short interest | `settlementDate`, `symbolCode`, `currentShortPositionQuantity`, `averageDailyVolumeQuantity`, `daysToCoverQuantity`, `marketClassCode` | Public data. The original research pulled it from FINRA's public API |
| Shares outstanding | `symbol`, `period_end`, `filed_date`, `shares` | `filed_date` is required, because the screen may only use what was filed |
| 1-minute bars | `t` (bar start, unix seconds), `o`, `h`, `l`, `c`, `v` | Regular hours only (09:30 to 15:59 bar starts) |

The synthetic generator (`sigap.synthetic`) produces all three. It plants one effect: in the post-publication window of a name with short interest of at least 10% of shares, the intraday drift is `0.03 x clip(-drift / 0.15, 0, 1)`, where `drift` is the settlement-to-publication price change. Names that fell before publication drift up. `signal_strength=0` removes the effect. The SHORT side has no planted effect.

## Timing

| Step | Rule |
|---|---|
| Publication | 7 exchange business days after settlement (holiday-aware) |
| First tradable session | The session after the publication date. The release time of day is not something to rely on |
| Ranking features | `price_drift_pct` and `volume_ratio` use closes from the settlement date to the last close before the first tradable session |
| Gap | The 09:30 open against the previous session's close. The gap is skipped if the previous session in the data is more than 5 days back |
| LONG signal / entry | Signal on the 09:45 bar's close. Entry at the 09:46 bar's open |
| SHORT signal / entry | Signal on the 10:00 bar's close. Entry at the 10:01 bar's open |
| Exit | Fixed target, or the last regular-hours bar's close. If an optional stop and the target fall inside the same bar, the stop is assumed to fill first |
| Cooling-off | A loss on session *d* blocks that ticker for sessions within N calendar days after *d*, never on *d* itself |

## Validation design

The original research ended with a good-looking LONG rule. Every step below exists because of how that rule was found.

1. **Grid.** 34 configs per side: ranking in {drift_low, drift_high, volume_low, volume_high} x top-N in {1, 2, 3, 5} x cooling-off in {0, 5 days}, plus "take all" with and without cooling-off. Each config is run once over the full sample as a causal day-by-day loop. Because every decision uses only earlier sessions, slicing that run by date is the same as trading it live.
2. **Naive in-sample.** The best config on the full sample is reported only as a contrast, with its deflated Sharpe ratio (Bailey and López de Prado, 2014). The DSR is the probability that its Sharpe beats the expected maximum of 34 noise trials.
3. **Walk-forward.** Inside the research period (everything except the last 4 months), the first 252 sessions are training only. Then every 63 sessions the config with the best training Sharpe is chosen and traded on the next 63 sessions. The out-of-sample blocks are stitched together.
4. **Hold-out.** The last 4 months are never used for any choice. They are scored once, for the config chosen on the whole research period and for a pre-registered spec (drift_low, top 5, 5-day cooling-off). That is the rule the original research ended with, minus its -18% wide stop. In the original test window the stop added almost nothing once cooling-off was on.
5. **Random control.** The chosen config's top-N and cooling-off are rerun 200 times with random daily picks. The report gives the share of random draws the chosen config beats on the hold-out.

## Portfolio accounting

The original reports summed per-trade percentage returns ("sum_ret"). That has no capital limit, so a "-300% drawdown" was possible, and the Sharpe ratios did not describe any account. Here the account has 5 slots. Each trade gets 1/max(5, trades that day) of equity, daily returns compound, every exchange session counts, and Sharpe is annualised with sqrt(252).

## Bias audit of the original research

| # | Issue in the original | Effect | Fix here |
|---|---|---|---|
| 1 | One 4-month TEST window was used to judge 20+ variants: ranking formula and direction, top-N, cooling-off, wide stop, tight stop, take-profit level, liquidity floor, drift floor, publication-day window, volume confirmation, market and sector gates, and partial scale-out. The final spec then reported full-sample numbers that mixed train and test | The TEST result had become in-sample. The headline figures were best-of-many | Walk-forward selection, a hold-out scored once, and a deflated Sharpe |
| 2 | The ranking hypothesis was "drift confirms direction". The opposite direction was adopted after a bug inverted the ranking and the inverted version looked better | Post-hoc hypothesis | Both directions are grid members, and only training data chooses |
| 3 | There was no control for "trade fewer names". In the original TEST window, every top-5 ranking, including the opposite direction, beat taking all candidates | Capacity limits credited to the signal | Random top-N control |
| 4 | Summed returns with no capital constraint | Unrealistic risk figures | Slot-based compounding portfolio |
| 5 | No costs | Overstated edge on illiquid small caps | 15 bps per side by default |
| 6 | Entry at the close of the bar that produced the signal | Zero-latency assumption | Next-bar open |
| 7 | Publication = settlement + 8 weekdays, ignoring holidays, with trading allowed on that day | When a holiday fell in the lag, a trade could come before the data was public | Holiday-aware 7 business days, with trading from the next session |
| 8 | Shares outstanding taken by period end (not filing date), the earliest *future* value when nothing earlier existed, or a vendor's current share count | Look-ahead in the screen | `filed_date <= settlement`, with no backfill |
| 9 | Two nested gap thresholds presented as independent confirmation | Evidence double-counted | One threshold, fixed in advance |
| 10 | Overlapping publication windows could produce two rows for one ticker and day | Double-counted trades | De-duplicated |
| 11 | Price data only for names the vendor still serves | Survivorship bias, which helps a LONG small-cap rule | Coverage is reported. Not fixable without delisted data |

The original code had no same-bar look-ahead in the ATR regime feature (it was shifted by one session), and the cooling-off already used only prior losses. Both were checked and are fine.

## What the original research concluded, and what I conclude

- **SHORT side: no edge.** Every exit rule, filter, ranking and control was negative out of sample, with one exception: three filters stacked together were near break-even on about 20 test trades before costs, too few to tell apart from noise. More conditioning only shrank the sample. This is a clear negative result.
- **LONG side: unvalidated.** "Top 5 by most negative pre-publication drift" was where the original research ended. It was positive in the reused test window, but so were several other top-5 rankings. Given issues 1 to 5, it is a candidate, not a finding. The right next step is this pipeline on fresh data (after the original sample ended), with costs and the random control.

## Checks in CI

- The publication date respects exchange holidays, and the first trade comes after it.
- Every candidate trades after publication, and its ranking features end before the trade date.
- LONG entry is on bar 16 (09:46) and SHORT entry is on bar 31 (10:01).
- Costs are charged on every trade.
- No duplicate (day, ticker, side).
- Shares are point-in-time, with no backfill.
- Exit rules: target, close, stop, and stop-first when both fall in one bar.
- Cooling-off only blocks later sessions.
- Top-N is respected and picks the lowest drift.
- Each walk-forward step trains only on earlier sessions.
- The planted effect is recovered out of sample. On pure noise, out-of-sample and hold-out average trades are negative, and the naive best-of-34 has a deflated Sharpe p below 0.5.
- The Parquet bar format round-trips against the synthetic source.

## Dropped exploratory work

The original repo had about 40 scripts. I kept the core pipeline (screen, windows, gap rules, drift feature, ranking, cooling-off) and dropped:

- the trailing-ATR exit and the ATR-percentile regime gate (failed out of sample)
- the liquidity floor, tight stop-loss grid, wide stop, take-profit sensitivity and partial scale-out (failed or inconclusive)
- the drift-floor and publication-day filters
- the volume-confirmation filter
- the market-index regime gate and the sector cooling-off
- the short-volume-ratio overlay from a data vendor
- the news-overlay probe
- spreadsheet and memo builders
- data fetchers with a hard-coded vendor API key

The cooling-off control is kept as a grid option rather than as part of a fixed spec.

## Known limitations

- The synthetic effect is deliberately strong, so the mechanics are visible. The real effect, if it exists, is much weaker. The weak-signal run (`--signal-strength 0.6`) shows how easily a single hold-out pick goes wrong.
- Intraday fills at 1-minute bar opens ignore queue position and spread widening at the open.
- Borrow availability and fees for the SHORT side are not modelled. The SHORT side loses money even without them.
