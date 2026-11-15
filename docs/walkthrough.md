# Walkthrough: a first run

This is a guided first run. It assumes you know Python but not trading. Read the "Key concepts" list in the
[README](../README.md#key-concepts) first if a term is new. All numbers below come from running the code on the
synthetic market.

## 1. Set up and run the tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

You should see `14 passed`. The timing tests are the ones to read first: when the data becomes public, which
session is the first one you may trade, and which share count was on file at the time.

## 2. Run with the planted effect

```bash
python scripts/run_backtest.py
```

The first lines describe the input:

```
screened windows 3932, coverage {'windows': 3932, 'no_price_data': 73, 'gap_sessions': 7382, ...}
candidates: {'LONG': 3677, 'SHORT': 2578}
```

3,932 (settlement date, ticker) windows passed the screen. 73 had no price data; on real data that count is
where survivorship hides. 7,382 sessions gapped up by 2.5% or more, and the gap rules turned them into 3,677 LONG
and 2,578 SHORT candidate trades.

Then one block per side. Read the LONG block from the top down:

| Line | Sharpe | What it is |
|---|---|---|
| naive in-sample best | 4.66 | Best of 34 configs on the whole sample. A contrast only. Deflated Sharpe p = 0.80 |
| walk-forward OOS | 5.02 | Config re-chosen each quarter from earlier data, traded on the next quarter |
| hold-out, chosen | 3.56 | Config chosen on the research period, scored once on the last 4 months |
| beats 100% of random top-N draws | | The same top-N picked at random 200 times has a median Sharpe of -0.98 |
| hold-out, take all | -1.06 | Trade every candidate |
| hold-out, pre-reg | 2.05 | The rule the original research ended with, fixed in advance |

The planted effect is strong on purpose, so everything works here. The SHORT block has no planted effect and
loses money on every line.

## 3. Run with no effect

```bash
python scripts/run_backtest.py --signal-strength 0
```

The LONG naive best still shows a positive Sharpe (0.44). Its deflated Sharpe p is 0.01 and the walk-forward is
-0.69, so both reject it. Now look at the random control: the chosen hold-out config beats 76% of random draws
(random median -2.74), and random top-2 picks beat taking everything (-4.48). Neither means the ranking works. On
noise, trading fewer names just means paying fewer costs. That is the main lesson of this repo.

## 4. Run from files

```bash
python scripts/make_synthetic_data.py --n-tickers 20 --out data/synthetic
python scripts/run_backtest.py --data-dir data/synthetic
```

The first command writes 20 tickers and 375 sessions in the on-disk format. The second reads them back. Open
`data/synthetic` to see the file layout you would need for your own data. With only 20 tickers the results are
small samples: the LONG walk-forward has 21 trades and the SHORT walk-forward 11, so read them as a format check,
not a result.

## 5. Where to go next

- [methodology.md](methodology.md) has the full method and the bias audit.
- `src/sigap/calendar.py` shows the publication-date rule in about 75 lines.
- `src/sigap/validation.py` holds the walk-forward, hold-out and random control.
