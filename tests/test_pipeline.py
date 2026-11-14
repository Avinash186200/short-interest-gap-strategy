import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sigap.calendar import first_tradable_session, publication_date, trading_sessions  # noqa: E402
from sigap.screen import screen, shares_asof  # noqa: E402
from sigap.selection import Config, run_selection  # noqa: E402
from sigap.sources import ParquetBarSource  # noqa: E402
from sigap.synthetic import SyntheticMarket, write_dataset  # noqa: E402
from sigap.trades import LONG_SIGNAL_BAR, SHORT_SIGNAL_BAR, build_candidates, simulate_exit  # noqa: E402
from sigap.validation import evaluate_side, run_grid, split_holdout, walk_forward  # noqa: E402


def _pipeline(signal_strength: float, n_tickers: int = 150, seed: int = 7):
    mkt = SyntheticMarket(n_tickers=n_tickers, seed=seed, signal_strength=signal_strength)
    cands, cov = build_candidates(screen(mkt.finra, mkt.shares), mkt)
    return mkt, cands, cov


@pytest.fixture(scope="module")
def planted():
    return _pipeline(1.0)


@pytest.fixture(scope="module")
def noise():
    return _pipeline(0.0)


# ---------------------------------------------------------------- timing / leakage
def test_publication_date_is_holiday_aware():
    # settlement Fri 2024-01-12; Mon 2024-01-15 is an exchange holiday
    assert publication_date(dt.date(2024, 1, 12)) == dt.date(2024, 1, 24)
    sessions = [d.date() for d in trading_sessions("2024-01-02", "2024-02-29")]
    assert first_tradable_session(dt.date(2024, 1, 12), sessions) == dt.date(2024, 1, 25)


def test_trades_only_after_publication(planted):
    _, c, _ = planted
    assert (c["trade_date"] > c["publication_date"]).all()
    assert (c["pub_eve_date"] < c["trade_date"]).all()  # ranking features end before the trade


def test_entry_is_after_signal_bar(planted):
    _, c, _ = planted
    assert (c.loc[c["side"] == "LONG", "entry_bar"] == LONG_SIGNAL_BAR + 1).all()
    assert (c.loc[c["side"] == "SHORT", "entry_bar"] == SHORT_SIGNAL_BAR + 1).all()
    assert (c["exit_bar"] >= c["entry_bar"]).all()


def test_costs_are_charged(planted):
    _, c, _ = planted
    assert np.allclose(c["gross_ret"] - c["net_ret"], 2 * 15 / 1e4)


def test_no_duplicate_trades_from_overlapping_windows(planted):
    _, c, _ = planted
    assert not c.duplicated(["trade_date", "ticker", "side"]).any()


def test_shares_are_point_in_time():
    shares = pd.DataFrame({"symbol": ["A", "A"], "period_end": ["2024-03-31", "2024-06-30"],
                           "filed_date": ["2024-05-10", "2024-08-09"], "shares": [10e6, 20e6]})
    assert shares_asof(shares, "2024-04-30").empty                 # nothing filed yet: no backfill
    assert shares_asof(shares, "2024-07-15")["A"] == 10e6          # Q2 period ended but not filed
    assert shares_asof(shares, "2024-08-09")["A"] == 20e6


def test_screen_filters(planted):
    mkt, _, _ = planted
    scr = screen(mkt.finra, mkt.shares)
    assert not scr["marketClassCode"].isin({"OTC", "OTCE"}).any()
    assert (scr["si_pct_shares"] >= 10).all() and (scr["daysToCoverQuantity"] >= 5).all()


# ---------------------------------------------------------------- execution
def test_exit_rules():
    h = np.array([100.5, 104.0, 101.0])
    l = np.array([99.0, 99.5, 80.0])
    c = np.array([100.0, 101.0, 85.0])
    assert simulate_exit(h, l, c, 100.0, "LONG", 3.0) == (103.0, 1, "take_profit")
    assert simulate_exit(h, l, c, 100.0, "LONG", 5.0) == (85.0, 2, "close")
    px, i, why = simulate_exit(h, l, c, 100.0, "LONG", 5.0, stop_pct=18.0)
    assert (round(px, 6), i, why) == (82.0, 2, "stop")
    # stop and target inside one bar: assume the stop filled first
    assert simulate_exit(np.array([110.0]), np.array([80.0]), np.array([100.0]), 100.0, "LONG", 3.0, 10.0)[2] == "stop"


def _toy_cands():
    d = [dt.date(2024, 3, k) for k in (4, 5, 6, 11, 12)]
    rows = [(d[0], "AAA", -0.02), (d[0], "BBB", 0.01), (d[1], "AAA", 0.01), (d[2], "AAA", 0.01),
            (d[3], "AAA", 0.01), (d[4], "AAA", 0.01)]
    df = pd.DataFrame(rows, columns=["trade_date", "ticker", "net_ret"])
    return df.assign(side="LONG", price_drift_pct=0.0, volume_ratio=1.0)


def test_cooloff_skips_only_later_sessions():
    taken = run_selection(_toy_cands(), Config("LONG", "drift_low", None, 5))
    aaa = taken[taken["ticker"] == "AAA"]["trade_date"].tolist()
    # loss on 3/4 -> 3/5, 3/6 skipped (<=5 days); 3/11 is 7 days later
    assert aaa == [dt.date(2024, 3, 4), dt.date(2024, 3, 11), dt.date(2024, 3, 12)]


def test_top_n_respected(planted):
    _, c, _ = planted
    taken = run_selection(c, Config("LONG", "drift_low", 2, 0))
    assert taken.groupby("trade_date").size().max() <= 2
    # the picks are the two lowest-drift names of the day
    day = taken["trade_date"].value_counts().idxmax()
    pool = c[(c["side"] == "LONG") & (c["trade_date"] == day)].nsmallest(2, "price_drift_pct")
    assert set(pool["ticker"]) == set(taken[taken["trade_date"] == day]["ticker"])


def test_walk_forward_trains_only_on_the_past(planted):
    mkt, c, _ = planted
    runs = run_grid(c, "LONG", mkt.sessions, 5)
    research, holdout = split_holdout(mkt.sessions, 4)
    assert research[-1] < holdout[0]
    wf = walk_forward(runs, research)
    assert wf["steps"]
    for st in wf["steps"]:
        assert st["train_end"] < st["test_start"]


# ---------------------------------------------------------------- signal vs noise
def test_planted_signal_is_recovered_out_of_sample(planted):
    mkt, c, _ = planted
    res = evaluate_side(c, "LONG", mkt.sessions, n_random=20)
    assert res["walk_forward_oos"]["sharpe"] > 1.0
    assert res["holdout_pre_registered"]["sharpe"] > res["holdout_take_all"]["sharpe"]


def test_noise_shows_no_edge(noise):
    mkt, c, _ = noise
    res = evaluate_side(c, "LONG", mkt.sessions, n_random=20)
    assert res["walk_forward_oos"]["avg_net_ret_pct"] < 0
    assert res["holdout_pre_registered"]["avg_net_ret_pct"] < 0
    assert res["naive_in_sample"]["deflated_sharpe_prob"] < 0.5  # best-of-34 is not significant


# ---------------------------------------------------------------- file format
def test_parquet_source_matches_synthetic(tmp_path):
    mkt = SyntheticMarket(n_tickers=4, start="2024-01-02", end="2024-03-29", seed=3)
    write_dataset(mkt, tmp_path, tickers=["SYN000", "SYN001"])
    src = ParquetBarSource(tmp_path / "bars")
    a, b = src.daily("SYN001"), mkt.daily("SYN001")
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert np.allclose(a["open"].to_numpy(), b["open"].to_numpy())
    s = mkt.sessions[10]
    assert np.allclose(src.session_bars("SYN000", s)["h"], mkt.session_bars("SYN000", s)["h"])
