"""Trading calendar and FINRA publication timing.

FINRA short interest is reported for a *settlement date* and made public on
the 7th business day after it. The original research approximated this as
"settlement + 8 weekdays", ignoring exchange holidays, and allowed a trade on
the publication date itself. Because the release time of day is not
guaranteed to be before the open, the only safe assumption is that the data
can first be acted on in the session *after* the publication date.
"""
from __future__ import annotations

import bisect
import datetime as dt
from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

PUBLICATION_LAG_BUSINESS_DAYS = 7


class ExchangeHolidayCalendar(AbstractHolidayCalendar):
    """Approximate US equity-exchange holidays (full-day closures only)."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=8)
def _holidays(start_year: int, end_year: int) -> tuple[dt.date, ...]:
    hol = ExchangeHolidayCalendar().holidays(f"{start_year}-01-01", f"{end_year}-12-31")
    return tuple(d.date() for d in hol)


def business_day(start_year: int = 2015, end_year: int = 2035) -> pd.offsets.CustomBusinessDay:
    """A holiday-aware business-day offset."""
    return pd.offsets.CustomBusinessDay(holidays=list(_holidays(start_year, end_year)))


def trading_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """All exchange sessions between `start` and `end` inclusive."""
    return pd.date_range(start, end, freq=business_day())


def publication_date(settlement: dt.date, lag: int = PUBLICATION_LAG_BUSINESS_DAYS) -> dt.date:
    """FINRA publication date: `lag` exchange business days after settlement."""
    return (pd.Timestamp(settlement) + lag * business_day()).date()


def first_tradable_session(settlement: dt.date, sessions: list[dt.date] | pd.Index,
                           lag: int = PUBLICATION_LAG_BUSINESS_DAYS) -> dt.date | None:
    """First session strictly after the publication date (data assumed public after the close)."""
    pub = publication_date(settlement, lag)
    seq = list(sessions)
    i = bisect.bisect_right(seq, pub)
    return seq[i] if i < len(seq) else None
