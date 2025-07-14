"""tradingbot.data.time_features
--------------------------------
Generate classic *calendar* features (day‑of‑week, month‑end, holidays, …)
that are often useful for time‑series models.

Usage
-----
>>> gen = TimeFeatureGenerator()
>>> feats = gen.generate_features(pd.date_range("2025-04-24","2025-05-24"))
>>> feats.head()
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

from tradingbot import get_logger

log = get_logger(__name__)


class TimeFeatureGenerator:
    """Create binary / categorical calendar features for any date array."""

    def __init__(self, logger=None):
        self.cal = USFederalHolidayCalendar()
        self.log = logger or get_logger(__name__)
        log.debug("US Federal Holiday calendar initialised")

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def generate_features(self, dates, as_dataframe: bool = False):
        """Return calendar features for *dates*.

        Parameters
        ----------
        dates : array‑like
            Any sequence that ``pd.to_datetime`` understands.
        as_dataframe : bool, default False
            If *True* returns a ``pd.DataFrame`` indexed by ``dates``; otherwise
            returns a NumPy array (shape = ``len(dates) × 12``).
        """
        # Normalise to DatetimeIndex
        dates = pd.to_datetime(dates)
        self.log.debug("Generating time features for %d dates", len(dates))

        df = pd.DataFrame(index=dates)

        df["day_of_week"]      = df.index.dayofweek  # 0–6
        df["is_weekend"]      = (df["day_of_week"] >= 5).astype(np.int8)

        df["is_month_start"]  = df.index.is_month_start.astype(np.int8)
        df["is_month_end"]    = df.index.is_month_end.astype(np.int8)

        df["is_quarter_start"] = df.index.is_quarter_start.astype(np.int8)
        df["is_quarter_end"]   = df.index.is_quarter_end.astype(np.int8)

        df["is_year_start"]   = df.index.is_year_start.astype(np.int8)
        df["is_year_end"]     = df.index.is_year_end.astype(np.int8)

        # Holidays
        holidays = self.cal.holidays(start=dates.min(), end=dates.max())
        df["is_holiday"]      = df.index.isin(holidays).astype(np.int8)

        log.debug("Time feature matrix shape: %s", df.shape)

        if as_dataframe:
            return df
        else:
            return df.values.astype(np.float32)
