"""tradingbot.data.market
------------------------
High‑level data collector that assembles:
1. **Stock OHLCV** from *yfinance* (daily, forward‑filled).
2. **Economic indicators** (GDP, UNRATE, CPI, VIX) from FRED.
3. **Headline news** from Finnhub scored by our TensorFlow `SentimentAnalyzer`.

Every important step logs a human‑readable message — shapes, heads/tails, merge
status — through the package logger so you see it both on the console **and**
in `logs/tradingbot.csv`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import finnhub
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from tradingbot import cfg, get_logger
from tradingbot.data.indicators import TechnicalIndicators
from tradingbot.data.news import NewsFetcher
from tradingbot.data.sentiment import SentimentAnalyzer

logger = get_logger(__name__)


class MarketDataCollector:  # pylint: disable=too-many-instance-attributes
    """Download + merge stock, econ, news/sentiment for a list of tickers."""

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        tickers: List[str],
    ) -> None:
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        self.tickers = tickers

        self.fh = finnhub.Client(api_key=cfg.api.FINNHUB_KEY)
        self.sent = SentimentAnalyzer()
        self.news = NewsFetcher(self.fh, self.sent, self.start, self.end, logger)
        self.tech = TechnicalIndicators(logger)

        self._cache: Dict[str, pd.DataFrame] = {}
        self.actual_end: pd.Timestamp | None = None

        logger.info("MarketDataCollector initialised for %d tickers", len(tickers))

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def fetch_data(
        self, tickers: List[str] | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """Return econ, news, stock, actual_end_date."""
        tks = tickers or self.tickers
        stock, self.actual_end = self._fetch_stock(tks)
        econ = self._fetch_econ()
        news = self.news.fetch_news(tks)
        return econ, news, stock, self.actual_end

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _fetch_econ(self) -> pd.DataFrame:
        logger.info("Fetching economic indicators…")
        gdp  = pdr.get_data_fred("GDP",       self.start, self.end)
        unrt = pdr.get_data_fred("UNRATE",    self.start, self.end)
        cpi  = pdr.get_data_fred("CPIAUCSL",  self.start, self.end)
        vix  = yf.download("^VIX", start=self.start, end=self.end, auto_adjust=True)["Close"]
        vix.name = "^VIX"  # so rename below sees it

        # ---- 2) concatenate & rename columns ---------------------------- #
        econ = (
            pd.concat([gdp, unrt, cpi, vix], axis=1)
              .rename(columns={"GDP": "GDP_Growth",
                               "UNRATE": "Unemployment_Rate",
                               "CPIAUCSL": "Inflation_Rate"})
        )

        # ---- 3) Up‑sample to daily frequency then expand to **full** span #
        econ = econ.asfreq("D")  # just to duplicate existing dates daily

        full_idx = pd.date_range(self.start, self.end, freq="D")
        econ = econ.reindex(full_idx).ffill()
        econ = econ.reindex(full_idx).bfill()

        econ = econ.rename_axis("Date").reset_index()

        logger.debug("Econ head:\n%s", econ.head())
        logger.debug("Econ tail:\n%s", econ.tail())

        # ---- 4) repeat rows for each ticker to make (Date × Ticker) idx --#
        econ = (
            pd.concat([econ.assign(Ticker=t) for t in self.tickers])
              .set_index(["Date", "Ticker"])
              .sort_index()
        )

        logger.info("Economic indicators ready – shape %s", econ.shape)
        return econ

    # ------------------------------------------------------------------ #
    def _fetch_stock(self, tks: List[str]) -> Tuple[pd.DataFrame, pd.Timestamp]:
        logger.info("Downloading OHLCV for %d tickers", len(tks))
        frames: List[pd.DataFrame] = []
        end_seen: pd.Timestamp | None = None

        for t in tks:
            if t in self._cache:
                frames.append(self._cache[t])
                continue

            df = yf.download(t, start=self.start, end=self.end, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df["Ticker"] = t
            df = df.reset_index().set_index(["Date", "Ticker"])
            logger.debug("Head for %s:\n%s", t, df.head())
            logger.debug("Shape for %s: %s", t, df.shape)

            self._cache[t] = df
            frames.append(df)
            logger.info("Downloaded %s – %d rows", t, len(df))

            end_seen = (
                df.index.get_level_values(0).max()
                if end_seen is None
                else min(end_seen, df.index.get_level_values(0).max())
            )

        all_df = pd.concat(frames).sort_index()

        # Reindex per ticker to daily calendar then concat again
        full_dates = pd.date_range(self.start, self.end, freq="D")
        filled: List[pd.DataFrame] = []
        for tkr, grp in all_df.groupby(level="Ticker"):
            tmp = (
                grp.droplevel("Ticker")
                .reindex(full_dates, method="ffill")
                .bfill()
                .assign(Ticker=tkr)
                .rename_axis("Date")
                .reset_index()
                .set_index(["Date", "Ticker"])
            )
            filled.append(tmp)

        combo = pd.concat(filled).sort_index()

        # foward fill OHLC where missing; zero‑fill volume
        for col in ["Open", "High", "Low"]:
            combo[col] = combo[col].fillna(combo["Close"])
        combo["Volume"] = combo["Volume"].fillna(0)

        logger.info("Combined OHLCV ready – shape %s", combo.shape)
        logger.debug("OHLCV preview:\n%s", combo.head(8))

        return combo, end_seen if end_seen is not None else self.end
