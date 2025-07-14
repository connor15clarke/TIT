"""tradingbot.data.news
----------------------
Fetch company news from Finnhub, score headlines with a SentimentAnalyzer, and
return a daily sentiment time‑series per ticker.

✓  Uses the shared logging setup (console + CSV).
✓  Accepts a pre‑created ``SentimentAnalyzer`` and Finnhub client.
"""
from __future__ import annotations

import time
from typing import List

import finnhub
import pandas as pd

from tradingbot import get_logger

class NewsFetcher:
    """Download headlines for *tickers* between *start_date* and *end_date* and
    aggregate them into daily sentiment scores.
    """

    def __init__(
        self,
        finnhub_client: finnhub.Client,
        sentiment_analyzer,  # expects .score(List[str]) -> List[float]
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        logger=None,
    ) -> None:
        self.finnhub_client = finnhub_client
        self.sentiment_analyzer = sentiment_analyzer
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.logger = logger or get_logger(__name__)
        self.logger.info("NewsFetcher initialised")

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    @staticmethod    
    def _map_hf_label(label: str, prob: float) -> float:
        """
        Handle both schemes returned by 🤗 models:
        • “positive / neutral / negative”
        • “LABEL_2 / LABEL_1 / LABEL_0”  (cardiffnlp/twitter-roberta…)
        """
        l = label.lower()
        if l in {"positive", "label_2"}:
            return +1.0 * prob
        if l in {"neutral",  "label_1"}:
            return  0.0      # keep prob for completeness, but neutral ⇒ 0
        if l in {"negative", "label_0"}:
            return -1.0 * prob
        raise ValueError(f"Unknown sentiment label: {label}")

    def fetch_news(self, tickers: List[str], batch_size: int = 10,
                   max_retries: int = 3) -> pd.DataFrame:

        self.logger.info("Fetching news for %d tickers (%s → %s)",
                         len(tickers),
                         self.start_date.strftime("%Y-%m-%d"),
                         self.end_date.strftime("%Y-%m-%d"))

        out_frames: list[pd.DataFrame] = []
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

        for batch in batches:
            retries = 0
            while retries < max_retries:
                try:
                    for tkr in batch:
                        arts = self.finnhub_client.company_news(
                            tkr,
                            _from=self.start_date.strftime("%Y-%m-%d"),
                            to=self.end_date.strftime("%Y-%m-%d"),
                        )
                        if not arts:
                            self.logger.debug("%s: 0 articles returned", tkr)
                            continue

                        df = (pd.DataFrame(arts)[["headline", "datetime"]]
                              .dropna()
                              .assign(datetime=lambda d: pd.to_datetime(d["datetime"], unit="s"),
                                      Date=lambda d: d["datetime"].dt.normalize(),
                                      Ticker=tkr))

                        # ── sentiment inference ───────────────────────────
                        raw = self.sentiment_analyzer.pipe(df["headline"].tolist())
                        df["sentiment_score"] = [self._map_hf_label(o["label"], o["score"])
                                                 for o in raw]
                        # ── daily aggregation ─────────────────────────────
                        daily = (df.groupby(["Date", "Ticker"], as_index=False)
                                   .agg(sentiment_score=("sentiment_score", "mean")))
                        out_frames.append(daily)
                        self.logger.debug("%s: %d headlines → %d daily rows",
                                          tkr, len(df), len(daily))
                    break           # whole batch succeeded
                except finnhub.FinnhubAPIException as exc:
                    # (rate-limit handling unchanged) …
                    ...
                except Exception as exc:          # pylint: disable=broad-except
                    self.logger.error("Unexpected error for %s: %s", batch, exc)
                    break

        if not out_frames:
            self.logger.warning("No news returned by Finnhub – emitting empty DataFrame")
            return (pd.DataFrame(columns=["Date", "Ticker", "sentiment_score"])
                    .set_index(["Date", "Ticker"]))

        news = (pd.concat(out_frames, ignore_index=True)
                  .set_index(["Date", "Ticker"])
                  .sort_index())

        # fill every (date, ticker) gap with 0
        full_dates = pd.date_range(self.start_date, self.end_date, freq="D")
        full_idx   = pd.MultiIndex.from_product([full_dates, tickers],
                                                names=["Date", "Ticker"])
        news = news.reindex(full_idx).fillna(0.0)

        self.logger.info("News sentiment ready – shape %s", news.shape)
        return news
