"""tradingbot.data.preprocess
--------------------------------
*DataProcessor* – merges econ, stock, news‑sentiment, technical indicators and
calendar features into a 3‑D NumPy array ready for modelling.

*DataPreprocessor* – scales / transforms the 3‑D tensor for training or
real‑time inference.

Both classes use the package‑wide logger, so INFO/DEBUG messages flow to the
console and *logs/tradingbot.csv* automatically.
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import finnhub
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tradingbot import cfg, get_logger
from tradingbot.data.indicators import TechnicalIndicators
from tradingbot.data.news import NewsFetcher
from tradingbot.data.sentiment import SentimentAnalyzer
from tradingbot.data.time_features import TimeFeatureGenerator

# --------------------------------------------------------------------------- #
# DataProcessor – build the raw 3‑D tensor
# --------------------------------------------------------------------------- #

class DataProcessor:
    """Generate a (dates × tickers × features) NumPy array."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tickers: List[str],
        logger=None,
    ) -> None:
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.tickers = tickers
        self.logger = logger or get_logger(__name__)

        # external helpers
        self.finnhub_client = finnhub.Client(api_key=cfg.api.FINNHUB_KEY)
        self.sentiment = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher(
            self.finnhub_client,
            self.sentiment,
            self.start_date,
            self.end_date,
            logger=self.logger.getChild("news"),
        )
        self.tech = TechnicalIndicators(self.logger.getChild("indicators"))
        self.time = TimeFeatureGenerator(self.logger.getChild("time"))

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def process_data(
        self,
        economic: pd.DataFrame,
        stock: pd.DataFrame,
        news: pd.DataFrame,
        window_size: int = 30,
    ) -> np.ndarray:
        """Return 3‑D float32 array shaped (dates, tickers, features)."""
        self.logger.info("Starting data processing – window_size=%d", window_size)
        # ------------------------------------------------------------------ #
        # Re‑index and log shapes / head / tail
        # ------------------------------------------------------------------ #
        econ = economic.reset_index().set_index(["Date", "Ticker"]).sort_index()
        stk = stock.reset_index().set_index(["Date", "Ticker"]).sort_index()
        nws = news.reset_index().set_index(["Date", "Ticker"]).sort_index()

        for name, df in [("Economic", econ), ("Stock", stk), ("News", nws)]:
            self.logger.info("%s shape after reindex: %s", name, df.shape)
            self.logger.info("%s Head:\n%s", name, df.head().to_string())
            self.logger.info("%s Tail:\n%s", name, df.tail().to_string())

        date_range = pd.date_range(self.start_date, self.end_date, freq="D")

        # 1) Technical indicators
        self.logger.info("Technical Indicators – computing …")
        indicators = self.tech.compute_indicators_for_tickers(
            stk, date_range, collection_start_offset=window_size
        )
        self.logger.info("Technical Indicators – done")
        # log indicator structure
        keys_overview = {t: list(ind.keys()) for t, ind in indicators.items()}
        self.logger.debug("Indicator Structure: %s", keys_overview)

        # 2) Time features
        time_feats = self.time.generate_features(date_range, as_dataframe=False)
        self.logger.info("Calendar features generated – shape %s", time_feats.shape)

        base_features = 18  # econ+stock+news+tech
        total_features = base_features + time_feats.shape[1]
        tensor = np.zeros((len(date_range), len(self.tickers), total_features), dtype=np.float32)

        # populate tensor
        for d_idx, date in enumerate(date_range):
            for t_idx, tkr in enumerate(self.tickers):
                try:
                    ei = econ.loc[(date, tkr)]
                    tensor[d_idx, t_idx, 0:4] = ei[[
                        "GDP_Growth",
                        "Unemployment_Rate",
                        "Inflation_Rate",
                        "VIX",
                    ]]
                except KeyError:
                    pass  # leave zeros

                try:
                    si = stk.loc[(date, tkr)]
                    tensor[d_idx, t_idx, 4:9] = si[[
                        "Open", "High", "Low", "Close", "Volume"]
                    ]
                except KeyError:
                    pass

                # news sentiment
                if (date, tkr) in nws.index:
                    score = nws.loc[(date, tkr)]["sentiment_score"]
                    if isinstance(score, pd.Series):
                        score = score.mean()
                    tensor[d_idx, t_idx, 9] = np.float32(score)

                # technical indicators (after warm‑up window)
                if d_idx >= window_size:
                    ind = indicators[tkr]
                    off = d_idx - window_size
                    tensor[d_idx, t_idx, 10:18] = [
                        ind["macd"][off],
                        ind["macd_signal"][off],
                        ind["rsi"][off],
                        ind["upper_band"][off],
                        ind["lower_band"][off],
                        ind["stoch_osc"][off],
                        ind["obv"][off],
                        ind["atr"][off],
                    ]

                # calendar
                tensor[d_idx, t_idx, base_features:] = time_feats[d_idx]

        self.logger.info("Process Data – complete. Tensor shape: %s", tensor.shape)
        # preview a small slice for debugging
        self.logger.debug("Tensor preview (first 2 tickers, last 3 days):\n%s",
                          tensor[-3:, :2, :])
        return tensor


# --------------------------------------------------------------------------- #
# DataPreprocessor – scaling utilities 
# --------------------------------------------------------------------------- #
class DataPreprocessor:
    def __init__(self):
        # Dictionaries to store scalers (fitted parameters)
        self.feature_scalers = {}  # For global/economic features
        self.ticker_scalers = {}   # For per-ticker features

        # Define feature indices for readability
        self.indices = {
            # Economic Indicators
            'gdp': 0,
            'unemployment': 1,
            'inflation': 2,
            'vix': 3,

            # Stock Data
            'open': 4,
            'high': 5,
            'low': 6,
            'close': 7,
            'volume': 8,

            # News Sentiment
            'sentiment': 9,

            # Technical Indicators
            'macd': 10,
            'macd_signal': 11,
            'rsi': 12,
            'upper_band': 13,
            'lower_band': 14,
            'stoch_osc': 15,
            'obv': 16,
            'atr': 17,

            # Time Features
            'day_of_week': 18,
            'is_weekend': 19,
            'is_month_start': 20,
            'is_month_end': 21,
            'is_quarter_start': 22,
            'is_quarter_end': 23,
            'is_year_start': 24,
            'is_year_end': 25,
            'is_holiday': 26
        }

    def fit_transform(self, tensor):
        """
        1) Fit all scalers on 'tensor' (historical data).
        2) Transform 'tensor' with those newly fitted scalers.
        3) Store fitted scalers in self.feature_scalers and self.ticker_scalers.
        
        Returns:
            np.ndarray: Scaled data (same shape as 'tensor').
        """
        tensor = np.nan_to_num(tensor)
        num_dates, num_tickers, _ = tensor.shape

        # Make a copy so we don’t mutate original input
        data_processed = np.copy(tensor)

        # ----------------------
        # Global / Economic features
        # ----------------------
        # GDP Growth (StandardScaler)
        gdp_vals = data_processed[:, :, self.indices['gdp']].reshape(-1, 1)
        scaler_gdp = StandardScaler()
        gdp_scaled = scaler_gdp.fit_transform(gdp_vals)
        data_processed[:, :, self.indices['gdp']] = gdp_scaled.reshape(num_dates, num_tickers)
        self.feature_scalers['gdp'] = scaler_gdp

        # Unemployment (MinMaxScaler)
        unemp_vals = data_processed[:, :, self.indices['unemployment']].reshape(-1, 1)
        scaler_unemp = MinMaxScaler()
        unemp_scaled = scaler_unemp.fit_transform(unemp_vals)
        data_processed[:, :, self.indices['unemployment']] = unemp_scaled.reshape(num_dates, num_tickers)
        self.feature_scalers['unemployment'] = scaler_unemp

        # Inflation (StandardScaler)
        infl_vals = data_processed[:, :, self.indices['inflation']].reshape(-1, 1)
        scaler_infl = StandardScaler()
        infl_scaled = scaler_infl.fit_transform(infl_vals)
        data_processed[:, :, self.indices['inflation']] = infl_scaled.reshape(num_dates, num_tickers)
        self.feature_scalers['inflation'] = scaler_infl

        # VIX (MinMaxScaler)
        vix_vals = data_processed[:, :, self.indices['vix']].reshape(-1, 1)
        scaler_vix = MinMaxScaler()
        vix_scaled = scaler_vix.fit_transform(vix_vals)
        data_processed[:, :, self.indices['vix']] = vix_scaled.reshape(num_dates, num_tickers)
        self.feature_scalers['vix'] = scaler_vix

        # ----------------------
        # News sentiment: map from [-1,1] to [0,1] (no scaler needed)
        # ----------------------
        sentiment_vals = data_processed[:, :, self.indices['sentiment']]
        data_processed[:, :, self.indices['sentiment']] = (sentiment_vals + 1) / 2

        # ----------------------
        # Time features
        # ----------------------
        # day_of_week (divide by 6), others are binary
        data_processed[:, :, self.indices['day_of_week']] /= 6.0

        # ----------------------
        # Ticker-specific features
        # ----------------------
        for t_idx in range(num_tickers):
            self.ticker_scalers[t_idx] = {}

            # Price features (MinMaxScaler)
            for f_name in ['open', 'high', 'low', 'close']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_vals = scaler.fit_transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()
                self.ticker_scalers[t_idx][f_idx] = scaler

            # Volume (log + MinMax)
            vol_idx = self.indices['volume']
            vol_vals = data_processed[:, t_idx, vol_idx]
            vol_log = np.log1p(vol_vals)  # log(1 + volume)
            scaler_vol = MinMaxScaler()
            vol_scaled = scaler_vol.fit_transform(vol_log.reshape(-1, 1))
            data_processed[:, t_idx, vol_idx] = vol_scaled.flatten()
            self.ticker_scalers[t_idx][vol_idx] = scaler_vol

            # MACD & MACD_signal (StandardScaler)
            for f_name in ['macd', 'macd_signal']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = StandardScaler()
                scaled_vals = scaler.fit_transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()
                self.ticker_scalers[t_idx][f_idx] = scaler

            # RSI (divide by 100)
            rsi_idx = self.indices['rsi']
            data_processed[:, t_idx, rsi_idx] /= 100.0

            # Bollinger Bands (MinMaxScaler)
            for f_name in ['upper_band', 'lower_band']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_vals = scaler.fit_transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()
                self.ticker_scalers[t_idx][f_idx] = scaler

            # Stochastic Oscillator (divide by 100)
            stoch_idx = self.indices['stoch_osc']
            data_processed[:, t_idx, stoch_idx] /= 100.0

            # OBV (sign * log1p(abs(x)), then StandardScaler)
            obv_idx = self.indices['obv']
            obv_vals = data_processed[:, t_idx, obv_idx]
            obv_sign = np.sign(obv_vals)
            obv_log = obv_sign * np.log1p(np.abs(obv_vals))
            scaler_obv = StandardScaler()
            obv_scaled = scaler_obv.fit_transform(obv_log.reshape(-1, 1))
            data_processed[:, t_idx, obv_idx] = obv_scaled.flatten()
            self.ticker_scalers[t_idx][obv_idx] = scaler_obv

            # ATR (MinMaxScaler)
            atr_idx = self.indices['atr']
            atr_vals = data_processed[:, t_idx, atr_idx].reshape(-1, 1)
            scaler_atr = MinMaxScaler()
            atr_scaled = scaler_atr.fit_transform(atr_vals)
            data_processed[:, t_idx, atr_idx] = atr_scaled.flatten()
            self.ticker_scalers[t_idx][atr_idx] = scaler_atr

        return data_processed

    def transform(self, tensor):
        """
        Apply *already-fitted* scalers to new data (e.g., real-time).
        No .fit(...) calls here—strictly .transform(...).
        
        Returns:
            np.ndarray: scaled data (same shape as 'tensor').
        """
        tensor = np.nan_to_num(tensor)
        num_dates, num_tickers, _ = tensor.shape
        data_processed = np.copy(tensor)

        # ----------------------
        # Global / Economic features
        # ----------------------
        gdp_vals = data_processed[:, :, self.indices['gdp']].reshape(-1, 1)
        scaler_gdp = self.feature_scalers['gdp']
        gdp_scaled = scaler_gdp.transform(gdp_vals)
        data_processed[:, :, self.indices['gdp']] = gdp_scaled.reshape(num_dates, num_tickers)

        unemp_vals = data_processed[:, :, self.indices['unemployment']].reshape(-1, 1)
        scaler_unemp = self.feature_scalers['unemployment']
        unemp_scaled = scaler_unemp.transform(unemp_vals)
        data_processed[:, :, self.indices['unemployment']] = unemp_scaled.reshape(num_dates, num_tickers)

        infl_vals = data_processed[:, :, self.indices['inflation']].reshape(-1, 1)
        scaler_infl = self.feature_scalers['inflation']
        infl_scaled = scaler_infl.transform(infl_vals)
        data_processed[:, :, self.indices['inflation']] = infl_scaled.reshape(num_dates, num_tickers)

        vix_vals = data_processed[:, :, self.indices['vix']].reshape(-1, 1)
        scaler_vix = self.feature_scalers['vix']
        vix_scaled = scaler_vix.transform(vix_vals)
        data_processed[:, :, self.indices['vix']] = vix_scaled.reshape(num_dates, num_tickers)

        # ----------------------
        # News sentiment: same mapping [-1,1] to [0,1]
        #   If real-time data is already in [-1,1], do the same shift.
        #   Or if you want to skip, adjust logic accordingly.
        # ----------------------
        sentiment_vals = data_processed[:, :, self.indices['sentiment']]
        data_processed[:, :, self.indices['sentiment']] = (sentiment_vals + 1) / 2

        # ----------------------
        # Time features
        # day_of_week, etc.
        # (No re-fitting needed, same logic as training)
        # ----------------------
        data_processed[:, :, self.indices['day_of_week']] /= 6.0

        # ----------------------
        # Ticker-specific features
        # ----------------------
        for t_idx in range(num_tickers):
            # Price features
            for f_name in ['open', 'high', 'low', 'close']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = self.ticker_scalers[t_idx][f_idx]
                scaled_vals = scaler.transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()

            # Volume
            vol_idx = self.indices['volume']
            vol_vals = data_processed[:, t_idx, vol_idx]
            vol_log = np.log1p(vol_vals)
            scaler_vol = self.ticker_scalers[t_idx][vol_idx]
            vol_scaled = scaler_vol.transform(vol_log.reshape(-1, 1))
            data_processed[:, t_idx, vol_idx] = vol_scaled.flatten()

            # MACD & MACD_signal
            for f_name in ['macd', 'macd_signal']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = self.ticker_scalers[t_idx][f_idx]
                scaled_vals = scaler.transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()

            # RSI
            rsi_idx = self.indices['rsi']
            data_processed[:, t_idx, rsi_idx] /= 100.0

            # Bollinger Bands
            for f_name in ['upper_band', 'lower_band']:
                f_idx = self.indices[f_name]
                vals = data_processed[:, t_idx, f_idx].reshape(-1, 1)
                scaler = self.ticker_scalers[t_idx][f_idx]
                scaled_vals = scaler.transform(vals)
                data_processed[:, t_idx, f_idx] = scaled_vals.flatten()

            # Stochastic Oscillator
            stoch_idx = self.indices['stoch_osc']
            data_processed[:, t_idx, stoch_idx] /= 100.0

            # OBV
            obv_idx = self.indices['obv']
            obv_vals = data_processed[:, t_idx, obv_idx]
            obv_sign = np.sign(obv_vals)
            obv_log = obv_sign * np.log1p(np.abs(obv_vals))
            scaler_obv = self.ticker_scalers[t_idx][obv_idx]
            obv_scaled = scaler_obv.transform(obv_log.reshape(-1, 1))
            data_processed[:, t_idx, obv_idx] = obv_scaled.flatten()

            # ATR
            atr_idx = self.indices['atr']
            atr_vals = data_processed[:, t_idx, atr_idx].reshape(-1, 1)
            scaler_atr = self.ticker_scalers[t_idx][atr_idx]
            atr_scaled = scaler_atr.transform(atr_vals)
            data_processed[:, t_idx, atr_idx] = atr_scaled.flatten()

        return data_processed

    def save_scalers(self, filepath):
        """
        Pickle the fitted scalers so you can load them later in production.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_scalers': self.feature_scalers,
                'ticker_scalers': self.ticker_scalers
            }, f)

    def load_scalers(self, filepath):
        """
        Load the scalers into this object (for real-time transform).
        """
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        self.feature_scalers = loaded['feature_scalers']
        self.ticker_scalers = loaded['ticker_scalers']