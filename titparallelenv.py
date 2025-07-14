
# Standard library imports
import csv
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
# print("Attempting to disable XLA Auto JIT via TF_XLA_FLAGS using level -1.")
import signal
import sys
import logging
import traceback
import multiprocessing # For parallel training
# Configure logging
logging.basicConfig(filename='termination.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def handle_termination(signum, frame):
    # Log that a signal was received
    logging.info(f"Process received signal {signum}. Terminating gracefully.")
    # Optionally, print the current stack trace (if available)
    logging.info("Stack trace:\n" + traceback.format_stack(frame))
    sys.exit(0)

# Register handlers for SIGTERM and SIGINT
signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"     # skip oneDNN (fixes many double-free bugs)
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"     # silence noisy INFO logs 

import cpuinfo
info = cpuinfo.get_cpu_info()

import requests
import time
from datetime import datetime, timedelta
import random
import time
import pickle 


# Data handling
from numba import njit, prange, set_num_threads
set_num_threads(12)  # For your 12-core CPU
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt

# Finance API
import finnhub
from finnhub.exceptions import FinnhubAPIException
import yfinance as yf

# Machine Learning: Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
from collections import deque, defaultdict

# Set TensorFlow configuration BEFORE importing TensorFlow
os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '12'
os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '12'

# Deep Learning: TensorFlow/Keras
import tensorflow as tf 
print(tf.sysconfig.get_build_info()) 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.keras import Sequential, mixed_precision
import tensorflow_probability as tfp 
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, Input, Masking, Layer, Lambda, Add, Concatenate, Reshape, BatchNormalization, LayerNormalization, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2, Regularizer
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.utils import register_keras_serializable

import gymnasium as gym 
from gymnasium import spaces

# Transformers
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification

class bot_logger:
    def __init__(self):
        self.log = []

    def record(self, event, msg):
        self.log.append((event, msg))
    
    def output_log(self, print_log=True, save_to_file="data_processor_log.csv"):
        if print_log:
            for event, msg in self.log:
                print(f"{event}: {msg}")
        
        if save_to_file:
            with open(save_to_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Event', 'Message'])
                for event, msg in self.log:
                    csv_writer.writerow([event, msg])

class SentimentAnalyzer:
    def __init__(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)  
        # Initialize the pipeline with the model and tokenizer
        self.analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0)  
        
    def analyze(self, text):
        try:
            result = self.analyzer(text, top_k=1)
            return {'label': result[0]['label'], 'score': result[0]['score']} if result else {'label': 'NEUTRAL', 'score': 0.0}
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.0}


class NewsFetcher:
    def __init__(self, finnhub_client, sentiment_analyzer, start_date, end_date, bot_logger):
        self.finnhub_client = finnhub_client
        self.sentiment_analyzer = sentiment_analyzer
        self.start_date = start_date
        self.end_date = end_date
        self.logger = bot_logger
        self.log_file = "data_processor_log.csv"
        self._create_log_file()

    def _create_log_file(self):
        with open(self.log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Step', 'Message'])
    
    def _log(self, step, message, save_to_file=False):
        if save_to_file:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])
        else:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])

    def fetch_news(self, tickers, batch_size=6, max_retries=3):
        all_news_data = []
        ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        self._log("News Fetching", f"Starting news fetch for dates {self.start_date} to {self.end_date}")

        for batch in ticker_batches:
            retries = 0
            while retries < max_retries:
                try:
                    news_data = []
                    for ticker in batch:
                        news = self.finnhub_client.company_news(ticker, _from=self.start_date, to=self.end_date)
                        if news:
                            news_df = pd.DataFrame(news)
                            news_df = news_df[['headline', 'datetime']].dropna()
                            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
                            news_df['Date'] = pd.to_datetime(news_df['datetime'].dt.date)
                            news_df['Ticker'] = ticker

                            news_df['sentiment_results'] = news_df['headline'].apply(self.sentiment_analyzer.analyze)
                            news_df['sentiment_score'] = news_df['sentiment_results'].apply(lambda x: x['score'] if x['label'] == 'POSITIVE' else -x['score'])

                            daily_sentiment = news_df.groupby(['Date', 'Ticker']).agg(
                                sentiment_score=('sentiment_score', 'mean'),
                                sentiment_results=('sentiment_results', 'first')
                            ).reset_index()

                            news_data.append(daily_sentiment)
                            self._log("News Fetched", f"News fetched for ticker {ticker}: {daily_sentiment.head().to_string()}")
                    if news_data:
                        all_news_data.extend(news_data)
                    break
                except finnhub.FinnhubAPIException as e:
                    if e.status_code == 429:
                        retries += 1
                        self._log("API Limit", f"API limit reached, waiting 60 seconds before retrying... (Retry {retries}/{max_retries})")
                        time.sleep(60)
                    else:
                        self._log("News Fetch Error", f"Failed to fetch news for batch {batch} due to: {e}")
                        break
                except Exception as e:
                    self._log("Unexpected Error", f"Unexpected error for batch {batch}: {e}")
                    break
        
        all_news_data.append(daily_sentiment)
        self._log("News Fetched", f"News fetched for ticker {ticker}: {daily_sentiment.head().to_string()}")

        if all_news_data:
            news_data_df = pd.concat(all_news_data, ignore_index=True)

            if not news_data_df.empty:
                news_data_df = news_data_df.sort_values(['Date', 'Ticker'])  # Sort by date and ticker
                
                news_data_df = news_data_df.groupby(['Date', 'Ticker'], as_index=False).agg({
                    'sentiment_score': 'mean',
                    'sentiment_results': 'first'
                })
                news_data_df = news_data_df.set_index(['Date', 'Ticker'])

            if isinstance(news_data_df.index, pd.MultiIndex):
                news_data_df.index = news_data_df.index.set_levels(
                    news_data_df.index.levels[0].normalize(), level='Date'
                )
            else:
                # Convert to MultiIndex if not already
                news_data_df['Date'] = pd.to_datetime(news_data_df['Date']).dt.normalize()
                news_data_df = news_data_df.set_index(['Date', 'Ticker'])

            # Create a MultiIndex with all date and ticker combinations
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D', inclusive='left').normalize()
            ticker_index = pd.Index(tickers, name='Ticker')
            multi_index = pd.MultiIndex.from_product([date_range, ticker_index], names=['Date', 'Ticker'])

            # Reindex the news_data_df to include all date and ticker combinations
            news_data_df = news_data_df.reindex(multi_index).sort_index().fillna(0)
            self._log("News Data Compiled", f"All news data compiled into DataFrame: {news_data_df.tail(5).to_string()}")
            return news_data_df
        else:
            print("No News Data", "No news data available.")
            return pd.DataFrame(columns=['Date', 'Ticker', 'sentiment_score']).set_index(['Date', 'Ticker'])

class MarketDataCollector:
    def __init__(self, start_date, end_date, bot_logger, tickers):
        self.start_date = start_date
        self.end_date = end_date
        self.logger = bot_logger
        self.tickers = tickers
        self.finnhub_client = finnhub.Client(api_key="coavp0pr01qro9kpjlb0coavp0pr01qro9kpjlbg")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher(self.finnhub_client, self.sentiment_analyzer, start_date, end_date, self.logger)
        self.indicator_calc = TechnicalIndicators(bot_logger)
        self.log_file = "data_processor_log.csv"
        self._create_log_file()
        self.stock_data_cache = {}
        self.actual_end_date = None
    
    def _create_log_file(self):
        with open(self.log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Step', 'Message'])

    def _log(self, step, message):
        with open(self.log_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([step, message])
    
    def fetch_data(self, tickers):
        stock_data, actual_end_date = self.fetch_stock_data(tickers)
        economic_indicators = self.fetch_economic_indicators()
        news_data = self.news_fetcher.fetch_news(tickers)

        return economic_indicators, news_data, stock_data, actual_end_date
    
    def fetch_economic_indicators(self):
        gdp_growth = pdr.get_data_fred('GDP', start=self.start_date, end=self.end_date)
        unemployment_rate = pdr.get_data_fred('UNRATE', start=self.start_date, end=self.end_date)
        inflation_rate = pdr.get_data_fred('CPIAUCSL', start=self.start_date, end=self.end_date)
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date)['Close']

        # Convert indices to datetime
        gdp_growth.index = pd.to_datetime(gdp_growth.index)
        unemployment_rate.index = pd.to_datetime(unemployment_rate.index)
        inflation_rate.index = pd.to_datetime(inflation_rate.index)
        vix.index = pd.to_datetime(vix.index)

        # Merge data into a single DataFrame
        economic_data = pd.concat([gdp_growth, unemployment_rate, inflation_rate, vix], axis=1)
        economic_data.columns = ['GDP_Growth', 'Unemployment_Rate', 'Inflation_Rate', 'VIX']

        # Create a date range from start to end date
        economic_data = economic_data.reindex(pd.date_range(start=economic_data.index.min(), end=economic_data.index.max(), freq='D'))

        # Forward fill missing values
        economic_data = economic_data.ffill()

        # If there are still NaN values at the beginning, backfill them
        economic_data = economic_data.bfill()

        # Reset the index to include 'Date' as a column
        economic_data.reset_index(inplace=True)
        economic_data.rename(columns={'index': 'Date'}, inplace=True)

        # Ensure 'Date' column is in datetime format
        economic_data['Date'] = pd.to_datetime(economic_data['Date'])

        # Repeat the economic indicators for all tickers
        repeated_data = []
        for ticker in self.tickers:
            ticker_data = economic_data.copy()
            ticker_data['Ticker'] = ticker
            repeated_data.append(ticker_data)

        economic_data = pd.concat(repeated_data)

        # Ensure the economic indicators are columns
        economic_data = economic_data.set_index(['Date', 'Ticker']).sort_index()

        self._log("Economic indicators data fetched", economic_data.head().to_string())
        
        return economic_data
    
    def fetch_stock_data(self, tickers):
        if not self.stock_data_cache:
            all_data = []
            for ticker in tickers:
                if ticker not in self.stock_data_cache:
                    data = yf.download(ticker, start=self.start_date, end=self.end_date)
                    # If the columns are multi-indexed, flatten them by taking the second level
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.index = pd.to_datetime(data.index)
                    data['Ticker'] = ticker
                    data_with_index = data.reset_index().set_index(['Date', 'Ticker'])
                    # Add debug log for the shape after processing
                    self._log('Debug - Fetched Data', f'Ticker: {ticker}, Shape: {data_with_index.shape}\n{data_with_index.head().to_string()}')
                    self.stock_data_cache[ticker] = data_with_index
                    all_data.append(data_with_index)
                    self._log('Fetched', f'Data for {ticker}, shape: {data.shape}')

                    if self.actual_end_date is None or data.index[-1] < self.actual_end_date:
                        self.actual_end_date = data.index[-1]

            combined_stock_data = pd.concat(all_data)
            self._log('Concatenated', f'Concatenated all data, shape: {combined_stock_data.shape}')

            # Reindex to include all dates and tickers (including weekends and holidays)
            all_dates = pd.date_range(
                start=combined_stock_data.index.get_level_values('Date').min(),
                end=combined_stock_data.index.get_level_values('Date').max(),
                freq='D'
            )
            all_tickers = combined_stock_data.index.get_level_values('Ticker').unique()
            combined_index = pd.MultiIndex.from_product(
                [all_dates, all_tickers],
                names=['Date', 'Ticker']
            )
            combined_stock_data = combined_stock_data.reindex(combined_index)

            # Forward-fill and back-fill 'Close' per ticker
            combined_stock_data['Close'] = combined_stock_data.groupby('Ticker')['Close'].ffill().bfill()

            # Fill 'Open', 'High', 'Low'' where missing
            for col in ['Open', 'High', 'Low']:
                combined_stock_data[col] = combined_stock_data[col].fillna(combined_stock_data['Close'])

            # Set 'Volume' to 0 where missing
            combined_stock_data['Volume'] = combined_stock_data['Volume'].fillna(0)

            # Sort the data
            combined_stock_data = combined_stock_data.sort_index()

            self._log('Debug - Sorted Combined Stock Data', combined_stock_data.head(8).to_string())

            self.stock_data_cache['combined'] = combined_stock_data

        return self.stock_data_cache['combined'], self.actual_end_date

class TechnicalIndicators:
    def __init__(self, bot_logger):
        self.logger = bot_logger
        self.log_file = "data_processor_log.csv"
        self._create_log_file()
    
    def _create_log_file(self):
        with open(self.log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Step', 'Message'])
    
    def _log(self, step, message, save_to_file=False):
        if save_to_file:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])
        else:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])

    def compute_macd(self, prices, short_window=9, long_window=24, signal_window=6):
        short_ema = prices.ewm(span=short_window).mean()
        long_ema = prices.ewm(span=long_window).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_window).mean()
        return macd.iloc[-1], macd_signal.iloc[-1]

    def compute_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def compute_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        return upper_band.iloc[-1], lower_band.iloc[-1]

    def compute_stochastic_oscillator(self, prices, window=14, smooth_window=3):
        high = prices.rolling(window=window).max()
        low = prices.rolling(window=window).min()
        stoch_k = 100 * (prices - low) / (high - low)
        stoch_d = stoch_k.rolling(window=smooth_window).mean()
        return stoch_k.iloc[-1]

    def compute_obv(self, prices, volumes):
        obv = (np.sign(prices.diff()) * volumes).cumsum()
        return obv.iloc[-1]

    def compute_atr(self, data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr.iloc[-1]
    
    def compute_indicators_for_tickers(self, stock_data, date_range, window_size):
        indicators = {}
        self._log("Indicator Computation", f"Starting indicator computation for {len(stock_data.index.get_level_values('Ticker').unique())} tickers")
        # Log initial data shapes
        self._log("Initial Data Shapes", 
                f"\nFull stock data shape: {stock_data.shape}"
                f"\nDate range length: {len(date_range)}"
                f"\nWindow size: {window_size}")
        
        unique_tickers = stock_data.index.get_level_values('Ticker').unique()
        self._log("Tickers", f"Processing {len(unique_tickers)} tickers: {unique_tickers.tolist()}")
    
        for ticker in stock_data.index.get_level_values('Ticker').unique():
            self._log("Indicator Computation", f"Computing indicators for {ticker}")
            indicators[ticker] = {
                'macd': [], 'macd_signal': [], 'rsi': [],
                'upper_band': [], 'lower_band': [], 'stoch_osc': [],
                'obv': [], 'atr': []
            }
            
            ticker_data = stock_data.xs(ticker, level='Ticker')
            self._log("Ticker Data", 
                 f"\nTicker: {ticker}"
                 f"\nTicker data shape: {ticker_data.shape}"
                 f"\nColumns: {ticker_data.columns.tolist()}"
                 f"\nIndex range: {ticker_data.index.min()} to {ticker_data.index.max()}")
            
            if ticker_data.empty:
                self._log("Data Error", f"No data found for ticker {ticker}")
                continue

            self._log("Data Info", f"Ticker {ticker} data shape: {ticker_data.shape}")
            
            for i in range(window_size, len(date_range)):
                end_date = date_range[i]
                start_date = date_range[i - window_size]
                
                subset = ticker_data.loc[start_date:end_date]
                
                if subset.empty:
                    self._log("Data Error", f"No data in subset for {ticker} from {start_date} to {end_date}")
                    continue

                try:
                    macd, macd_signal = self.compute_macd(subset['Close'])
                    rsi = self.compute_rsi(subset['Close'])
                    upper_band, lower_band = self.compute_bollinger_bands(subset['Close'])
                    stoch_osc = self.compute_stochastic_oscillator(subset['Close'])
                    obv = self.compute_obv(subset['Close'], subset['Volume'])
                    atr = self.compute_atr(subset)

                    # Log indicator values periodically
                    if i % 1000 == 0:
                        self._log("Indicator Values", 
                                f"\nTicker: {ticker}, Window {i}"
                                f"\nMACD: {macd:.4f}"
                                f"\nMACD Signal: {macd_signal:.4f}"
                                f"\nRSI: {rsi:.4f}"
                                f"\nBollinger Bands: {upper_band:.4f}, {lower_band:.4f}"
                                f"\nStochastic: {stoch_osc:.4f}"
                                f"\nOBV: {obv:.4f}"
                                f"\nATR: {atr:.4f}")

                    indicators[ticker]['macd'].append(macd)
                    indicators[ticker]['macd_signal'].append(macd_signal)
                    indicators[ticker]['rsi'].append(rsi)
                    indicators[ticker]['upper_band'].append(upper_band)
                    indicators[ticker]['lower_band'].append(lower_band)
                    indicators[ticker]['stoch_osc'].append(stoch_osc)
                    indicators[ticker]['obv'].append(obv)
                    indicators[ticker]['atr'].append(atr)

                except Exception as e:
                    self._log("Computation Error", f"Error computing indicators for {ticker} from {start_date} to {end_date}: {str(e)}")

        # Final summary
        self._log("Final Summary", 
                f"\nProcessed {len(unique_tickers)} tickers"
                f"\nIndicator shape for each ticker:"
                f"\n{pd.DataFrame({ticker: {k: len(v) for k, v in ticker_data.items()} 
                                for ticker, ticker_data in indicators.items()}).to_string()}")
        
        return indicators

class TimeFeatureGenerator:
    def __init__(self):
        self.cal = USFederalHolidayCalendar()

    def generate_features(self, dates):
        # Convert dates to datetime if they're not already
        dates = pd.to_datetime(dates)

        # Create a DataFrame with the dates
        df = pd.DataFrame({'date': dates})

        # Add day of week (0 = Monday, 6 = Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek

        # Add is_weekend feature (0 = weekday, 1 = weekend)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Add is_month_start and is_month_end features
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

        # Add is_quarter_start and is_quarter_end features
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

        # Add is_year_start and is_year_end features
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)

        # Add is_holiday feature
        holidays = self.cal.holidays(start=dates.min(), end=dates.max())
        df['is_holiday'] = df['date'].isin(holidays).astype(int)

        return df.drop('date', axis=1).values

class DataProcessor:
    def __init__(self, start_date, end_date, bot_logger, tickers):
        self.start_date = start_date
        self.end_date = end_date
        self.logger = bot_logger
        self.finnhub_client = finnhub.Client(api_key="coavp0pr01qro9kpjlb0coavp0pr01qro9kpjlbg")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher(self.finnhub_client, self.sentiment_analyzer, start_date, end_date, self.logger)
        self.log_file = "data_processor_log.csv"
        self._create_log_file()
        self.tickers = tickers
        self.technical_indicators = TechnicalIndicators(self.logger)
        self.time_feature_generator = TimeFeatureGenerator()  
    
    def _create_log_file(self):
        with open(self.log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Step', 'Message'])
    
    def _log(self, step, message, save_to_file=False):
        if save_to_file:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])
        else:
            with open(self.log_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, message])
    
    def process_data(self, economic_indicators, stock_data, news_data, window_size=30):
        self._log("Process Data", "Starting data processing")
        
        # Calculate technical indicators
        self._log("Technical Indicators", "Starting technical indicator computation")
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        indicators = self.technical_indicators.compute_indicators_for_tickers(stock_data, date_range, window_size)
        self._log("Technical Indicators", "Finished technical indicator computation")

        # Combine all data
        processed_data = self._combine_data(economic_indicators, stock_data, news_data, indicators, window_size)
        
        self._log("Process Data", f"Processing completed. Data shape: {processed_data.shape}")
        return processed_data
    
    
    def _combine_data(self, economic_indicators, stock_data, news_data, indicators, window_size):
        # Reset index and set it again with the desired levels
        economic_indicators = economic_indicators.reset_index().set_index(['Date', 'Ticker'])
        stock_data = stock_data.reset_index().set_index(['Date', 'Ticker'])
        news_data = news_data.reset_index().set_index(['Date', 'Ticker'])

        # Print the shapes of the DataFrames after reindexing
        self._log("Economic Indicators Shape after Reindexing", str(economic_indicators.shape), save_to_file=True)
        self._log("Stock Data Shape after Reindexing", str(stock_data.shape), save_to_file=True)
        self._log("News Data Shape after Reindexing", str(news_data.shape), save_to_file=True)

        # Print the head and tail of each DataFrame to check for any missing or inconsistent data
        self._log("Economic Indicators Head", str(economic_indicators.head()), save_to_file=True)
        self._log("Economic Indicators Tail", str(economic_indicators.tail()), save_to_file=True)
        self._log("Stock Data Head", str(stock_data.head()), save_to_file=True)
        self._log("Stock Data Tail", str(stock_data.tail()), save_to_file=True)
        self._log("News Data Head", str(news_data.head()), save_to_file=True)
        self._log("News Data Tail", str(news_data.tail()), save_to_file=True)

        # Get the start and end dates of the simulation duration
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D', inclusive='left').normalize()
        dates = len(date_range)
        time_features = self.time_feature_generator.generate_features(date_range)

        # Calculate the total number of features
        base_features = 18  # Original number of features
        time_feature_count = time_features.shape[1]
        total_features = base_features + time_feature_count

        # Initialize the 3D array with the new total number of features
        tickers = self.tickers
        data_3d = np.zeros((dates, len(tickers), total_features), dtype=np.float32)

        # Log the shape of stock_data before computing indicators
        self._log("Data Shape", f"Stock data shape before computing indicators: {stock_data.shape}")

        # Log the structure of the computed indicators
        self._log("Indicator Structure", f"Computed indicators structure: {indicators.keys()}")
        for ticker in indicators:
            self._log("Indicator Structure", f"Indicator keys for {ticker}: {indicators[ticker].keys()}")

        # Populate the 3D array with the processed data 
        for date_idx, date in enumerate(date_range):
            for ticker_idx, ticker in enumerate(tickers):
                try:
                    # Economic indicators
                    data_3d[date_idx, ticker_idx, 0] = economic_indicators.loc[(date, ticker), 'GDP_Growth']
                    data_3d[date_idx, ticker_idx, 1] = economic_indicators.loc[(date, ticker), 'Unemployment_Rate']
                    data_3d[date_idx, ticker_idx, 2] = economic_indicators.loc[(date, ticker), 'Inflation_Rate']
                    data_3d[date_idx, ticker_idx, 3] = economic_indicators.loc[(date, ticker), 'VIX']

                    # Stock data
                    data_3d[date_idx, ticker_idx, 4] = stock_data.loc[(date, ticker), 'Open']
                    data_3d[date_idx, ticker_idx, 5] = stock_data.loc[(date, ticker), 'High']
                    data_3d[date_idx, ticker_idx, 6] = stock_data.loc[(date, ticker), 'Low']
                    data_3d[date_idx, ticker_idx, 7] = stock_data.loc[(date, ticker), 'Close']
                    data_3d[date_idx, ticker_idx, 8] = stock_data.loc[(date, ticker), 'Volume']

                    # News sentiment
                    try:
                        news_row = news_data.loc[(date, ticker)]
                        sentiment_score = news_row['sentiment_score']
                        if isinstance(sentiment_score, pd.Series):
                            sentiment_score = sentiment_score.mean()
                        data_3d[date_idx, ticker_idx, 9] = sentiment_score if not np.isnan(sentiment_score) else 0
                    except KeyError:
                        data_3d[date_idx, ticker_idx, 9] = 0

                    # Technical indicators
                    if date_idx >= window_size:
                        data_3d[date_idx, ticker_idx, 10] = indicators[ticker]['macd'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 11] = indicators[ticker]['macd_signal'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 12] = indicators[ticker]['rsi'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 13] = indicators[ticker]['upper_band'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 14] = indicators[ticker]['lower_band'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 15] = indicators[ticker]['stoch_osc'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 16] = indicators[ticker]['obv'][date_idx - window_size]
                        data_3d[date_idx, ticker_idx, 17] = indicators[ticker]['atr'][date_idx - window_size]

                    # Add time features
                    data_3d[date_idx, ticker_idx, base_features:] = time_features[date_idx]
                
                except KeyError as e:
                    self._log("Data Population Error", f"KeyError for date {date}, ticker {ticker}: {str(e)}")
                except IndexError as e:
                    self._log("Data Population Error", f"IndexError for date {date}, ticker {ticker}: {str(e)}")
                except Exception as e:
                    self._log("Data Population Error", f"Unexpected error for date {date}, ticker {ticker}: {str(e)}")

                except KeyError:
                    # Handle missing data by setting the corresponding values to 0 or an appropriate default value
                    data_3d[date_idx, ticker_idx, :] = 0
                    self._log(f"Date: {date}, Ticker: {ticker}", "KeyError occurred, data set to zeros", save_to_file=True)

        self._log("Combined Data Preview", f"Combined data shape: {data_3d.shape}", save_to_file=True)
        self._log("3D Array Structure", str(data_3d[:-3, :2, :]), save_to_file=False)

        return data_3d

# Data preprocessing function
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

    def fit_transform(self, data_3d):
        """
        1) Fit all scalers on 'data_3d' (historical data).
        2) Transform 'data_3d' with those newly fitted scalers.
        3) Store fitted scalers in self.feature_scalers and self.ticker_scalers.
        
        Returns:
            np.ndarray: Scaled data (same shape as 'data_3d').
        """
        data_3d = np.nan_to_num(data_3d)
        num_dates, num_tickers, _ = data_3d.shape

        # Make a copy so we don’t mutate original input
        data_processed = np.copy(data_3d)

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

    def transform(self, data_3d):
        """
        Apply *already-fitted* scalers to new data (e.g., real-time).
        No .fit(...) calls here—strictly .transform(...).
        
        Returns:
            np.ndarray: scaled data (same shape as 'data_3d').
        """
        data_3d = np.nan_to_num(data_3d)
        num_dates, num_tickers, _ = data_3d.shape
        data_processed = np.copy(data_3d)

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

# Custom Trading Environment
class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data, tradeable_tickers, feature_tickers, feature_scalers, ticker_scalers, initial_balance=10000, transaction_cost=0.001, window_size=10,
                 bot_logger=None, entropy_scale=0.1, is_validation=False, seed=None): # Added seed
        super(TradingEnv, self).__init__()

        # Seeding for reproducibility within the environment instance
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            # Note: TF seeding should happen outside, per process

        self.data = data
        self.feature_scalers = feature_scalers
        self.ticker_scalers = ticker_scalers
        self.price_indices = [4, 5, 6, 7]  # Indices of price features
        self.open_price_index = 4  # index for 'open'
        self.close_price_index = 7  # Index for Close price

        if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
             raise ValueError(f"Invalid data shape: {data.shape}. Must be 3D and non-empty.")
        self.num_dates, self.num_total_tickers_in_data, self.num_features = data.shape

        self.tradeable_tickers = tradeable_tickers
        self.feature_tickers = feature_tickers
        self.num_tradeable = len(tradeable_tickers)
        self.num_feature_tickers = len(feature_tickers)

        # --- Crucial Check: Ensure data dimensions match tickers ---
        expected_total_tickers = self.num_feature_tickers + self.num_tradeable
        if self.num_total_tickers_in_data != expected_total_tickers:
            raise ValueError(f"Data shape mismatch: data has {self.num_total_tickers_in_data} tickers, "
                             f"but expected {expected_total_tickers} ({self.num_feature_tickers} feature + {self.num_tradeable} tradeable).")
        # --- End Check ---


        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.epsilon = 1e-9

        # Start current_step at window_size to ensure enough data for the first observation
        self.window_size = window_size
        self.current_step = self.window_size # Start after the first window

        self.bot_logger = bot_logger if bot_logger else bot_logger() # Use provided or default logger
        self.entropy_scale = entropy_scale  # Scale factor for entropy bonus
        self.is_validation = is_validation

        # Indices for shared features (Ensure these indices are valid for self.num_features)
        self.economic_indicator_indices = [0, 1, 2, 3] # Indices for GDP, Unemployment, Inflation, VIX
        self.news_sentiment_index = 9 # Index for news sentiment - MAKE SURE THIS EXISTS if used
        self.time_feature_indices = list(range(18, 26)) # Indices for time features

        # Validate feature indices
        all_indices = self.economic_indicator_indices + [self.news_sentiment_index] + self.time_feature_indices
        if any(idx >= self.num_features for idx in all_indices):
             raise ValueError(f"One or more feature indices are out of bounds (num_features={self.num_features}). Check indices.")


        # After preprocessing, 'is_weekend' and 'is_holiday' are among the time features
        self.is_weekend_index = 19 # Ensure this index exists and corresponds to 'is_weekend'
        self.is_holiday_index = 26 # Ensure this index exists and corresponds to 'is_holiday'

        # Compute is_trading_day array (using data from the first ticker for shared features)
        if self.is_weekend_index >= self.num_features or self.is_holiday_index >= self.num_features:
            raise ValueError("is_weekend_index or is_holiday_index out of bounds.")

        # Use data from the *first ticker* (index 0) for shared features like is_weekend/is_holiday
        is_weekend = self.data[:, 0, self.is_weekend_index]
        is_holiday = self.data[:, 0, self.is_holiday_index]
        self.is_trading_day = (is_weekend == 0) & (is_holiday == 0)


        # Define action space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_tradeable,),
            dtype=np.float32
        )

        # Indices for all shared features
        self.shared_feature_indices = list(set( # Use set to avoid duplicates if indices overlap
             self.economic_indicator_indices +
             self.time_feature_indices +
             ([self.news_sentiment_index] if self.news_sentiment_index < self.num_features else []) # Conditionally add news index
        ))
        self.shared_feature_indices.sort()


        # Indices for ticker-specific features (excluding shared features)
        self.ticker_feature_indices = [
             i for i in range(self.num_features) if i not in self.shared_feature_indices
        ]
        self.ticker_feature_indices.sort()


        # Define observation space using Gymnasium spaces
        obs_spaces = {
            'shared_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, len(self.shared_feature_indices)), dtype=np.float32
            ),
            'ticker_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, self.num_tradeable, len(self.ticker_feature_indices)), dtype=np.float32
            ),
            'portfolio_holdings': spaces.Box(
                low=0, high=np.inf, # Holdings cannot be negative
                shape=(self.num_tradeable,), dtype=np.float32
            ),
            'cash_balance': spaces.Box(
                low=0, high=np.inf, # Balance cannot be negative (or handle bankruptcy)
                shape=(1,), dtype=np.float32
            )
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # Add debugging print statements
        print(f"Data shape: {self.data.shape}")
        print(f"Is weekend index: {self.is_weekend_index}")
        print(f"Is holiday index: {self.is_holiday_index}")
        print(f"Close index: {self.close_price_index}")
        print(f"Is trading day array shape: {self.is_trading_day.shape}")
        print(f"Number of trading days: {np.sum(self.is_trading_day)}")
        print(f"First 20 is_trading_day values: {self.is_trading_day[:20]}")

        # Logging
        if self.bot_logger:
            self.bot_logger.record("init", f"Initialized TradingEnv. Tradeable: {self.num_tradeable}, Features: {self.num_features}, Window: {self.window_size}, Days: {self.num_dates}")
            self.bot_logger.record("init", f"Shared Feature Indices: {self.shared_feature_indices}")
            self.bot_logger.record("init", f"Ticker Feature Indices: {self.ticker_feature_indices}")


        # --- Precompute Unscaled Prices ---
        # Validate price indices exist
        if self.open_price_index >= self.num_features or self.close_price_index >= self.num_features:
             raise ValueError("Open or Close price index out of bounds.")

        self.unscaled_open_prices = np.zeros((self.num_dates, self.num_tradeable))
        self.unscaled_close_prices = np.zeros((self.num_dates, self.num_tradeable))

        for date_idx in range(self.num_dates):
             for i in range(self.num_tradeable):
                 # Map tradeable ticker index 'i' to its index in the full data array
                 # Feature tickers come first, then tradeable tickers
                 data_ticker_idx = self.num_feature_tickers + i

                 # --- Unscale Open Price ---
                 # Ensure the scaler exists for this ticker and feature index
                 if data_ticker_idx not in self.ticker_scalers or self.open_price_index not in self.ticker_scalers[data_ticker_idx]:
                      raise KeyError(f"Scaler not found for ticker index {data_ticker_idx}, feature index {self.open_price_index} (open price)")
                 scaler_open = self.ticker_scalers[data_ticker_idx][self.open_price_index]
                 scaled_open = self.data[date_idx, data_ticker_idx, self.open_price_index]
                 # Inverse transform requires a 2D array
                 unscaled_open = scaler_open.inverse_transform([[scaled_open]])[0, 0]
                 self.unscaled_open_prices[date_idx, i] = unscaled_open

                 # --- Unscale Close Price ---
                 if data_ticker_idx not in self.ticker_scalers or self.close_price_index not in self.ticker_scalers[data_ticker_idx]:
                      raise KeyError(f"Scaler not found for ticker index {data_ticker_idx}, feature index {self.close_price_index} (close price)")
                 scaler_close = self.ticker_scalers[data_ticker_idx][self.close_price_index]
                 scaled_close = self.data[date_idx, data_ticker_idx, self.close_price_index]
                 # Inverse transform requires a 2D array
                 unscaled_close = scaler_close.inverse_transform([[scaled_close]])[0, 0]
                 self.unscaled_close_prices[date_idx, i] = unscaled_close


        # Final check for NaNs in unscaled prices
        if np.any(np.isnan(self.unscaled_open_prices)) or np.any(np.isnan(self.unscaled_close_prices)):
             self.bot_logger.warning("NaN values detected in precomputed unscaled prices after inverse transform.")
             # Consider handling this, e.g., by filling NaNs or raising an error earlier during scaling checks.

        self.reset()


    def reset(self, seed=None, options=None):
        # Handle seeding according to Gymnasium standard
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            # TF seeding should be handled per process outside

        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.num_tradeable, dtype=np.float32) # Use float32 consistently

        # Reset step to the start of the window
        self.current_step = self.window_size # Start after the first window

        # Ensure there are enough steps left for at least one trade
        if self.current_step >= self.num_dates:
             raise ValueError(f"Window size ({self.window_size}) is too large for the number of dates ({self.num_dates}). No trading possible.")

        self.trade_history = []
        # self.position_open_time = np.full(self.num_tradeable, -1) # If needed for PDT rules etc.
        # self.day_trade_history = deque(maxlen=5) # If needed
        # self.day_trades_today = 0 # If needed

        if self.bot_logger:
            self.bot_logger.record("Environment Reset", f"Env reset. Initial balance: {self.initial_balance:.2f}. Start step: {self.current_step}")

        observation = self._get_observation()
        info = self._get_info() # Standard Gymnasium practice

        # Ensure observation matches space definition
        if not self.observation_space.contains(observation):
             print("Observation Space:", self.observation_space)
             print("Generated Observation:", observation)
             for key in self.observation_space.spaces:
                 if key not in observation:
                     print(f"Missing key in observation: {key}")
                 elif self.observation_space[key].shape != observation[key].shape:
                     print(f"Shape mismatch for key '{key}': Expected {self.observation_space[key].shape}, Got {observation[key].shape}")
                 elif self.observation_space[key].dtype != observation[key].dtype:
                      print(f"Dtype mismatch for key '{key}': Expected {self.observation_space[key].dtype}, Got {observation[key].dtype}")
             raise ValueError("Reset observation does not match observation space.")


        return observation, info

    # Numba helper for observation calculation (Keep static if possible)
    @staticmethod
    #@njit(parallel=True)
    def compute_observation_window(data, shared_indices, ticker_indices, start_idx, window_size, num_tradeable, num_feature_tickers):
        """
        Computes the observation window using data slicing.

        Args:
            data (np.ndarray): Shape (num_dates, num_total_tickers, num_features)
            shared_indices (np.ndarray): Indices of shared features.
            ticker_indices (np.ndarray): Indices of ticker-specific features.
            start_idx (int): The starting index for the window slice (inclusive).
            window_size (int): The number of steps in the window.
            num_tradeable (int): Number of tradeable tickers.
            num_feature_tickers (int): Number of feature-only tickers.

        Returns:
            tuple: (shared_window, ticker_window)
                   shared_window shape: (window_size, num_shared_features)
                   ticker_window shape: (window_size, num_tradeable, num_ticker_features)
        """
        end_idx = start_idx + window_size # Exclusive end index

        # --- Shared Features ---
        # Assumed to be taken from the *first ticker* (index 0) in the data
        # Shape: (window_size, num_features) -> select columns -> (window_size, num_shared_features)
        shared_window = data[start_idx:end_idx, 0, shared_indices]

        # --- Ticker Features ---
        # Shape: (window_size, num_tradeable, num_features) -> select columns -> (window_size, num_tradeable, num_ticker_features)
        # Need to select the data corresponding to tradeable tickers
        tradeable_data_start_idx = num_feature_tickers
        tradeable_data_end_idx = num_feature_tickers + num_tradeable
        ticker_window = data[start_idx:end_idx, tradeable_data_start_idx:tradeable_data_end_idx, ticker_indices]

        return shared_window.astype(np.float32), ticker_window.astype(np.float32)


    def _get_observation(self):
         """
         Gets the observation for the current step.
         Uses a window of past data ending at the step *before* the current one.
         """
         if self.current_step < self.window_size:
             raise ValueError(f"Cannot get observation at step {self.current_step}, window size is {self.window_size}. Need at least {self.window_size} steps.")

         # The window should contain data from [current_step - window_size, current_step)
         start_idx = self.current_step - self.window_size
         end_idx = self.current_step # Exclusive

         # Ensure indices are valid
         if start_idx < 0 or end_idx > self.num_dates:
              raise IndexError(f"Calculated window indices [{start_idx}, {end_idx}) are out of bounds for data with {self.num_dates} dates.")

         # Convert feature index lists to numpy arrays for Numba/slicing
         shared_indices = np.array(self.shared_feature_indices, dtype=np.int64)
         ticker_indices = np.array(self.ticker_feature_indices, dtype=np.int64)

         # Call the Numba function for efficient slicing
         shared_window, ticker_window = TradingEnv.compute_observation_window(
             self.data,
             shared_indices,
             ticker_indices,
             start_idx,
             self.window_size,
             self.num_tradeable,
             self.num_feature_tickers # Pass the number of feature tickers
         )

         # Construct the observation dictionary
         obs = {
             'shared_features': shared_window, # Shape: (window_size, num_shared)
             'ticker_features': ticker_window, # Shape: (window_size, num_tradeable, num_ticker)
             'portfolio_holdings': self.portfolio.astype(np.float32), # Shape: (num_tradeable,)
             'cash_balance': np.array([self.balance], dtype=np.float32) # Shape: (1,)
         }

         # --- Verification ---
         if obs['shared_features'].shape != self.observation_space['shared_features'].shape:
             raise ValueError(f"Shape mismatch for shared_features: Got {obs['shared_features'].shape}, Expected {self.observation_space['shared_features'].shape}")
         if obs['ticker_features'].shape != self.observation_space['ticker_features'].shape:
             raise ValueError(f"Shape mismatch for ticker_features: Got {obs['ticker_features'].shape}, Expected {self.observation_space['ticker_features'].shape}")
         if obs['portfolio_holdings'].shape != self.observation_space['portfolio_holdings'].shape:
              raise ValueError(f"Shape mismatch for portfolio_holdings: Got {obs['portfolio_holdings'].shape}, Expected {self.observation_space['portfolio_holdings'].shape}")
         if obs['cash_balance'].shape != self.observation_space['cash_balance'].shape:
              raise ValueError(f"Shape mismatch for cash_balance: Got {obs['cash_balance'].shape}, Expected {self.observation_space['cash_balance'].shape}")
         # --- End Verification ---

         return obs


    def _get_info(self):
        """Returns auxiliary information dictionary (standard in Gymnasium)."""
        return {
            'current_step': self.current_step,
            'portfolio_value': self._get_portfolio_value(self.current_step -1), # Value at end of *previous* day
            'balance': self.balance,
            'holdings': self.portfolio.copy()
        }

    def _calculate_reward(self, portfolio_value, prev_portfolio_value):
        """
        Calculates the reward for the *transition* that just occurred.
        Uses portfolio value *before* and *after* the step.
        """
        # --- Basic Portfolio Return ---
        # Avoid division by zero if previous value was 0
        portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > self.epsilon else 0
        # Scale and clip the return to keep rewards bounded
        scaled_portfolio_return = np.clip(portfolio_return * 10, -1, 1) # Example scaling factor 10

        # --- Indices for Price Lookups ---
        # Reward is calculated *after* the step, so use current_step - 1 for the *end* price of the step
        idx = min(self.current_step - 1, self.num_dates - 1) # Index for current prices (end of step)
        prev_idx = max(0, idx - 1)                          # Index for previous prices (start of step)

        # --- Vectorized Holding Rewards Calculation ---
        # Use prices corresponding to the *end* of the step (idx) and *start* of the step (prev_idx)
        current_prices = self.unscaled_close_prices[idx, :]  # Prices at end of step
        prev_prices = self.unscaled_close_prices[prev_idx, :] # Prices at start of step

        # Calculate position values at the *end* of the step using *current* holdings and *current* prices
        # Note: self.portfolio reflects holdings *after* trades might have occurred at the beginning of the step
        position_values_end = self.portfolio * current_prices

        # Mask for positions held *at the end* of the step (quantity > epsilon)
        holding_mask = self.portfolio > self.epsilon

        # Calculate returns for each asset over the step duration
        position_returns = np.zeros_like(self.portfolio, dtype=float)
        valid_return_mask = holding_mask & (prev_prices > self.epsilon) # Ensure previous price is valid
        position_returns[valid_return_mask] = (current_prices[valid_return_mask] - prev_prices[valid_return_mask]) / prev_prices[valid_return_mask]

        # Calculate weights based on position values *at the end* of the step relative to total portfolio value *at the end*
        position_weights = np.zeros_like(self.portfolio, dtype=float)
        if portfolio_value > self.epsilon:
            # Calculate weights only for positions held at the end
            position_weights[holding_mask] = position_values_end[holding_mask] / portfolio_value

        # Calculate weighted holding rewards: sum of (return_of_asset * weight_of_asset_at_end)
        # This rewards holding profitable assets.
        holding_rewards = np.sum(position_returns[holding_mask] * position_weights[holding_mask])
        scaled_holding_rewards = np.clip(holding_rewards * 10, -1, 1) # Example scaling factor 10

        # --- Logging ---
        if self.bot_logger:
            self.bot_logger.record("Reward Calc", f"Step {self.current_step-1}: Prev Value={prev_portfolio_value:.2f}, Curr Value={portfolio_value:.2f}, Return={portfolio_return:.4f}")
            # Log holdings and their individual contribution if needed (can be verbose)
            # self.bot_logger.record("Reward Calc", f"Holdings: {self.portfolio}")
            # self.bot_logger.record("Reward Calc", f"Position Returns: {position_returns}")
            # self.bot_logger.record("Reward Calc", f"Position Weights: {position_weights}")
            self.bot_logger.record("Reward Calc", f"Holding Rewards (Raw): {holding_rewards:.4f}, Scaled: {scaled_holding_rewards:.4f}")


        # --- Sharpe Ratio (Optional, requires history) ---
        sharpe_reward = 0
        # Example: Calculate over last 30 steps if history is sufficient
        if len(self.trade_history) >= 30:
            recent_returns = np.array([trade['return'] for trade in self.trade_history[-30:]])
            std_dev = np.std(recent_returns)
            if std_dev > self.epsilon:
                sharpe = np.mean(recent_returns) / std_dev * np.sqrt(252) # Annualized (example)
                sharpe_reward = np.clip(sharpe * 0.1, -1, 1) # Example scaling

        # --- Concentration Penalty (Optional) ---
        # Penalize if portfolio is too concentrated in a few assets
        # Use weights calculated earlier (based on end-of-step values)
        concentration_penalty = 0
        if portfolio_value > self.epsilon:
             # Herfindahl-Hirschman Index (HHI) - sum of squared weights
             hhi = np.sum(np.square(position_weights))
             # Scale penalty (e.g., higher penalty as HHI approaches 1)
             concentration_penalty = np.clip(hhi * 0.05, 0, 1) # Example scaling


        # --- Drawdown Penalty (Optional, requires history) ---
        drawdown_penalty = 0
        if self.trade_history:
             # Calculate max portfolio value over a recent window (e.g., last 30 steps)
             rolling_max = max([trade.get('portfolio_value', self.initial_balance)
                                for trade in self.trade_history[-30:]], default=self.initial_balance)
             if rolling_max > self.epsilon:
                 drawdown = max(0, (rolling_max - portfolio_value) / rolling_max)
                 drawdown_penalty = np.clip(drawdown * 0.1, 0, 1) # Example scaling

        # --- Combine Reward Components (Example Weights) ---
        reward = (
            scaled_portfolio_return * 0.50 +   # Primary: overall return
            scaled_holding_rewards * 0.20 +    # Reward holding profitable assets
            sharpe_reward * 0.00 +             # risk-adjusted return (weight 0 if unused)
            -concentration_penalty * 0.10 +    # diversification penalty
            -drawdown_penalty * 0.20           # risk management penalty
        )

        # Final clipping to ensure reward stays within a standard range (e.g., [-1, 1])
        reward = np.clip(reward, -1, 1)

        # --- Store Trade History (for potential later analysis or reward calculation) ---
        self.trade_history.append({
            'step': self.current_step - 1, # Step number where this transition ended
            'return': portfolio_return,    # Raw portfolio return for this step
            'portfolio_value': portfolio_value, # Value at the end of this step
            'cash_balance': self.balance,
            'holdings_vector': self.portfolio.copy(),
            'reward': reward, # Store the final calculated reward for this step
            # Store raw components if needed for analysis
            'raw_holding_rewards': holding_rewards,
            'raw_concentration_penalty': concentration_penalty,
            'raw_drawdown_penalty': drawdown_penalty,
        })

        if self.bot_logger:
             log_msg = (
                 f"\nReward Breakdown (Step {self.current_step-1}):"
                 f"\n  Scaled Port Return : {scaled_portfolio_return:+.4f} (w: 0.50)"
                 f"\n  Scaled Hold Reward : {scaled_holding_rewards:+.4f} (w: 0.20)"
                 f"\n  Sharpe Reward      : {sharpe_reward:+.4f} (w: 0.00)"
                 f"\n  Concentration Pen. : {-concentration_penalty:.4f} (w: 0.10)"
                 f"\n  Drawdown Penalty   : {-drawdown_penalty:.4f} (w: 0.20)"
                 f"\n  ---------------------------------"
                 f"\n  Final Reward (Clip): {reward:+.4f}"
             )
             self.bot_logger.record("Reward Components", log_msg)


        return reward


    def step(self, action):
        """
        Executes one time step within the environment.
        Follows the Gymnasium API: returns observation, reward, terminated, truncated, info.
        """
        # --- Input Validation ---
        if not self.action_space.contains(action):
             self.bot_logger.error(f"Invalid action received: {action}. Expected shape {self.action_space.shape} and bounds {self.action_space.low} to {self.action_space.high}")
             # Handle invalid action, e.g., clip it or raise error
             action = np.clip(action, self.action_space.low, self.action_space.high)
             # raise ValueError(f"Action {action} is not in the action space {self.action_space}")

        # Store portfolio value *before* taking the step (at the end of the previous day)
        prev_portfolio_value = self._get_portfolio_value(self.current_step - 1)

        # --- Execute Trades ---
        # Trades are executed based on the action and *opening* prices of the *current* day (self.current_step)
        # Check if it's a valid trading day based on precomputed flags
        is_today_trading_day = False
        if self.current_step < self.num_dates:
            is_today_trading_day = self.is_trading_day[self.current_step]
            if is_today_trading_day:
                self._execute_trades(action, self.current_step)
            else:
                if self.bot_logger: self.bot_logger.record("Step", f"Step {self.current_step}: Non-trading day. No trades executed.")
        # else:
            # if self.bot_logger: self.bot_logger.record("Step", f"Step {self.current_step}: Beyond data range. No trades executed.")


        # --- Move to the next day ---
        self.current_step += 1

        # --- Determine Termination and Truncation ---
        terminated = False # Typically False in trading unless a specific condition like bankruptcy is met
        truncated = False # True if the episode ends because the time limit (end of data) is reached

        if self.current_step >= self.num_dates:
            # print(f"Truncating at step {self.current_step} (num_dates={self.num_dates})")
            truncated = True
            # Ensure we don't try to access data beyond the limits in reward calculation
            current_portfolio_value = self._get_portfolio_value(self.num_dates - 1)
        else:
            # Calculate portfolio value at the *end* of the day that just finished (current_step - 1)
            current_portfolio_value = self._get_portfolio_value(self.current_step - 1)


        # --- Calculate Reward ---
        # Reward is based on the transition from prev_portfolio_value to current_portfolio_value
        reward = self._calculate_reward(current_portfolio_value, prev_portfolio_value)

        # --- Get Next Observation and Info ---
        # Observation depends on the *new* current_step
        # If truncated, the observation might not be valid or needed, but Gymnasium expects it.
        # We might return the last valid observation or handle it in the agent.
        # For simplicity, let's get the observation if possible.
        if truncated:
             # If truncated, we can't get a valid observation for the *next* step.
             # Return the last valid observation or a zeroed-out one matching the space.
             # Getting the last valid one requires storing it. Let's return the current one for now.
             observation = self._get_observation() # Gets obs based on window ending *before* truncated step
             # Or create a dummy observation matching the space structure:
             # observation = self.observation_space.sample() # Or zeros
        else:
             observation = self._get_observation()


        info = self._get_info() # Info reflects state *after* the step

        # --- Logging ---
        if self.bot_logger:
             self.bot_logger.record("Step End",
                 f"Step: {self.current_step-1} -> {self.current_step} | "
                 f"Action: {np.round(action, 2)} | "
                 f"Reward: {reward:.4f} | "
                 f"Value: {current_portfolio_value:.2f} | "
                 f"Balance: {self.balance:.2f} | "
                 f"Term: {terminated} | Trunc: {truncated}"
             )

        # --- Final Check ---
        if not self.observation_space.contains(observation):
            print("Observation Space:", self.observation_space)
            print("Generated Observation:", observation)
            raise ValueError("Step observation does not match observation space.")


        # Return according to Gymnasium API v26
        return observation, reward, terminated, truncated, info


    def _execute_trades(self, actions, day_index):
        """
        Vectorized execution of trades based on continuous actions for a specific day.
        actions: numpy array of shape (num_tradeable,) with values between -1 and 1.
        day_index: The index of the day for which to use opening prices.
        """
        MIN_ACTION_THRESHOLD = 0.001  # Minimum action magnitude to trigger a trade

        # --- Price Check ---
        if day_index < 0 or day_index >= self.num_dates:
            if self.bot_logger: self.bot_logger.warning(f"Invalid day_index {day_index} in _execute_trades, skipping.")
            return

        # Get open prices for all tradeable tickers for the *current* day
        try:
            # Ensure we only take prices for tradeable tickers
            prices = self.unscaled_open_prices[day_index, :self.num_tradeable]
        except IndexError:
            if self.bot_logger: self.bot_logger.error(f"IndexError accessing open prices at day_index {day_index}.")
            return

        # --- Input Validation ---
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            if self.bot_logger: self.bot_logger.error(f"NaN/Inf in actions: {actions}. Skipping trade.")
            return
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)) or np.any(prices <= self.epsilon):
             # Identify problematic tickers
             bad_price_mask = np.isnan(prices) | np.isinf(prices) | (prices <= self.epsilon)
             bad_tickers = [self.tradeable_tickers[i] for i, bad in enumerate(bad_price_mask) if bad]
             if self.bot_logger: self.bot_logger.warning(f"Invalid prices detected for tickers {bad_tickers} at step {day_index}. Masking actions for these tickers.")
             # Mask actions for tickers with bad prices - don't trade them
             actions = np.where(bad_price_mask, 0, actions) # Set action to 0 if price is bad
             prices = np.where(bad_price_mask, 1, prices) # Set price to 1 to avoid division by zero later, action is 0 anyway
             # If all prices are bad, skip entirely
             if np.all(bad_price_mask):
                  if self.bot_logger: self.bot_logger.error(f"All ticker prices invalid at step {day_index}. Skipping all trades.")
                  return


        # Use float64 for precision in calculations
        actions = np.asarray(actions, dtype=np.float64)
        portfolio = self.portfolio.astype(np.float64) # Current holdings before trades
        tc = self.transaction_cost
        current_balance = float(self.balance) # Start with current balance

        # --- Sell Orders ---
        sell_mask = (actions < -MIN_ACTION_THRESHOLD) & (portfolio > self.epsilon) # Sell only if holding shares
        if np.any(sell_mask):
            sell_fraction = np.abs(actions[sell_mask])
            # Quantity to sell is fraction of *current* holdings
            sell_quantity_desired = sell_fraction * portfolio[sell_mask]
            # Ensure we don't sell more than we have (due to float precision)
            sell_quantity_actual = np.minimum(portfolio[sell_mask], np.round(sell_quantity_desired, 8)) # Round to avoid tiny sells

            # Filter out sells too small to matter
            non_zero_sell_mask_local = sell_quantity_actual > self.epsilon
            if np.any(non_zero_sell_mask_local):
                sell_indices_global = np.where(sell_mask)[0][non_zero_sell_mask_local]
                actual_sell_quantity = sell_quantity_actual[non_zero_sell_mask_local]
                actual_sell_prices = prices[sell_mask][non_zero_sell_mask_local]

                proceeds = actual_sell_quantity * actual_sell_prices * (1 - tc)
                total_proceeds = np.sum(proceeds)

                # --- Log Sells ---
                if self.bot_logger:
                     for i, global_idx in enumerate(sell_indices_global):
                         ticker = self.tradeable_tickers[global_idx]
                         qty = actual_sell_quantity[i]
                         price_val = actual_sell_prices[i]
                         proc = proceeds[i]
                         self.bot_logger.record("Sell Trade", f"Day {day_index}: Sold {qty:.4f} {ticker} @ {price_val:.2f}. Proceeds: {proc:.2f}")


                # Update balance and portfolio (use global indices)
                current_balance += total_proceeds
                portfolio[sell_indices_global] -= actual_sell_quantity


        # --- Buy Orders ---
        # Use the potentially updated balance after sells
        buy_mask = (actions > MIN_ACTION_THRESHOLD) & (prices > self.epsilon) # Buy only if price is valid
        if np.any(buy_mask) and current_balance > self.epsilon:
            desired_buy_fraction = actions[buy_mask] # Positive values
            current_prices_buy = prices[buy_mask]
            price_plus_cost = current_prices_buy * (1 + tc) # Cost per share including tc

            # --- Allocation Strategy ---
            # Allocate available balance proportionally to the *strength* of the buy signal (action value)
            total_positive_action_strength = np.sum(desired_buy_fraction)

            if total_positive_action_strength > self.epsilon:
                allocation_fraction = desired_buy_fraction / total_positive_action_strength
                target_cash_allocation = current_balance * allocation_fraction

                # Calculate desired quantity based on allocated cash
                # Avoid division by zero (already checked price_plus_cost > epsilon indirectly via buy_mask)
                desired_buy_quantity_raw = target_cash_allocation / price_plus_cost
                desired_buy_quantity = np.round(desired_buy_quantity_raw, 8) # Round to avoid tiny buys

                # Calculate the cost of these desired quantities
                desired_cost = desired_buy_quantity * price_plus_cost
                total_desired_cost = np.sum(desired_cost)

                # --- Scaling Logic ---
                scale = 1.0
                if total_desired_cost > current_balance:
                    # Not enough balance, scale down proportionally
                    scale = (current_balance / total_desired_cost) * 0.99999 # Multiply by slightly less than 1 for safety

                # Calculate final buy quantity after scaling
                # Ensure we only scale quantities > 0
                buy_mask_scaled = desired_buy_quantity > self.epsilon
                buy_quantity_actual = np.zeros_like(desired_buy_fraction, dtype=np.float64)
                buy_quantity_actual[buy_mask_scaled] = np.round(desired_buy_quantity[buy_mask_scaled] * scale, 8)

                # Recalculate actual cost based on final scaled quantity
                cost_actual = np.zeros_like(desired_buy_fraction, dtype=np.float64)
                cost_actual[buy_mask_scaled] = buy_quantity_actual[buy_mask_scaled] * price_plus_cost[buy_mask_scaled]
                total_actual_cost = np.sum(cost_actual)

                # --- Update State & Log ---
                if total_actual_cost > self.epsilon:
                    # Get global indices corresponding to the buy_mask
                    buy_indices_global = np.where(buy_mask)[0]

                    # Filter to only buys that actually happened (quantity > epsilon)
                    actual_buy_mask_local = buy_quantity_actual > self.epsilon
                    if np.any(actual_buy_mask_local):
                         actual_buy_indices_global = buy_indices_global[actual_buy_mask_local]
                         final_buy_quantity = buy_quantity_actual[actual_buy_mask_local]
                         final_buy_cost = cost_actual[actual_buy_mask_local]
                         final_buy_prices = current_prices_buy[actual_buy_mask_local] # Prices for actual buys

                         # --- Log Buys ---
                         if self.bot_logger:
                              for i, global_idx in enumerate(actual_buy_indices_global):
                                  ticker = self.tradeable_tickers[global_idx]
                                  qty = final_buy_quantity[i]
                                  price_val = final_buy_prices[i]
                                  c = final_buy_cost[i]
                                  self.bot_logger.record("Buy Trade", f"Day {day_index}: Bought {qty:.4f} {ticker} @ {price_val:.2f}. Cost: {c:.2f}")


                         # Update balance and portfolio
                         self.balance -= np.sum(final_buy_cost) # Use the final calculated cost sum
                         portfolio[actual_buy_indices_global] += final_buy_quantity


        # --- Final State Update ---
        # Update the master portfolio state. Ensure no NaNs/Infs and non-negative.
        self.portfolio = np.maximum(0, np.nan_to_num(portfolio)).astype(np.float32) # Ensure non-negative and float32
        self.balance = max(0, np.nan_to_num(self.balance)) # Ensure non-negative

        # Final assertions for safety
        # assert np.all(np.isfinite(self.portfolio)), f"NaN/Inf in portfolio after trade: {self.portfolio}"
        # assert np.isfinite(self.balance), f"NaN/Inf in balance after trade: {self.balance}"
        # assert self.balance >= -self.epsilon, f"Balance negative after trade: {self.balance}" # Allow slightly below zero for float errors
        # assert np.all(self.portfolio >= -self.epsilon), f"Portfolio negative after trade: {self.portfolio}"


    def _get_portfolio_value(self, day_index):
        """Calculates portfolio value based on *closing* prices for a given day index."""
        if day_index < 0 or day_index >= self.num_dates:
             if self.bot_logger: self.bot_logger.warning(f"Invalid day_index {day_index} in _get_portfolio_value. Returning balance only.")
             # Decide on fallback: return balance, 0, or raise error? Let's return balance.
             return self.balance

        try:
            # Use precomputed unscaled *close* prices for the given day
            prices = self.unscaled_close_prices[day_index, :self.num_tradeable] # Shape: (num_tradeable,)
        except IndexError:
             if self.bot_logger: self.bot_logger.error(f"IndexError accessing close prices at day_index {day_index}.")
             return self.balance # Fallback

        # Handle potential NaNs in prices (e.g., if data is missing for that day)
        if np.any(np.isnan(prices)):
             # if self.bot_logger: self.bot_logger.warning(f"NaN prices detected at day_index {day_index} in _get_portfolio_value. Using 0 for these holdings.")
             prices = np.nan_to_num(prices) # Replace NaN with 0

        # Perform element-wise multiplication and sum
        holdings_value = np.sum(self.portfolio * prices)
        portfolio_value = self.balance + holdings_value

        # Optional detailed logging (can be verbose)
        # if self.bot_logger and day_index % 50 == 0: # Log less frequently
        #     self.bot_logger.record("Portfolio Value Calc", f"Day {day_index}: HoldingsVal={holdings_value:.2f}, Balance={self.balance:.2f}, Total={portfolio_value:.2f}")
            # Log individual holdings if needed
            # for i in range(self.num_tradeable):
            #     self.bot_logger.record("Portfolio", f"  Ticker {self.tradeable_tickers[i]}: Price={prices[i]:.2f}, Qty={self.portfolio[i]:.4f}, Val={self.portfolio[i] * prices[i]:.2f}")

        return portfolio_value

    def render(self, mode='human'):
        # Use render_mode instead of mode='human'
        portfolio_value = self._get_portfolio_value(self.current_step - 1)
        print(f'Step: {self.current_step}, Portfolio Value: {portfolio_value:.2f}, Balance: {self.balance:.2f}')
        # Add holdings display if desired
        holdings_str = ", ".join([f"{ticker}: {qty:.2f}" for ticker, qty in zip(self.tradeable_tickers, self.portfolio) if qty > self.epsilon])
        print(f' Holdings: {holdings_str if holdings_str else "None"}')

    def close(self):
        """Clean up any resources (e.g., rendering windows)."""
        # Add cleanup logic if needed (e.g., closing plot windows)
        print("Closing TradingEnv.")


# SAC Replay Buffer (Seems okay, ensure state keys match)
class SACReplayBuffer:
    def __init__(self, capacity, state_shape_dict, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0 # Track current number of items

        # Ensure state_shape_dict contains the expected keys from TradingEnv observation space
        expected_keys = ['shared_features', 'ticker_features', 'portfolio_holdings', 'cash_balance']
        if not all(key in state_shape_dict for key in expected_keys):
             raise ValueError(f"state_shape_dict missing one or more expected keys: {expected_keys}")

        # Initialize buffer arrays based on the dictionary structure
        self.states = {}
        self.next_states = {}
        for key, shape_obj in state_shape_dict.items():
             # Use shape directly from Gymnasium space object
             shape = shape_obj.shape
             dtype = shape_obj.dtype
             self.states[key] = np.zeros((capacity, *shape), dtype=dtype)
             self.next_states[key] = np.zeros((capacity, *shape), dtype=dtype)

        # Buffers for actions, rewards, dones
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        # Store dones as boolean or float32? TF usually expects float32 for (1-dones) calculation
        self.dones = np.zeros((capacity, 1), dtype=np.float32) # Store as float32

    def push(self, state, action, reward, next_state, done):
        """Stores a transition in the buffer."""
        # Store state components
        for key in self.states.keys():
            if key not in state or key not in next_state:
                 print(f"Warning: Key '{key}' not found in state or next_state during push.")
                 continue # Or raise error
            self.states[key][self.position] = state[key]
            self.next_states[key][self.position] = next_state[key]

        # Store other components
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done) # Convert boolean done to float

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        if self.size < batch_size:
            # Not enough samples yet, maybe return None or raise error
            # For training loop, it's better to check __len__ before calling sample
            raise ValueError(f"Cannot sample {batch_size} elements, only {self.size} available.")

        batch_indices = np.random.choice(self.size, batch_size, replace=False)

        # Sample states
        batch_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.states.items()
        }

        # Sample next states
        batch_next_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.next_states.items()
        }

        # Sample other components
        batch_actions = tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.float32)
        batch_rewards = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        batch_dones = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        """Returns the current number of transitions stored."""
        return self.size


# Define a named function to expand dimensions (Keep as is)
@register_keras_serializable()
def expand_dims_channel(x):
    return tf.expand_dims(x, axis=-1)

# SAC Agent (Modified for Gymnasium spaces and clarity)
class SACAgent:
    def __init__(self, observation_space, action_space, load_path=None, learning_rate=3e-5, gamma=0.99, tau=0.005, replay_buffer_capacity=int(1e6), log_dir_base='logs/sac_agent'):
        """
        Initialize SAC agent.

        Args:
            observation_space (gym.spaces.Dict): Environment observation space.
            action_space (gym.spaces.Box): Environment action space.
            load_path (str, optional): Path to load pre-trained model (base name without extensions). Defaults to None.
            learning_rate (float, optional): Learning rate for optimizers. Defaults to 3e-5.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update coefficient for target networks. Defaults to 0.005.
            replay_buffer_capacity (int, optional): Capacity of the replay buffer. Defaults to 1e6.
            log_dir_base (str, optional): Base directory for TensorBoard logs for this agent.
        """
        if not isinstance(observation_space, spaces.Dict):
             raise TypeError("observation_space must be a Gymnasium Dict space.")
        if not isinstance(action_space, spaces.Box):
             raise TypeError("action_space must be a Gymnasium Box space.")


        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = self.action_space.shape[0]

        # Get dimensions from spaces (using .spaces for Dict)
        self.window_size = observation_space['shared_features'].shape[0]
        self.num_shared_features = observation_space['shared_features'].shape[1]
        # Ticker features shape: (window, num_tradeable, num_ticker_features)
        self.num_tradeable = observation_space['ticker_features'].shape[1] # Get num_tradeable here
        self.num_ticker_features = observation_space['ticker_features'].shape[2]
        # Verify num_tradeable consistency
        if observation_space['portfolio_holdings'].shape[0] != self.num_tradeable:
             raise ValueError("Mismatch in num_tradeable between ticker_features and portfolio_holdings spaces.")
        
        # Initialize temperature parameter alpha
        # Use a variable for log_alpha, and a DeferredTensor for alpha itself
        self.log_alpha = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='log_alpha')
        # self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp) # tfp version
        self.alpha = tf.Variable(tf.exp(self.log_alpha), trainable=False, dtype=tf.float32) # Simpler TF variable for alpha value
        # Target entropy heuristic: -|A|
        self.target_entropy = -float(self.action_dim)
        # Initialize training parameters
        self.min_replay_size_to_train = 1000 # Start training only when buffer has this many samples
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64) # Use TF variable for step counting


        # --- TensorBoard Logging Setup ---
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        # Ensure log_dir_base exists or can be created by the writer
        self.log_dir = os.path.join(log_dir_base, f"SAC_{current_time}")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        print(f"Agent TensorBoard logs will be saved to: {self.log_dir}")


        # Initialize replay buffer
        self.replay_buffer = SACReplayBuffer(
            capacity=int(replay_buffer_capacity),
            state_shape_dict=self.observation_space.spaces, # Pass the dict of spaces
            action_dim=self.action_dim
        )

        # Initialize networks and optimizers
        self.actor = None
        self.critic_1, self.critic_2 = None, None
        self.target_critic_1, self.target_critic_2 = None, None
        self.actor_optimizer, self.critic_1_optimizer, self.critic_2_optimizer, self.alpha_optimizer = None, None, None, None

        if load_path and self._check_model_files_exist(load_path):
             print(f"Loading existing model from {load_path}")
             try:
                 self.load(load_path) # Load handles network and optimizer init
             except Exception as e:
                 print(f"Error loading model from {load_path}: {e}. Initializing new model.")
                 self._initialize_new_model()
        else:
             if load_path:
                  print(f"Model files not found at {load_path}. Initializing new model.")
             else:
                  print("No load path provided. Initializing new model.")
             self._initialize_new_model()

    def _check_model_files_exist(self, path_base):
        """Check if all necessary model files exist for loading."""
        required_suffixes = [
            "_actor.keras", "_critic1.keras", "_critic2.keras",
            "_target_critic1.keras", "_target_critic2.keras", "_params.pkl"
        ]
        return all(os.path.exists(f"{path_base}{suffix}") for suffix in required_suffixes)

    def _initialize_new_model(self):
        """Initialize a new model from scratch"""
        try:
            # Build networks
            self.actor = self._build_actor()
            self.critic_1 = self._build_critic()
            self.critic_2 = self._build_critic()
            self.target_critic_1 = self._build_critic()
            self.target_critic_2 = self._build_critic()

            # Copy weights to target networks
            self.target_critic_1.set_weights(self.critic_1.get_weights())
            self.target_critic_2.set_weights(self.critic_2.get_weights())
            
            # Create base optimizers first
            base_actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_critic_1_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_critic_2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_alpha_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            
            # Wrap the optimizers with LossScaleOptimizer for mixed precision
            self.actor_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_actor_optimizer)
            self.critic_1_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_critic_1_optimizer)
            self.critic_2_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_critic_2_optimizer)
            self.alpha_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_alpha_optimizer)
                
            print("Successfully initialized new model")
        except Exception as e:
            print(f"Error in _initialize_new_model: {str(e)}")
            raise

    def _get_inputs(self):
        """Helper function to create input layers"""
        return {
                'shared_features': Input(shape=(self.window_size, self.num_shared_features), name='shared_features_input'),
                'ticker_features': Input(shape=(self.window_size, self.num_tradeable, self.num_ticker_features), name='ticker_features_input'),
                'portfolio_holdings': Input(shape=(self.num_tradeable,), name='portfolio_holdings_input'),
                'cash_balance': Input(shape=(1,), name='cash_balance_input')
            }

    def _process_shared_features(self, shared_features):
        """
        Process shared features through LSTM layers with mixed precision support.
        Uses float16 for computation while maintaining stability with layer normalization.
        """
        # Convert input to float16 for efficient computation
        x = shared_features
        
        # Add layer normalization before LSTM for better numerical stability
        x = LayerNormalization(dtype='float32', name='shared_norm_1')(x)
        x = LSTM(128, activation='tanh', return_sequences=True,  name='shared_lstm_1')(x)
        
        # Another normalization layer between LSTMs
        x = LayerNormalization(dtype='float32', name='shared_norm_2')(x)
        x = LSTM(64, activation='tanh', name='shared_lstm_2')(x)
        
        # Final normalization and dense layer
        x = LayerNormalization(dtype='float32', name='shared_norm_3')(x)
        return Dense(64, activation='relu', kernel_regularizer=l2(1e-5), name='shared_features_output')(x)

    def _process_ticker_features(self, ticker_features_input):
        """Process ticker-specific features (e.g., OHLCV, indicators per ticker)."""
        # Expand dims -> ConvLSTM -> Norm -> ConvLSTM -> Norm -> Flatten -> Dense
        # Input shape: (batch, window, num_tradeable, num_ticker_features)
        x = Lambda(expand_dims_channel, name='expand_dims')(ticker_features_input)
        # Output shape: (batch, window, num_tradeable, num_ticker_features, 1)

        x = ConvLSTM2D(filters=64, kernel_size=(1, 3), padding='same', return_sequences=True, activation='tanh', name='ticker_convlstm_1')(x)
        # Output shape: (batch, window, num_tradeable, num_ticker_features, 64) -> Check this! ConvLSTM acts on spatial dims (tradeable, features)
        # Let's adjust kernel_size or architecture if needed. Assuming kernel (1,3) means conv over features.
        x = LayerNormalization(name='ticker_norm_1')(x)

        x = ConvLSTM2D(filters=32, kernel_size=(1, 3), padding='same', return_sequences=False, activation='tanh', name='ticker_convlstm_2')(x)
        # Output shape: (batch, num_tradeable, num_ticker_features, 32) - time dimension removed
        x = LayerNormalization(name='ticker_norm_2')(x)

        x = Flatten(name='ticker_flatten')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5), name='ticker_dense_out')(x)
        return x

    def _process_portfolio_state(self, portfolio_input, cash_input):
        """Process portfolio holdings and cash balance."""
        # Process holdings
        h = Dense(64, activation='relu', kernel_regularizer=l2(1e-5), name='portfolio_dense')(portfolio_input)
        h = LayerNormalization(name='portfolio_norm')(h)
        # Process cash (simple dense layer)
        c = Dense(16, activation='relu', kernel_regularizer=l2(1e-5), name='cash_dense')(cash_input)
        c = LayerNormalization(name='cash_norm')(c)
        # Combine portfolio and cash features
        combined_portfolio = Concatenate(name='portfolio_cash_concat')([h, c])
        return Dense(64, activation='relu', name='portfolio_cash_out')(combined_portfolio) # Final projection


    def _build_common_feature_extractor(self, state_inputs):
         """Builds the common part of the network that processes state inputs."""
         shared_processed = self._process_shared_features(state_inputs['shared_features'])
         ticker_processed = self._process_ticker_features(state_inputs['ticker_features'])
         portfolio_processed = self._process_portfolio_state(state_inputs['portfolio_holdings'], state_inputs['cash_balance'])

         # Combine all processed features
         combined = Concatenate(name='feature_combiner')([
             shared_processed,
             ticker_processed,
             portfolio_processed
         ])
         return combined


    def _build_actor(self):
        """Builds the actor network (policy)."""
        state_inputs = inputs = self._get_inputs()
        common_features = self._build_common_feature_extractor(state_inputs)

        # Actor-specific dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-5), name='actor_dense_1')(common_features)
        x = LayerNormalization(name='actor_norm_1')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5), name='actor_dense_2')(x)
        x = LayerNormalization(name='actor_norm_2')(x)

        # Output layers for mean and log_std of action distribution
        # Use linear activation for means, can be positive or negative
        means = Dense(self.action_dim, activation='linear', kernel_regularizer=l2(1e-5), name='action_means')(x)
        # Log standard deviations - ensure output is reasonable (e.g., using tanh or clipping later)
        log_stds = Dense(self.action_dim, activation='linear', kernel_regularizer=l2(1e-5), name='action_log_stds')(x)

        model = Model(inputs=list(state_inputs.values()), outputs=[means, log_stds], name='actor')
        print("\n--- Actor Summary ---")
        model.summary(line_length=120)
        return model

    def _build_critic(self, name="critic"):
        """Builds a critic network (Q-value function)."""
        state_inputs = self._get_inputs()
        action_input = Input(shape=(self.action_dim,), dtype=tf.float32, name='action_input')

        common_features = self._build_common_feature_extractor(state_inputs)

        # Process action input separately
        action_features = Dense(64, activation='relu', kernel_regularizer=l2(1e-5), name=f'{name}_action_dense')(action_input)
        action_features = LayerNormalization(name=f'{name}_action_norm')(action_features)

        # Combine state features and action features
        combined_state_action = Concatenate(name=f'{name}_state_action_concat')([
            common_features,
            action_features
        ])

        # Critic-specific dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-5), name=f'{name}_dense_1')(combined_state_action)
        x = LayerNormalization(name=f'{name}_norm_1')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5), name=f'{name}_dense_2')(x)
        x = LayerNormalization(name=f'{name}_norm_2')(x)

        # Output Q-value (single value)
        q_value = Dense(1, activation='linear', kernel_regularizer=l2(1e-5), name=f'{name}_q_value')(x)

        model = Model(inputs=[*list(state_inputs.values()), action_input], outputs=q_value, name=name)
        print(f"\n--- {name} Summary ---")
        model.summary(line_length=120)
        return model

    def _process_state_batch(self, state_dict_batch):
         """Converts a batch of state dictionaries to a list of tensors for the model."""
         key_order = ['shared_features', 'ticker_features', 'portfolio_holdings', 'cash_balance']
         return [state_dict_batch[key] for key in key_order]

    def select_action(self, state, evaluate=False):
        """Selects an action based on the current state."""
        # Preprocess state: convert dict to list/tuple of tensors and add batch dim
        state_processed = {k: tf.expand_dims(tf.convert_to_tensor(v, dtype=tf.float32), axis=0)
                           for k, v in state.items()}
        state_list = self._process_state_batch(state_processed)

        # Get action distribution parameters from actor network
        means, log_stds = self.actor(state_list, training=False) # Ensure training=False for inference

        # During evaluation, return the mean of the distribution (deterministic)
        if evaluate:
            action = tf.tanh(means) # Apply tanh squashing
        # During training, sample from the distribution (stochastic)
        else:
            action, _ = self._sample_actions_and_log_probs(means, log_stds) # Get sampled action

        # Remove batch dimension and convert to numpy
        action_numpy = action.numpy().squeeze(axis=0)

        # Clip actions to be within the action space bounds (safety check)
        action_numpy = np.clip(action_numpy, self.action_space.low, self.action_space.high)

        # Final check for NaNs/Infs
        if np.any(np.isnan(action_numpy)) or np.any(np.isinf(action_numpy)):
             print(f"WARNING: NaN/Inf detected in final action: {action_numpy}. Clipping might be needed earlier.")
             action_numpy = np.nan_to_num(action_numpy, nan=0.0, posinf=1.0, neginf=-1.0) # Replace NaNs/Infs
             action_numpy = np.clip(action_numpy, self.action_space.low, self.action_space.high)


        return action_numpy

    @tf.function
    def _sample_actions_and_log_probs(self, means, log_stds):
        """Samples actions using the reparameterization trick and computes log probabilities."""
        # Clamp log_stds for numerical stability
        log_stds = tf.clip_by_value(log_stds, -20.0, 2.0)
        stds = tf.exp(log_stds)

        # Create the normal distribution
        normal_dist = tfp.distributions.Normal(means, stds)

        # Sample using reparameterization trick: mean + std * noise
        # This allows gradients to flow back through the sampling process
        z = normal_dist.sample() # Shape: (batch, action_dim)

        # Apply tanh squashing to get actions in [-1, 1]
        actions = tf.tanh(z) # Shape: (batch, action_dim)

        # Compute log probability of the *unsquashed* actions (z) under the normal distribution
        log_probs_z = normal_dist.log_prob(z)

        # Adjust log probability for the tanh transformation
        # log_prob(action) = log_prob(z) - sum(log(1 - tanh(z)^2))
        # Need to sum across the action dimension
        log_det_jacobian = tf.reduce_sum(tf.math.log(1.0 - tf.square(actions) + 1e-6), axis=1, keepdims=True) # Add epsilon for stability
        log_probs = tf.reduce_sum(log_probs_z, axis=1, keepdims=True) - log_det_jacobian # Shape: (batch, 1)

        # Check for NaNs/Infs during computation (useful for debugging)
        # tf.debugging.check_numerics(means, "NaN/Inf in means")
        # tf.debugging.check_numerics(log_stds, "NaN/Inf in log_stds")
        # tf.debugging.check_numerics(z, "NaN/Inf in z (sampled)")
        # tf.debugging.check_numerics(actions, "NaN/Inf in actions (tanh)")
        # tf.debugging.check_numerics(log_probs, "NaN/Inf in log_probs")

        return actions, log_probs

    @tf.function 
    def train_step(self, batch):
        """Performs a single training step (optimize critics, actor, and alpha)."""
        states, actions, rewards, next_states, dones = batch

        # Prepare inputs for the models (convert dicts to lists)
        states_list = self._process_state_batch(states)
        next_states_list = self._process_state_batch(next_states)

        # --- Critic Update ---
        with tf.GradientTape(persistent=True) as tape:
            # Get actions and log-probs for the *next* state from the *current* policy
            next_means, next_log_stds = self.actor(next_states_list, training=True)
            next_actions, next_log_probs = self._sample_actions_and_log_probs(next_means, next_log_stds)

            # Get Q-values for the next state-action pairs from the *target* critics
            target_q1_values = self.target_critic_1(next_states_list + [next_actions], training=False)
            target_q2_values = self.target_critic_2(next_states_list + [next_actions], training=False)
            target_q_min = tf.minimum(target_q1_values, target_q2_values) # Clipped Double-Q

            # Calculate the target value for the current Q-functions (Bellman backup)
            # Target = r + gamma * (1 - d) * (min_Q_target(s', a') - alpha * log_pi(a'|s'))
            alpha_val = tf.exp(self.log_alpha) # Get current alpha value
            target_q_values = rewards + self.gamma * (1.0 - dones) * (target_q_min - alpha_val * next_log_probs)
            # Stop gradient flow to target values (they are treated as fixed labels)
            target_q_values = tf.stop_gradient(target_q_values)

            # Get Q-values for the *current* state-action pairs from the *current* critics
            current_q1_values = self.critic_1(states_list + [actions], training=True)
            current_q2_values = self.critic_2(states_list + [actions], training=True)

            # Calculate critic losses (Mean Squared Bellman Error)
            critic_1_loss = tf.reduce_mean(tf.square(current_q1_values - target_q_values))
            critic_2_loss = tf.reduce_mean(tf.square(current_q2_values - target_q_values))
            total_critic_loss = critic_1_loss + critic_2_loss

        # Compute and apply gradients for critics
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))


        # --- Actor Update ---
        with tf.GradientTape() as tape:
            # Get actions and log-probs for the *current* state from the *current* policy
            means, log_stds = self.actor(states_list, training=True)
            sampled_actions, log_probs = self._sample_actions_and_log_probs(means, log_stds)

            # Get Q-values for the *current* state and *newly sampled* actions from the *current* critics
            q1_values_new = self.critic_1(states_list + [sampled_actions], training=False) # Use critics non-trainable here
            q2_values_new = self.critic_2(states_list + [sampled_actions], training=False)
            q_min_new = tf.minimum(q1_values_new, q2_values_new)

            # Calculate actor loss: E[alpha * log_pi(a|s) - min_Q(s, a)]
            # maximize Q - alpha * log_pi, so minimize alpha * log_pi - Q
            alpha_val = tf.exp(self.log_alpha) # Use current alpha
            actor_loss = tf.reduce_mean(alpha_val * log_probs - q_min_new)

        # Compute and apply gradients for actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # --- Alpha (Temperature) Update ---
        with tf.GradientTape() as tape:
            # Need log_probs from the actor update step (calculated above)
            # Loss = E[-alpha * (log_pi(a|s) + target_entropy)]
            # Adjust Alpha so that E[log_pi] approaches -target_entropy
            alpha_val = tf.exp(self.log_alpha) # Current alpha
            # log_probs needs to be associated with the actions sampled *during the actor update*
            # Note: log_probs were calculated based on the policy *before* the actor gradient step.
            alpha_loss = -tf.reduce_mean(alpha_val * (log_probs + self.target_entropy))
            # Need gradients w.r.t. log_alpha, not alpha itself
            alpha_loss_wrt_log_alpha = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy))


        # Compute and apply gradients for log_alpha
        # Use alpha_loss_wrt_log_alpha for gradient calculation w.r.t log_alpha
        alpha_grads = tape.gradient(alpha_loss_wrt_log_alpha, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # Update the detached alpha variable after optimizer step
        self.alpha.assign(tf.exp(self.log_alpha))

        # --- Soft Update Target Networks ---
        self._update_targets()

        # Increment step counter
        self.step_counter.assign_add(1)

        # Return metrics for logging
        return {
            'actor_loss': actor_loss,
            'critic_1_loss': critic_1_loss,
            'critic_2_loss': critic_2_loss,
            'alpha_loss': alpha_loss,
            'alpha': tf.exp(self.log_alpha), # Current alpha value
            'mean_q_value': tf.reduce_mean(q_min_new), # Avg Q value estimated by actor
            'mean_log_prob': tf.reduce_mean(log_probs) # Avg log prob of sampled actions
        }

    def _update_targets(self):
        """Performs soft update of target critic networks."""
        # Update Target Critic 1
        for target_w, source_w in zip(self.target_critic_1.trainable_variables, self.critic_1.trainable_variables):
            target_w.assign(self.tau * source_w + (1.0 - self.tau) * target_w)
        # Update Target Critic 2
        for target_w, source_w in zip(self.target_critic_2.trainable_variables, self.critic_2.trainable_variables):
            target_w.assign(self.tau * source_w + (1.0 - self.tau) * target_w)

    def update(self, batch_size):
        """Samples from buffer and performs one training step."""
        # Check if buffer has enough samples to start training
        if len(self.replay_buffer) < self.min_replay_size_to_train:
            return None # Not ready to train yet

        # Sample a batch
        try:
            batch = self.replay_buffer.sample(batch_size)
        except ValueError as e:
            print(f"Skipping update: {e}") # Handle case where sampling fails (e.g., not enough data)
            return None

        # Perform the training step
        metrics = self.train_step(batch)

        # Log metrics to TensorBoard
        with self.summary_writer.as_default(step=self.step_counter.numpy()):
             for name, value in metrics.items():
                  tf.summary.scalar(f'Training/{name}', value)
        self.summary_writer.flush() # Ensure logs are written

        return metrics # Return metrics dictionary


    def save_experience(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)


    def save(self, path_base):
        """Saves the agent's state (networks and parameters)."""
        try:
             dir_name = os.path.dirname(path_base)
             if dir_name:
                 os.makedirs(dir_name, exist_ok=True)

             print(f"Saving agent networks to {path_base}...")
             # Save networks
             self.actor.save(f"{path_base}_actor.keras")
             self.critic_1.save(f"{path_base}_critic1.keras")
             self.critic_2.save(f"{path_base}_critic2.keras")
             self.target_critic_1.save(f"{path_base}_target_critic1.keras")
             self.target_critic_2.save(f"{path_base}_target_critic2.keras")

             print(f"Saving agent parameters to {path_base}_params.pkl...")
             # Save parameters 
             params_to_save = {
                 "step_counter": self.step_counter.numpy(),
                 "log_alpha": self.log_alpha.numpy(),
                 "target_entropy": self.target_entropy,
                 # --- Replay Buffer Saving (Optional - can make files huge) ---
                 "replay_buffer_states": self.replay_buffer.states,
                 "replay_buffer_actions": self.replay_buffer.actions,
                 "replay_buffer_rewards": self.replay_buffer.rewards,
                 "replay_buffer_next_states": self.replay_buffer.next_states,
                 "replay_buffer_dones": self.replay_buffer.dones,
                 "replay_buffer_position": self.replay_buffer.position,
                 "replay_buffer_size": self.replay_buffer.size,
                 # ----------------------------------------------------------
             }

             with open(f"{path_base}_params.pkl", "wb") as f:
                 pickle.dump(params_to_save, f)

             print(f"Agent saved successfully to {path_base}")

        except Exception as e:
             print(f"Error saving agent to {path_base}: {e}")
             # raise e


    def load(self, path_base):
        """Loads the agent's state."""
        if not self._check_model_files_exist(path_base):
            raise FileNotFoundError(f"Cannot load agent. Required files not found at base path: {path_base}")

        try:
            print(f"Loading agent networks from {path_base}...")
            # Load networks (requires custom objects like the Lambda layer)
            custom_objects = {'expand_dims_channel': expand_dims_channel}
            self.actor = tf.keras.models.load_model(f"{path_base}_actor.keras", custom_objects=custom_objects,
                safe_mode=False)
            self.critic_1 = tf.keras.models.load_model(f"{path_base}_critic1.keras", custom_objects=custom_objects,
                safe_mode=False)
            self.critic_2 = tf.keras.models.load_model(f"{path_base}_critic2.keras", custom_objects=custom_objects,
                safe_mode=False)
            self.target_critic_1 = tf.keras.models.load_model(f"{path_base}_target_critic1.keras", custom_objects=custom_objects,
                safe_mode=False)
            self.target_critic_2 = tf.keras.models.load_model(f"{path_base}_target_critic2.keras", custom_objects=custom_objects,
                safe_mode=False)

            print(f"Loading agent parameters from {path_base}_params.pkl...")
            # Load parameters
            with open(f"{path_base}_params.pkl", "rb") as f:
                params = pickle.load(f)

            # Restore agent attributes from params
            self.step_counter.assign(params["step_counter"])
            self.log_alpha.assign(params["log_alpha"])
            self.alpha.assign(tf.exp(self.log_alpha)) # Update detached alpha
            self.target_entropy = params["target_entropy"]

            # --- Replay Buffer Loading (Optional) ---
            if "replay_buffer_position" in params:
                 print("Restoring replay buffer state...")
                 # Ensure buffer exists and has the right capacity/structure before loading
                 self.replay_buffer.states = params["replay_buffer_states"]
                 self.replay_buffer.actions = params["replay_buffer_actions"]
                 # ... load other buffer arrays ...
                 self.replay_buffer.position = params["replay_buffer_position"]
                 self.replay_buffer.size = params["replay_buffer_size"]
            else:
                 print("Replay buffer state not found in saved params. Starting with empty buffer.")
            # -----------------------------------------

            print("Initializing optimizers...")
            # Re-initialize optimizers 
            self.actor_optimizer = Adam(learning_rate=self.learning_rate)
            self.critic_1_optimizer = Adam(learning_rate=self.learning_rate)
            self.critic_2_optimizer = Adam(learning_rate=self.learning_rate)
            self.alpha_optimizer = Adam(learning_rate=self.learning_rate)

            print(f"Agent loaded successfully from {path_base}")

        except Exception as e:
            print(f"Error loading agent from {path_base}: {e}")
            raise e # Re-raise exception to indicate loading failure

# --- Evaluation Function ---
def evaluate_agent(agent, env, num_episodes=1):
    """Evaluates the agent's performance deterministically."""
    all_episode_rewards = []
    all_portfolio_values = []

    for i in range(num_episodes):
        state, _ = env.reset() # Get initial state
        done = False
        truncated = False
        episode_reward = 0
        episode_portfolio_values = [env.initial_balance] # Start with initial balance

        while not done and not truncated:
            action = agent.select_action(state, evaluate=True) # Use deterministic actions
            next_state, reward, done, truncated, info = env.step(action)

            state = next_state
            episode_reward += reward
            episode_portfolio_values.append(info.get('portfolio_value', env.initial_balance)) # Get portfolio value from info

        all_episode_rewards.append(episode_reward)
        all_portfolio_values.extend(episode_portfolio_values) # Collect portfolio values over time
        print(f"Evaluation Episode {i+1} Reward: {episode_reward:.2f}, Final Value: {episode_portfolio_values[-1]:.2f}")


    avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
    return avg_reward, all_portfolio_values


# --- Worker Function for Parallel Training ---
def worker_function(worker_id, config, train_data, val_data, tradeable_tickers, feature_tickers, feature_scalers, ticker_scalers, result_queue):
    """
    Function executed by each parallel worker process.

    Args:
        worker_id (int): Unique ID for this worker.
        config (dict): Dictionary containing training configuration.
        train_data (np.ndarray): Training data slice for this worker (can be shared).
        val_data (np.ndarray): Validation data slice (can be shared).
        tradeable_tickers (list): List of tradeable ticker symbols.
        feature_tickers (list): List of feature-only ticker symbols.
        feature_scalers (dict): Dictionary of fitted scalers for features.
        ticker_scalers (dict): Dictionary of fitted scalers for tickers/features.
        result_queue (multiprocessing.Queue): Queue to send results back to main process.
    """
    try:
        tf.config.optimizer.set_jit(False)
        print("XLA JIT Compilation Disabled via tf.config.")
    except Exception as e:
        print(f"Could not disable XLA via tf.config: {e}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for all GPUs *visible to this process*
            for gpu in gpus:
                gpu_index = worker_id % len(gpus) # Assign GPUs round-robin
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Worker {worker_id}] Assigned to GPU {gpu_index}: {gpus[gpu_index].name}")
        except RuntimeError as e:
            print(f"[Worker {worker_id}] Error setting memory growth: {e}")
    else:
         print(f"[Worker {worker_id}] No GPUs detected by TensorFlow.")
    print(f"[Worker {worker_id}] Starting...")
    worker_start_time = time.time()

    # --- Configuration ---
    episodes = config.get('episodes', 100)
    batch_size = config.get('batch_size', 256)
    min_buffer_size = config.get('min_buffer_size', 1000)
    updates_per_step = config.get('updates_per_step', 1)
    save_frequency = config.get('save_frequency', 10)
    learning_rate = config.get('learning_rate', 3e-5)
    initial_balance = config.get('initial_balance', 10000)
    window_size = config.get('window_size', 10)
    base_save_path = config.get('base_save_path', 'models/sac_parallel')
    log_dir_base = config.get('log_dir_base', 'logs/sac_parallel')

    # --- Setup Paths and Seed ---
    worker_save_base = f"{base_save_path}_worker_{worker_id}"
    worker_checkpoint_path = f"{worker_save_base}_checkpoint"
    worker_best_model_path = f"{worker_save_base}_best"
    worker_log_dir = os.path.join(log_dir_base, f"worker_{worker_id}")

    # Set seeds for this specific worker process
    process_seed = config.get('global_seed', 42) + worker_id
    np.random.seed(process_seed)
    random.seed(process_seed)
    tf.random.set_seed(process_seed)
    print(f"[Worker {worker_id}] Seed set to {process_seed}")


    # --- Initialize Environment and Agent ---
    try:
        # Use a separate logger instance per worker if logging is complex
        worker_logger = bot_logger() # Or configure file logging per worker

        print(f"[Worker {worker_id}] Initializing Training Environment...")
        train_env = TradingEnv(train_data, tradeable_tickers, feature_tickers,
                             feature_scalers, ticker_scalers,
                             initial_balance=initial_balance, window_size=window_size,
                             bot_logger=worker_logger, seed=process_seed) # Pass seed

        print(f"[Worker {worker_id}] Initializing Validation Environment...")
        val_env = TradingEnv(val_data, tradeable_tickers, feature_tickers,
                           feature_scalers, ticker_scalers,
                           initial_balance=initial_balance, window_size=window_size,
                           bot_logger=worker_logger, is_validation=True, seed=process_seed + 1000) # Different seed for val env

        print(f"[Worker {worker_id}] Initializing SAC Agent...")
        # Try loading checkpoint specific to this worker, otherwise create new
        agent = SACAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            learning_rate=learning_rate,
            load_path=worker_checkpoint_path if os.path.exists(f"{worker_checkpoint_path}_params.pkl") else None, # Check if params exist
            log_dir_base=worker_log_dir # Pass worker-specific log dir
        )
        agent.min_replay_size_to_train = min_buffer_size

    except Exception as e:
        print(f"[Worker {worker_id}] CRITICAL ERROR during initialization: {e}")
        result_queue.put({'worker_id': worker_id, 'status': 'error', 'message': str(e)})
        return # Terminate worker

    # --- Training Loop ---
    best_val_reward = -np.inf
    has_started_training = len(agent.replay_buffer) >= min_buffer_size # Check if loaded buffer is sufficient

    for episode in range(episodes):
        print(f"[Worker {worker_id}] Starting Episode {episode + 1}/{episodes}")
        episode_start_time = time.time()
        state, _ = train_env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False

        while not done and not truncated:
            # Select action (stochastic during training)
            action = agent.select_action(state, evaluate=False)

            # Execute action in environment
            next_state, reward, done, truncated, info = train_env.step(action)

            # Store experience
            agent.save_experience(state, action, reward, next_state, float(done or truncated)) # Store done/truncated as float

            # Update agent if buffer is ready
            if len(agent.replay_buffer) >= min_buffer_size:
                if not has_started_training:
                     print(f"[Worker {worker_id}] Starting agent updates at episode {episode + 1}, step {episode_steps}.")
                     has_started_training = True
                     best_val_reward = -np.inf # Reset best reward when training starts

                # Perform multiple updates per step if configured
                for _ in range(updates_per_step):
                     metrics = agent.update(batch_size)
                     # Optional: Log detailed metrics per update step if needed

            state = next_state
            episode_reward += reward
            episode_steps += 1

            # Check for termination/truncation
            if done or truncated:
                 break # Exit inner loop

        # --- End of Episode ---
        print(f"[Worker {worker_id}] Episode {episode + 1} finished. Steps: {episode_steps}, Reward: {episode_reward:.2f}")

        # --- Validation ---
        val_reward = -np.inf # Default if no validation occurs
        if has_started_training and len(val_data) > 0:
             print(f"[Worker {worker_id}] Running validation...")
             val_reward, _ = evaluate_agent(agent, val_env) # Evaluate deterministically
             print(f"[Worker {worker_id}] Validation Reward: {val_reward:.2f}")

             # Check if this is the best validation reward so far for *this worker*
             if val_reward > best_val_reward:
                 print(f"[Worker {worker_id}] New best validation reward! Saving model to {worker_best_model_path}")
                 best_val_reward = val_reward
                 agent.save(worker_best_model_path) # Save best model

        # --- Periodic Checkpoint Saving ---
        if has_started_training and (episode + 1) % save_frequency == 0:
            print(f"[Worker {worker_id}] Saving checkpoint at episode {episode + 1}...")
            agent.save(worker_checkpoint_path)

        episode_duration = time.time() - episode_start_time
        print(f"[Worker {worker_id}] Episode {episode + 1} duration: {episode_duration:.2f}s")


    # --- End of Training for Worker ---
    worker_duration = time.time() - worker_start_time
    print(f"[Worker {worker_id}] Training complete. Total time: {worker_duration:.2f}s")

    # Final save of the checkpoint
    if has_started_training:
        agent.save(worker_checkpoint_path)

    # Send final results back to the main process
    result_queue.put({
        'worker_id': worker_id,
        'status': 'completed',
        'best_validation_reward': best_val_reward,
        'total_steps': agent.step_counter.numpy(),
        'final_buffer_size': len(agent.replay_buffer),
        'training_duration': worker_duration
    })
    print(f"[Worker {worker_id}] Finished.")


# --- Main Training Function (Modified for Parallelism) ---
def train_sac_parallel(num_workers=4):
    """
    Main function to set up and run parallel SAC training.

    Args:
        num_workers (int): Number of parallel worker processes to launch.
    """
    print("--- Starting Parallel SAC Training ---")
    print(f"Number of workers: {num_workers}")

    # --- Global Config ---
    global_seed = 42
    np.random.seed(global_seed)
    random.seed(global_seed)
    tf.random.set_seed(global_seed)
    # Enable mixed precision if desired and supported
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Using mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")


    # --- Data Loading and Preprocessing ---
    tradeable_tickers = ["TQQQ", "SQQQ", "QQQ"]
    feature_tickers = ["^IXIC", "^GSPC", "^DJI"]
    tickers = feature_tickers + tradeable_tickers
    start_date = '2010-10-18'
    end_date = '2025-04-18'

    main_logger = bot_logger() # Logger for the main process

    try:
        finnhub_client = finnhub.Client(api_key='coavp0pr01qro9kpjlb0coavp0pr01qro9kpjlbg') 
    except Exception as e:
        print(f"Warning: Finnhub client init failed: {e}. Using dummy data.")
        finnhub_client = None 

    sentiment_analyzer = SentimentAnalyzer()
    news_fetcher = NewsFetcher(finnhub_client, sentiment_analyzer, start_date, end_date, main_logger)

    print("Fetching market data...")
    market_data_collector = MarketDataCollector(start_date, end_date, main_logger, tickers)
    economic_indicators, news_data, stock_data, actual_end_date = market_data_collector.fetch_data(tickers)

    print("Processing data...")
    data_processor = DataProcessor(start_date, actual_end_date, main_logger, tickers)
    data_3d = data_processor.process_data(economic_indicators, stock_data, news_data)

    print("Preprocessing data and getting scalers...")
    preprocessor = DataPreprocessor()
    data_processed = preprocessor.fit_transform(data_3d)
    feature_scalers = preprocessor.feature_scalers
    ticker_scalers = preprocessor.ticker_scalers
    preprocessor.save_scalers("my_scalers_parallel.pkl") 


    # --- Data Splitting ---
    total_steps = len(data_processed)
    if total_steps < 20: # Need enough data for train/val split and window size
         raise ValueError(f"Not enough processed data steps ({total_steps}) for training/validation.")

    train_size = int(0.8 * total_steps)
    train_data = data_processed[:train_size]
    val_data = data_processed[train_size:]

    print(f"Total data steps: {total_steps}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    if len(val_data) == 0:
        print("Warning: No validation data after split.")


    # --- Worker Configuration ---
    config = {
        'episodes': 50, # Reduce episodes for quicker testing
        'batch_size': 256, # Adjust based on memory
        'min_buffer_size': 2000, # Minimum buffer size to start training
        'updates_per_step': 1,
        'save_frequency': 10, # Save frequency
        'learning_rate': 5e-5, 
        'initial_balance': 10000,
        'window_size': 7,
        'base_save_path': 'models/sac_parallel_run', # Base path for this run
        'log_dir_base': 'logs/sac_parallel_run', # Base log dir for this run
        'global_seed': global_seed,
        # Add other env/agent params if needed
    }

    # --- Process Management ---
    result_queue = multiprocessing.Queue()
    processes = []

    print(f"\nLaunching {num_workers} worker processes...")
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker_function,
            args=(
                i, config, train_data, val_data,
                tradeable_tickers, feature_tickers,
                feature_scalers, ticker_scalers,
                result_queue
            )
        )
        processes.append(p)
        p.start()
        print(f" Started worker {i} (PID: {p.pid})")
        time.sleep(1) # Stagger start slightly

    # --- Wait for Workers and Collect Results ---
    results = []
    print("\nWaiting for workers to complete...")
    # Wait for processes to finish
    for i, p in enumerate(processes):
        p.join() # Wait for this process to terminate
        print(f" Worker {i} (PID: {p.pid}) finished.")


    print("\nCollecting results from workers...")
    while not result_queue.empty():
        results.append(result_queue.get())

    # --- Process Results ---
    print("\n--- Training Summary ---")
    successful_workers = 0
    failed_workers = 0
    best_overall_reward = -np.inf

    if not results:
         print("No results received from workers.")
    else:
        for result in results:
            worker_id = result.get('worker_id', 'N/A')
            status = result.get('status', 'unknown')
            print(f"\nWorker {worker_id}: Status = {status}")
            if status == 'completed':
                successful_workers += 1
                val_reward = result.get('best_validation_reward', -np.inf)
                print(f"  Best Validation Reward: {val_reward:.4f}")
                print(f"  Total Agent Steps: {result.get('total_steps', 'N/A')}")
                print(f"  Final Buffer Size: {result.get('final_buffer_size', 'N/A')}")
                print(f"  Training Duration: {result.get('training_duration', 0):.2f}s")
                best_overall_reward = max(best_overall_reward, val_reward)
            else:
                failed_workers += 1
                print(f"  Error Message: {result.get('message', 'No message')}")

        print("\n--- Overall Results ---")
        print(f"Successful workers: {successful_workers}")
        print(f"Failed workers: {failed_workers}")
        print(f"Best validation reward across all workers: {best_overall_reward:.4f}")

    print("\n--- Parallel Training Finished ---")
    return results # Return collected results


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Set start method for multiprocessing (important on some OS)
    # 'fork' is default on Unix, 'spawn' might be needed on Windows/macOS
    try:
        multiprocessing.set_start_method('spawn', force=True) # Use 'spawn' for better cross-platform compatibility
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or cannot be changed.")


    # Run the parallel training
    num_parallel_agents = 8 # Set the number of agents to run in parallel
    training_results = train_sac_parallel(num_workers=num_parallel_agents)

    # You can further analyze training_results here
    print("\nFull Results Dictionary:")
    print(training_results)

