
# Standard library imports
import csv
import os
import signal
import sys
import logging
import traceback
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  

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
import ctypes
# load the default libcudnn.so
_lib = ctypes.cdll.LoadLibrary("libcudnn.so")
version = _lib.cudnnGetVersion()
print("cudnnGetVersion() →", version)   # 9300 means 9.3.0


# Deep Learning: TensorFlow/Keras
import tensorflow as tf 
info = tf.sysconfig.get_build_info()
print("cuDNN version:", info["cudnn_version"])
print(tf.sysconfig.get_build_info()) 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs detected.")
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
# Set global policy for mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# mixed_precision.set_global_policy('mixed_bfloat16')
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
    def __init__(self, start_date, end_date, bot_logger):
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
    def __init__(self, start_date, end_date, bot_logger):
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
    metadata = {'render.modes': ['human']}

    def __init__(self, data, tradeable_tickers, feature_tickers, feature_scalers, ticker_scalers, initial_balance=10000, transaction_cost=0.001, window_size=10, 
                 bot_logger=None, entropy_scale=0.1, is_validation=False):
        super(TradingEnv, self).__init__()

        self.data = data
        self.feature_scalers = feature_scalers
        self.ticker_scalers = ticker_scalers
        self.price_indices = [4, 5, 6, 7]  # Indices of price features
        self.open_price_index = 4  # index for 'open'
        self.close_price_index = 7  # Index for Close price
        self.num_dates, self.num_tickers, self.num_features = data.shape
        self.tradeable_tickers = tradeable_tickers
        self.feature_tickers = feature_tickers
        self.num_tradeable = len(tradeable_tickers)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.epsilon = 1e-9
        
        # -----------------------
        # We'll start current_step at 1 so that Day 0 data 
        # is never traded on, but always used to inform Day 1 trades.
        # -----------------------
        self.current_step = 1  

        self.bot_logger = bot_logger
        self.window_size = window_size  # Number of previous steps to include in the observation
        self.entropy_scale = entropy_scale  # Scale factor for entropy bonus
        self.is_validation = is_validation

        # Indices for shared features
        self.economic_indicator_indices = [0, 1, 2, 3]  # Indices for GDP, Unemployment, Inflation, VIX
        self.news_sentiment_index = 9  # Index for news sentiment
        self.time_feature_indices = list(range(18, 26))  # Indices for time features

        # After preprocessing, 'is_weekend' and 'is_holiday' are among the time features
        self.is_weekend_index = 19
        self.is_holiday_index = 26

        # Compute is_trading_day array
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
        self.shared_feature_indices = (
            self.economic_indicator_indices +
            self.time_feature_indices
        )

        # Indices for ticker-specific features (excluding shared features)
        self.ticker_feature_indices = [
            i for i in range(self.num_features) if i not in self.shared_feature_indices
        ]

        # Define observation space
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
                low=0, high=np.inf,
                shape=(self.num_tradeable,), dtype=np.float32
            ),
            'cash_balance': spaces.Box(
                low=0, high=np.inf,
                shape=(1,), dtype=np.float32
            )
        }

        self.observation_space = spaces.Dict(obs_spaces)
        
        # Logging
        if self.bot_logger:
            self.bot_logger.record("init", f"Initialized TradingEnv with {self.num_tickers} tickers and {self.num_dates} trading days")
            self.bot_logger.record("init", f"Tradeable tickers: {self.tradeable_tickers}")
            self.bot_logger.record("init", f"Feature tickers: {self.feature_tickers}")

        # Add debugging print statements
        print(f"Data shape: {self.data.shape}")
        print(f"Is weekend index: {self.is_weekend_index}")
        print(f"Is holiday index: {self.is_holiday_index}")
        print(f"Close index: {self.close_price_index}")
        print(f"Is trading day array shape: {self.is_trading_day.shape}")
        print(f"Number of trading days: {np.sum(self.is_trading_day)}")
        print(f"First 20 is_trading_day values: {self.is_trading_day[:20]}")

        self.unscaled_open_prices = np.zeros((self.num_dates, self.num_tradeable))
        for date_idx in range(self.num_dates):
            for i in range(self.num_tradeable):
                ticker_idx = len(self.feature_tickers) + i
                scaled_open = self.data[date_idx, ticker_idx, self.open_price_index]
                scaler = self.ticker_scalers[ticker_idx][self.open_price_index]  # The scaler used for the 'open' feature
                unscaled_open = scaler.inverse_transform([[scaled_open]])[0, 0]
                self.unscaled_open_prices[date_idx, i] = unscaled_open

        # Precompute unscaled close prices for tradeable tickers
        self.unscaled_close_prices = np.zeros((self.num_dates, self.num_tradeable))
        for date_idx in range(self.num_dates):
            for i in range(self.num_tradeable):
                ticker_idx = len(self.feature_tickers) + i
                scaled_price = self.data[date_idx, ticker_idx, self.close_price_index]
                scaler = self.ticker_scalers[ticker_idx][self.close_price_index]
                unscaled_price = scaler.inverse_transform([[scaled_price]])[0, 0]
                self.unscaled_close_prices[date_idx, i] = unscaled_price

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.num_tradeable)
        
        # -----------------------
        # Shift initial step to 1
        # so that the day-0 data is used 
        # (but no trades happen on day 0).
        # -----------------------
        self.current_step = 1

        if self.current_step >= self.num_dates:
            raise ValueError("No trading days available in the data.")

        self.trade_history = []
        self.position_open_time = np.full(self.num_tradeable, -1)
        self.day_trade_history = deque(maxlen=5)
        self.day_trades_today = 0

        if self.bot_logger:
            self.bot_logger.record("Environment Reset", f"Environment reset. Initial balance: {self.initial_balance}")

        return self._get_observation()
    
    # Define a Numba-compiled helper to compute shared and ticker windows
    @staticmethod
    @njit(parallel=True)
    def compute_observation(data, shared_indices, shared_today_flags, ticker_indices, ticker_today_flags, start_step, num_days, num_tradeable, pad_size, feature_tickers_len):
        # data shape: (num_dates, total_tickers, num_features)
        num_shared = shared_indices.shape[0]
        num_ticker = ticker_indices.shape[0]
        
        # Create output arrays for the window (without padding)
        shared_window = np.zeros((num_days, num_shared), dtype=np.float32)
        ticker_window = np.zeros((num_days, num_tradeable, num_ticker), dtype=np.float32)
        
        for offset in prange(num_days):
            day = start_step + offset
            # Use previous day if needed (ensure day_y is at least 0)
            day_y = day - 1 if day - 1 >= 0 else 0
            # Process shared features (assumed to be taken from index 0 in ticker axis)
            for j in range(num_shared):
                if shared_today_flags[j]:
                    shared_window[offset, j] = data[day, 0, shared_indices[j]]
                else:
                    shared_window[offset, j] = data[day_y, 0, shared_indices[j]]
            # Process ticker-specific features
            for t in range(num_tradeable):
                # t_idx = feature_tickers_len + t
                t_idx = feature_tickers_len + t
                for j in range(num_ticker):
                    if ticker_today_flags[j]:
                        ticker_window[offset, t, j] = data[day, t_idx, ticker_indices[j]]
                    else:
                        ticker_window[offset, t, j] = data[day_y, t_idx, ticker_indices[j]]
        
        # If padding is needed, pad at the beginning by repeating the first row
        if pad_size > 0:
            new_shared = np.empty((pad_size + num_days, num_shared), dtype=np.float32)
            new_ticker = np.empty((pad_size + num_days, num_tradeable, num_ticker), dtype=np.float32)
            for i in range(pad_size):
                for j in range(num_shared):
                    new_shared[i, j] = shared_window[0, j]
                for t in range(num_tradeable):
                    for j in range(num_ticker):
                        new_ticker[i, t, j] = ticker_window[0, t, j]
            # Copy the computed window after the padded rows
            for i in range(num_days):
                for j in range(num_shared):
                    new_shared[pad_size + i, j] = shared_window[i, j]
                for t in range(num_tradeable):
                    for j in range(num_ticker):
                        new_ticker[pad_size + i, t, j] = ticker_window[i, t, j]
            return new_shared, new_ticker
        else:
            return shared_window, ticker_window

    def _get_observation(self):

        TODAY_FEATURES = {
            'gdp': 0,
            'unemployment': 1,
            'inflation': 2,
            'open': 4,
            'day_of_week': 18,
            'is_weekend_index' : 19,
            'is_holiday_index' : 26
        }

        # Convert shared and ticker feature index lists to numpy arrays
        shared_indices = np.array(self.shared_feature_indices, dtype=np.int64)
        ticker_indices = np.array(self.ticker_feature_indices, dtype=np.int64)
        
        # Precompute boolean flags for each feature index
        shared_today_flags = np.empty(shared_indices.shape[0], dtype=np.bool_)
        for i in range(shared_indices.shape[0]):
            if shared_indices[i] in TODAY_FEATURES:
                shared_today_flags[i] = True
            else:
                shared_today_flags[i] = False

        ticker_today_flags = np.empty(ticker_indices.shape[0], dtype=np.bool_)
        for i in range(ticker_indices.shape[0]):
            if ticker_indices[i] in TODAY_FEATURES:
                ticker_today_flags[i] = True
            else:
                ticker_today_flags[i] = False

        # Compute the window parameters
        current_step = self.current_step
        # -----------------------
        # Window ends at current_step (non-inclusive),
        # so for step=1, we only see day 0 data.
        # For step=2, we see day 0..1, etc.
        # -----------------------
        start_step = max(0, current_step - self.window_size)
        num_days = current_step - start_step
        pad_size = self.window_size - num_days if self.window_size > num_days else 0

        # Call the Numba function (assumes self.data is a NumPy array)
        shared_window, ticker_window = TradingEnv.compute_observation(
            self.data,
            shared_indices, shared_today_flags,
            ticker_indices, ticker_today_flags,
            start_step, num_days,
            self.num_tradeable, pad_size, len(self.feature_tickers)
        )

        # ----------------------------------------------
        # Construct and return the final observation dict
        # ----------------------------------------------
        obs = {
            'shared_features': shared_window,
            'ticker_features': ticker_window,
            'portfolio_holdings': self.portfolio.astype(np.float32),
            'cash_balance': np.array([self.balance], dtype=np.float32)
        }
        return obs
    
    def _calculate_reward(self, portfolio_value, prev_portfolio_value):
        """
        Calculates the reward for the current step using vectorized operations.
        """
        # --- Basic Portfolio Return ---
        portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        scaled_portfolio_return = np.clip(portfolio_return * 10, -1, 1) # Scale factor 10 assumed

        # --- Indices ---
        idx = min(self.current_step, self.num_dates - 1)
        prev_idx = max(0, idx - 1)

        # --- Vectorized Holding Rewards Calculation ---
        current_prices = self.unscaled_close_prices[idx, :]  # Prices for all assets at current step
        prev_prices = self.unscaled_close_prices[prev_idx, :] # Prices for all assets at previous step

        # Calculate position values vectorially
        position_values = self.portfolio * current_prices

        # Mask for positions currently held (quantity > 0)
        holding_mask = self.portfolio > 0

        # Calculate returns only for held positions, avoid division by zero
        # Initialize returns as 0
        position_returns = np.zeros_like(self.portfolio, dtype=float)
        # Calculate returns only where prev_price > 0 and quantity > 0
        valid_return_mask = holding_mask & (prev_prices > 0)
        position_returns[valid_return_mask] = (current_prices[valid_return_mask] - prev_prices[valid_return_mask]) / prev_prices[valid_return_mask]

        # Calculate weights only for held positions, avoid division by zero
        # Initialize weights as 0
        position_weights = np.zeros_like(self.portfolio, dtype=float)
        if portfolio_value > 0:
            # Calculate weights only where quantity > 0
            position_weights[holding_mask] = position_values[holding_mask] / portfolio_value

        # Calculate weighted holding rewards using the mask
        # Only sum returns for positions that were actually held
        holding_rewards = np.sum(position_returns[holding_mask] * position_weights[holding_mask])
        scaled_holding_rewards = np.clip(holding_rewards * 10, -1, 1) # Scale factor 10 assumed

        # --- Logging (Remains Iterative) ---
        # Log current portfolio state
        self.bot_logger.record("Portfolio State", f"\nCash Balance: ${self.balance:.2f}")
        self.bot_logger.record("Portfolio State", "\nCurrent Holdings:")
        for i, ticker in enumerate(self.tradeable_tickers):
            quantity = self.portfolio[i]
            current_price = current_prices[i] # Use already fetched current_prices
            position_value = position_values[i] # Use already calculated position_values

            # Log detailed position information
            self.bot_logger.record("Position Details",
                f"{ticker}:\n  Shares: {quantity:.6f}\n  Price: ${current_price:.2f}\n  Position Value: ${position_value:.2f}")

            # Log position performance (using already calculated returns)
            if holding_mask[i]: # Check if the position was held
                self.bot_logger.record("Position Performance", f"{ticker} Return: {position_returns[i]:.2%}")

        self.bot_logger.record("Portfolio Summary",
            f"\nTotal Portfolio Value: ${portfolio_value:.2f}"
            f"\nPrevious Portfolio Value: ${prev_portfolio_value:.2f}"
            f"\nReturn: {portfolio_return:.2%}"
            f"\nCash Ratio: {(self.balance / portfolio_value):.2%}" if portfolio_value > 0 else "\nCash Ratio: N/A")

        # --- Sharpe Ratio ---
        sharpe_reward = 0
        # Ensure history has enough data points AND is not empty
        if len(self.trade_history) >= 30:
            # Extract returns more efficiently if trade_history stores them consistently
            recent_returns = np.array([trade['return'] for trade in self.trade_history[-30:]])
            std_dev = np.std(recent_returns)
            if std_dev > 1e-9: # Avoid division by zero or near-zero std dev
                # Annualized Sharpe Ratio (assuming daily returns and 252 trading days)
                sharpe = np.mean(recent_returns) / std_dev * np.sqrt(252)
                sharpe_reward = np.clip(sharpe * 0.1, -1, 1) # Scale factor 0.1 assumed

        # --- Trading Costs ---
        # trading_cost = self.transaction_cost * np.sum(np.abs(delta_shares * current_prices)) 
        # scaled_trading_cost = ...

        # --- Vectorized Concentration Penalty ---
        # Use position_weights already calculated, handle portfolio_value = 0 case implicitly
        # Squaring weights directly from the holding calculation
        concentration_penalty = np.clip(np.sum(np.square(position_weights)), 0, 1) # Clip unnecessary if weights sum to <= 1

        # --- Drawdown Penalty ---
        # Ensure history is not empty before accessing potentially non-existent keys
        if self.trade_history:
            rolling_max = max([trade.get('portfolio_value', self.initial_balance)
                            for trade in self.trade_history[-30:]], default=self.initial_balance)
        else:
            rolling_max = self.initial_balance # Use initial balance if history is empty

        drawdown = 0
        if rolling_max > 0: # Avoid division by zero
            drawdown = np.clip(max(0, (rolling_max - portfolio_value) / rolling_max), 0, 1)
        drawdown_penalty = drawdown * 0.05 # Scale factor 0.05 assumed

        # --- Combine Reward Components ---
        reward = (
            scaled_portfolio_return * 0.35 +    # Primary objective
            scaled_holding_rewards * 0.20 +     # Reward for good position management
            # sharpe_reward * 0.15 +            # Risk-adjusted returns (uncomment if used)
            # -scaled_trading_cost * 0.10 +     # Trading efficiency (uncomment if used)
            # -concentration_penalty * 0.10 +   # Portfolio diversification (uncomment if used)
            -drawdown_penalty                   # Risk management
        )

        # Final scaling
        reward = np.clip(reward, -1, 1)

        # --- Store Trade History ---
        self.trade_history.append({
            'step': self.current_step, # Good practice to store step number
            'return': portfolio_return,
            'portfolio_value': portfolio_value,
            'cash_balance': self.balance,
            # Store holdings efficiently if needed, maybe just the portfolio array
            'holdings_vector': self.portfolio.copy(),
            # Or keep the dictionary if preferred for readability later
            # 'holdings': {ticker: qty for ticker, qty in zip(self.tradeable_tickers, self.portfolio)},
            'holding_rewards': holding_rewards, # Store the raw value
            'sharpe_reward': sharpe_reward,
            # 'trading_cost': trading_cost, # Store raw value (uncomment if used)
            # 'concentration_penalty': concentration_penalty, # Store raw value (uncomment if used)
            'drawdown_penalty': drawdown_penalty # Store raw value
        })

        # --- Log Reward Calculation ---
        self.bot_logger.record("Reward Components",
            f"\nReward Breakdown:"
            f"\n  Scaled Portfolio Return: {scaled_portfolio_return:.4f} (raw: {portfolio_return:.4f}, weight: 0.35)"
            f"\n  Scaled Holding Rewards: {scaled_holding_rewards:.4f} (raw: {holding_rewards:.4f}, weight: 0.20)"
            # f"\n  Sharpe Reward: {sharpe_reward:.4f} (weight: 0.15)" # Uncomment if used
            # f"\n  Trading Cost Penalty: {-scaled_trading_cost:.4f} (raw: {trading_cost:.4f}, weight: 0.10)" # Uncomment if used
            # f"\n  Concentration Penalty: {-concentration_penalty:.4f} (raw: {concentration_penalty:.4f}, weight: 0.10)" # Uncomment if used
            f"\n  Drawdown Penalty: {-drawdown_penalty:.4f} (raw: {drawdown:.4f})"
            f"\n  Final Reward (Clipped): {reward:.4f}")

        return reward

    def step(self, action):
        assert isinstance(action, np.ndarray)
        assert action.shape == (self.num_tradeable,)
        assert np.all(np.abs(action) <= 1.0)

        self.last_action = action
        
        # Portfolio value before trades
        prev_portfolio_value = self._get_portfolio_value()

        # If it's a valid trading day, execute trades
        if self.current_step < self.num_dates and self.is_trading_day[self.current_step]:
            self._execute_trades(action)

        # Move to next day
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= self.num_dates - 1:
            print("Reached end of data")
            done = True
        else:
            done = False

        obs = self._get_observation()
        portfolio_value = self._get_portfolio_value()

        reward = self._calculate_reward(portfolio_value, prev_portfolio_value)
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'holdings': self.portfolio.copy()
        }

        if self.bot_logger:
            self.bot_logger.record("Step", 
                f"Step: {self.current_step}, Reward: {reward}, Portfolio Value: {portfolio_value}, Balance: {self.balance}")

        return obs, reward, done, info

    def _execute_trades(self, actions):
        """
        Vectorized execution of trades based on continuous actions.
        actions: numpy array of shape (num_tradeable,) with values between -1 and 1.
        """
        MIN_ACTION_THRESHOLD = 0.001  # Minimum action magnitude to trigger a trade

        day_index = self.current_step
        if day_index < 0 or day_index >= len(self.unscaled_open_prices):
            self.bot_logger.warning(f"Invalid day_index {day_index}, skipping trade execution.")
            return  # No valid price data available for this index

        # Get open prices for all tradeable tickers for the *current* day
        # Ensure day_index is valid for the price array dimension
        try:
            prices = self.unscaled_open_prices[day_index, :self.num_tradeable] # Ensure we only take tradeable tickers
        except IndexError:
            self.bot_logger.error(f"IndexError accessing prices at day_index {day_index}.")
            return

        actions = np.asarray(actions, dtype=np.float64) # Use float64 for precision
        # Ensure portfolio is also float64
        portfolio = self.portfolio.astype(np.float64)
        tc = self.transaction_cost
        current_balance = self.balance # Work with a copy for calculations within the buy section

        # --- Input Validation ---
        # Check for NaN/Inf in inputs before proceeding
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            self.bot_logger.error(f"NaN or Inf detected in input actions: {actions}. Skipping trade.")
            return
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
             self.bot_logger.error(f"NaN or Inf detected in prices: {prices} at step {day_index}. Skipping trade.")
             # Potentially handle this more gracefully, e.g., skip trades for affected tickers
             return # Or apply a mask
        if np.isnan(current_balance) or np.isinf(current_balance):
             self.bot_logger.error(f"NaN or Inf detected in balance: {current_balance}. Skipping trade.")
             return
        if np.any(np.isnan(portfolio)) or np.any(np.isinf(portfolio)):
             self.bot_logger.error(f"NaN or Inf detected in portfolio: {portfolio}. Skipping trade.")
             return

        # Identify valid trades (only if price > epsilon and action is sufficiently large)
        valid_price_mask = prices > self.epsilon
        trade_mask = valid_price_mask & (np.abs(actions) >= MIN_ACTION_THRESHOLD)

        # Separate sell and buy orders
        sell_mask = trade_mask & (actions < 0)
        buy_mask = trade_mask & (actions > 0)

        # --- Process Sell Orders ---
        if np.any(sell_mask):
            # For sells, compute the sell fraction and quantity per ticker.
            sell_fraction = np.abs(actions[sell_mask])
            # Multiply by current holding to get the desired quantity to sell
            # Ensure only sells what is in the portfolio
            sell_quantity = np.minimum(portfolio[sell_mask],
                                       np.round(sell_fraction * portfolio[sell_mask], 8))

            # Filter out zero-quantity sells
            non_zero_sell_mask = sell_quantity > self.epsilon
            sell_indices_local = np.where(non_zero_sell_mask)[0] # Indices within the sell_mask slice

            if sell_indices_local.size > 0:
                # Map local indices back to original indices within the sell_mask view
                original_sell_indices = np.where(sell_mask)[0][sell_indices_local]

                # Calculate proceeds only for actual sells
                actual_sell_quantity = sell_quantity[non_zero_sell_mask]
                actual_sell_prices = prices[sell_mask][non_zero_sell_mask]
                proceeds = actual_sell_quantity * actual_sell_prices * (1 - tc)

                # Update global cash and portfolio vectorially
                total_proceeds = np.sum(proceeds)
                self.balance += total_proceeds
                portfolio[original_sell_indices] -= actual_sell_quantity # Update the main portfolio array

                # Log each sell trade
                for i, global_idx in enumerate(original_sell_indices):
                    ticker_symbol = self.tradeable_tickers[global_idx]
                    qty = actual_sell_quantity[i]
                    price_val = actual_sell_prices[i]
                    proc = proceeds[i]
                    self.bot_logger.record(
                        "Sell Trade", f"Sold {qty:.4f} shares of {ticker_symbol} at {price_val:.2f}. Proceeds: {proc:.2f}")

        # --- Process Buy Orders ---
        # Use the potentially updated balance after sells
        current_balance = self.balance
        if np.any(buy_mask) and current_balance > self.epsilon:
            desired_buy_fraction = actions[buy_mask]  # positive values
            current_prices_buy = prices[buy_mask]
            price_plus_cost = current_prices_buy * (1 + tc) # Cost per share including tc

            # Calculate allocation based on action fractions of available balance
            total_positive_action = np.sum(desired_buy_fraction)

            # Initialize actual buy quantities and costs
            buy_quantity = np.zeros_like(desired_buy_fraction, dtype=np.float64)
            cost = np.zeros_like(desired_buy_fraction, dtype=np.float64)

            if total_positive_action > self.epsilon:
                # Allocate available balance proportionally to action strength
                allocation_fraction = desired_buy_fraction / total_positive_action
                target_cash_allocation = current_balance * allocation_fraction

                # Calculate desired quantity based on allocated cash (for valid prices)
                # price_plus_cost should be > EPSILON because buy_mask depends on valid_price_mask
                desired_buy_quantity_raw = target_cash_allocation / price_plus_cost
                # Round desired quantity (or do it later)
                desired_buy_quantity = np.round(desired_buy_quantity_raw, 8)

                # Calculate the cost of these desired quantities
                desired_cost = desired_buy_quantity * price_plus_cost
                total_desired_cost = np.sum(desired_cost)

                # --- Scaling Logic ---
                # Proceed only if it desires to spend *something*
                if total_desired_cost > self.epsilon:
                    scale = 1.0
                    if total_desired_cost > current_balance:
                        # Scale down proportionally - SAFE DIVISION ASSURED HERE
                        scale = current_balance / total_desired_cost

                    # Calculate final buy quantity after scaling
                    # Ensure it only scales quantities > 0 to avoid potential 0 * large_scale issues if logic changes
                    buy_mask_scaled = desired_buy_quantity > self.epsilon
                    # Apply scale only where desired_buy_quantity > 0
                    buy_quantity[buy_mask_scaled] = np.round(desired_buy_quantity[buy_mask_scaled] * scale, 8)

                    # Recalculate actual cost based on final scaled quantity
                    cost[buy_mask_scaled] = buy_quantity[buy_mask_scaled] * price_plus_cost[buy_mask_scaled]

                    # Final check for safety - ensure cost does not exceed balance due to float errors
                    total_actual_cost = np.sum(cost)
                    if total_actual_cost > current_balance:
                        if total_actual_cost > self.epsilon: # Avoid division by zero if cost became ~0
                            final_scale = (current_balance / total_actual_cost) * 0.9999 # Slightly less than 1 to be safe
                            buy_quantity *= final_scale
                            cost *= final_scale
                            total_actual_cost = np.sum(cost)
                        else:
                           # Cost is effectively zero, ensure quantities are zero too
                           buy_quantity.fill(0)
                           cost.fill(0)
                           total_actual_cost = 0.0


                    # --- Update State & Log ---
                    if total_actual_cost > self.epsilon:
                         self.balance -= total_actual_cost

                         # Get global indices corresponding to the buy_mask
                         buy_indices_global = np.where(buy_mask)[0]

                         # Update the main portfolio array
                         portfolio[buy_indices_global] += buy_quantity # buy_quantity is already filtered/scaled

                         # Log actual buys made (where buy_quantity > epsilon)
                         actual_buy_indices_local = np.where(buy_quantity > self.epsilon)[0]
                         for local_idx in actual_buy_indices_local:
                             global_idx = buy_indices_global[local_idx] # Map local index back to original
                             ticker_symbol = self.tradeable_tickers[global_idx]
                             qty = buy_quantity[local_idx]
                             price_val = current_prices_buy[local_idx] # Use price from the buy slice
                             c = cost[local_idx]                       # Use cost calculated for the buy slice
                             self.bot_logger.record(
                                "Buy Trade", f"Bought {qty:.4f} shares of {ticker_symbol} at {price_val:.2f}. Cost: {c:.2f}")

            # Else (total_positive_action is zero): buy_quantity and cost remain zero.

        # Update the master portfolio state. Ensure no NaNs
        self.portfolio = np.nan_to_num(portfolio) # Replace NaN with 0, Inf with large number
        self.balance = np.nan_to_num(self.balance)

        # Optional: Add assertions to catch issues early
        assert np.all(np.isfinite(self.portfolio)), f"NaN/Inf in portfolio after trade: {self.portfolio}"
        assert np.isfinite(self.balance), f"NaN/Inf in balance after trade: {self.balance}"
        assert self.balance >= -self.epsilon, f"Balance negative after trade: {self.balance}" # Allow slightly below zero for float errors

    def _get_portfolio_value(self):
        idx = min(self.current_step, self.num_dates - 1)
        prices = self.unscaled_close_prices[idx] # Shape: (num_tradeable,)
        # Perform element-wise multiplication and sum
        holdings_value = np.sum(self.portfolio * prices)
        portfolio_value = self.balance + holdings_value
        if self.bot_logger:
            self.bot_logger.record("Portfolio", f"Calculating portfolio value at day index = {idx}")
            # Log individual holdings if needed 
            for i in range(self.num_tradeable):
                # Log price and quantity for each ticker
                self.bot_logger.record("Portfolio", f"  Ticker {self.tradeable_tickers[i]}: "
                                                    f"Price={prices[i]:.2f}, Quantity={self.portfolio[i]:.6f}, "
                                                    f"Value={self.portfolio[i] * prices[i]:.2f}")
            # Log cash balance and total calculated value
            self.bot_logger.record("Portfolio", f"  Cash Balance: ${self.balance:.2f}")
            self.bot_logger.record("Portfolio", f"  Holdings Value: ${holdings_value:.2f}")
            self.bot_logger.record("Portfolio", f"Total portfolio value: ${portfolio_value:.2f}")
        return portfolio_value

    def render(self, mode='human'):
        portfolio_value = self._get_portfolio_value()
        print(f'Step: {self.current_step}, Portfolio Value: {portfolio_value}')
        if self.bot_logger:
            self.bot_logger.record("Render", f'Step: {self.current_step}, Portfolio Value: {portfolio_value}')

class SACReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim): 
        
        self.capacity = capacity
        self.position = 0
        
        # Initialize arrays with correct keys matching the model's expected input
        self.states = {
            'shared_features': np.zeros((capacity, *state_shape['shared_features'].shape), dtype=np.float32),
            'ticker_features': np.zeros((capacity, *state_shape['ticker_features'].shape), dtype=np.float32),
            'portfolio_holdings': np.zeros((capacity, state_shape['portfolio_holdings'].shape[0]), dtype=np.float32),
            'cash_balance': np.zeros((capacity, 1), dtype=np.float32)
        }
        self.next_states = {
            'shared_features': np.zeros((capacity, *state_shape['shared_features'].shape), dtype=np.float32),
            'ticker_features': np.zeros((capacity, *state_shape['ticker_features'].shape), dtype=np.float32),
            'portfolio_holdings': np.zeros((capacity, state_shape['portfolio_holdings'].shape[0]), dtype=np.float32),
            'cash_balance': np.zeros((capacity, 1), dtype=np.float32)
        }

        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.dones = np.zeros((capacity, 1))
        
    def push(self, state, action, reward, next_state, done):
        """Improved storage method"""
        # Store state
        for key in self.states.keys():
            self.states[key][self.position] = state[key]
            self.next_states[key][self.position] = next_state[key]
            
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        max_pos = min(self.position, self.capacity)
        batch_indices = np.random.choice(max_pos, batch_size)
        
        batch_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.states.items()
        }
        
        batch_next_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.next_states.items()
        }
        
        return (
            batch_states,
            tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.float32),
            tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32),
            batch_next_states,
            tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)
        )
        
    def __len__(self):
        return min(self.position, self.capacity)
    
# Define a named function to expand dimensions
@register_keras_serializable()
def expand_dims_channel(x):
    return tf.expand_dims(x, axis=-1)

# SAC Agent
class SACAgent: 
    def __init__(self, observation_space, action_space, load_path=None, learning_rate=3e-5, gamma=0.99, tau=0.005):
        """
        Initialize SAC agent with LSTM layers for sequential processing
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = self.action_space.shape[0]  # Define action dimension
        
        # Get dimensions from spaces
        self.window_size = observation_space['shared_features'].shape[0]
        self.num_shared_features = observation_space['shared_features'].shape[1]
        self.num_ticker_features = observation_space['ticker_features'].shape[2]
        self.num_tradeable = observation_space['portfolio_holdings'].shape[0]

        # Initialize replay buffer
        self.replay_buffer = SACReplayBuffer(
            capacity=int(1e7),
            state_shape=self.observation_space.spaces,
            action_dim=self.action_dim
        )

        # Initialize temperature parameter alpha
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True, name='log_alpha')
        self.alpha = tf.Variable(tf.exp(self.log_alpha), trainable=False, dtype=tf.float32, name='alpha')
        self.target_entropy = -float(self.action_dim)
        
        # Initialize networks
        try:
            if load_path and self._check_model_files_exist(load_path):
                print(f"Loading existing model from {load_path}")
                self.load(load_path)
            else:
                print("Initializing new model")
                self._initialize_new_model()
        except Exception as e:
            print(f"Error during model initialization/loading: {str(e)}")
            print("Falling back to new model initialization")
            self._initialize_new_model()
        
        # Initialize training parameters
        self.min_replay_size = 1000
        self.polyak = 1 - tau
        self.best_val_reward = -np.inf
        self.step_counter = 0

        # Initialize logging metrics
        self.training_metrics = {
            'actor_loss': [],
            'critic_1_loss': [],
            'critic_2_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'avg_reward': [],
            'avg_q_value': []
        }

    def _check_model_files_exist(self, path):
        """Check if all necessary model files exist"""
        required_files = [
            f"{path}_actor.keras",
            f"{path}_critic1.keras",
            f"{path}_critic2.keras",
            f"{path}_target_critic1.keras",
            f"{path}_target_critic2.keras",
            f"{path}_params.pkl"
        ]
    
        return all(os.path.exists(f) for f in required_files)
    
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
        
    def select_action(self, state, evaluate=False):
        """Updated select_action method to handle input properly"""
        means, log_stds = self.actor(self._process_state(state))

        # Print raw network outputs
        # print(f"Raw means: {means}")
        # print(f"Raw log_stds: {log_stds}")
        
        if evaluate:
            # During evaluation, use mean action
            action = tf.tanh(means)
        else:
            # During training, sample from distribution
            actions, _ = self.sample_actions(means, log_stds)
            action = actions

        # Ensure the shape matches (self.num_tradeable,)
        # Convert to NumPy array and ensure correct shape
        action = action.numpy()
        # print(f"Action after sampling/tanh: {action}")

        if action.shape[0] == 1:
            action = action.squeeze(axis=0)

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print("WARNING: NaN or Inf detected in action!")

        # Clip actions to [-1, 1]
        pre_clip = action.copy()
        action = np.clip(action, -1.0, 1.0)
        
        # Final safety check
        assert np.all(np.abs(action) <= 1.0), f"Actions still outside [-1,1] range: {action}"

        return action
            
    def _process_state(self, state):
        """Add helper method for state preprocessing"""
        # Convert state dict to proper format for network
        shared_features = tf.convert_to_tensor(state['shared_features'], dtype=tf.float32)
        ticker_features = tf.convert_to_tensor(state['ticker_features'], dtype=tf.float32)
        portfolio_holdings = tf.convert_to_tensor(state['portfolio_holdings'], dtype=tf.float32)
        cash_balance = tf.convert_to_tensor(state['cash_balance'], dtype=tf.float32)
        
        # Add batch dimension if necessary
        if len(shared_features.shape) == 2:
            shared_features = tf.expand_dims(shared_features, axis=0)
        if len(ticker_features.shape) == 3:
            ticker_features = tf.expand_dims(ticker_features, axis=0)
        if len(portfolio_holdings.shape) == 1:
            portfolio_holdings = tf.expand_dims(portfolio_holdings, axis=0)
        if len(cash_balance.shape) == 0:
            cash_balance = tf.expand_dims(cash_balance, axis=0)
        
        return [shared_features, ticker_features, portfolio_holdings, cash_balance]

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

    def _process_ticker_features(self, ticker_features):
        """
        Process ticker-specific features through ConvLSTM2D layers to capture both
        temporal and inter-ticker relationships.
        
        Input shape: (batch, window_size, num_tradeable, num_ticker_features)
        Output: A dense representation of the ticker features.
        """
        # 1. Add a channel dimension so that the data shape becomes:
        #    (batch, window_size, num_tradeable, num_ticker_features, 1)
        x = Lambda(expand_dims_channel, name='expand_dims')(ticker_features)       

        # 2. Apply the first ConvLSTM2D layer.
        #    Here, we use a kernel size that spans across the feature dimension.
        #    'return_sequences=True' preserves the time dimension for further processing.
        x = ConvLSTM2D(filters=64,
                    kernel_size=(1, 3),
                    padding='same',
                    return_sequences=True,
                    activation='tanh',
                    name='conv_lstm_1')(x)
        x = LayerNormalization(dtype='float32', name='ticker_norm_1')(x)
        
        # 3. Apply a second ConvLSTM2D layer.
        #    Setting return_sequences=False will collapse the time dimension.
        x = ConvLSTM2D(filters=32,
                    kernel_size=(1, 3),
                    padding='same',
                    return_sequences=False,
                    activation='tanh',
                    name='conv_lstm_2')(x)
        x = LayerNormalization(dtype='float32', name='ticker_norm_2')(x)
        
        # 4. Flatten the output and project it with a Dense layer.
        x = Flatten(name='ticker_flatten')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5), name='ticker_features_output')(x)
        return x

    def _process_portfolio_state(self, portfolio_input):
        """
        Process portfolio state with mixed precision support.
        """
        x = portfolio_input
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-5), name='portfolio_dense')(x)

        return LayerNormalization(dtype='float32', name='portfolio_norm')(x)

    def _build_actor(self):
        """
        Build actor network with mixed precision support.
        Uses a custom loss scaling wrapper for better training stability.
        """
        # Get input layers (keeping inputs as float32 for stability)
        inputs = self._get_inputs()
        
        # Process features with mixed precision
        shared_features = self._process_shared_features(inputs['shared_features'])
        ticker_features = self._process_ticker_features(inputs['ticker_features'])
        portfolio_features = self._process_portfolio_state(inputs['portfolio_holdings'])
        cash_balance = inputs['cash_balance']
        
        # Combine features
        combined = Concatenate(name='feature_concatenation')([
            shared_features,
            ticker_features,
            portfolio_features,
            cash_balance
        ])
        
        # Dense layers with mixed precision
        x = self._build_dense_layers(combined)
        
        # Output layers (keep as float32 for better precision in action space)
        means = Dense(self.num_tradeable, activation='relu', kernel_regularizer=l2(1e-5), dtype='float32', name='action_means')(x)
        log_stds = Dense(self.num_tradeable, activation='relu', dtype='float32', kernel_regularizer=l2(1e-5), name='action_log_stds')(x)
        
        model = Model(inputs=list(inputs.values()), outputs=[means, log_stds], name='actor')
    
        return model

    
    def _build_dense_layers(self, x):
        """
        Build dense layers for policy network with mixed precision support.
        Includes careful normalization for numerical stability.
        """
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-5), name='dense_1')(x)
        x = LayerNormalization(dtype='float32', name='layer_norm_1')(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-5), name='dense_2')(x)
        x = LayerNormalization(dtype='float32', name='layer_norm_2')(x)
        return x

    def _build_critic(self):
        """
        Build critic network with mixed precision support.
        Includes additional normalization layers for stability.
        """
        # Get input layers (keeping inputs as float32 for stability)
        state_inputs = self._get_inputs()
        action_input = Input(shape=(self.num_tradeable,), dtype='float32', name='action_input')
        
        # Process state features with mixed precision
        shared_features = self._process_shared_features(state_inputs['shared_features'])
        tf.print("shared_features:", shared_features)
        ticker_features = self._process_ticker_features(state_inputs['ticker_features'])
        tf.print("ticker_features:", ticker_features)
        portfolio_features = self._process_portfolio_state(state_inputs['portfolio_holdings'])
        tf.print("portfolio_features:", portfolio_features)
        
        # Process action with mixed precision
        action = action_input
        action_features = Dense(64, activation='relu', kernel_regularizer=l2(1e-5), name='action_processing')(action)
        tf.print("action_features:", action_features)
        
        # Combine features
        combined = Concatenate(name='feature_concatenation')([
            shared_features,
            ticker_features,
            portfolio_features,
            action_features,
            state_inputs['cash_balance']  
        ])
        tf.print("combined features:", combined)
        
        # Dense layers with mixed precision
        x = self._build_dense_layers(combined)
        tf.print("after dense layers:", x)
        
        # Output Q-value (keep as float32 for better precision)
        x = LayerNormalization(dtype='float32', name='final_norm')(x)
        tf.print("after final norm:", x)
        q_value = Dense(1, dtype='float32', kernel_regularizer=l2(1e-5), name='q_value')(x)
        tf.print("q_value:", q_value)
        
        model = Model(
            inputs=[*list(state_inputs.values()), action_input],
            outputs=q_value,
            name='critic'
        )
        
        return model

    @tf.function
    def train_step(self, batch):
        """Performs a single training step (optimize critics, actor, and alpha)."""
        states, actions, rewards, next_states, dones = batch

        # Prepare inputs for the models (convert dicts to lists)
        states_list = self._process_state(states)
        next_states_list = self._process_state(next_states)

        # --- Critic Update ---
        with tf.GradientTape(persistent=True) as tape:
            # Get actions and log-probs for the *next* state from the *current* policy
            next_means, next_log_stds = self.actor(next_states_list, training=True)
            next_actions, next_log_probs = self.sample_actions(next_means, next_log_stds)

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
            sampled_actions, log_probs = self.sample_actions(means, log_stds)

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
            # We want alpha to adjust so that E[log_pi] approaches -target_entropy
            alpha_val = tf.exp(self.log_alpha) # Current alpha
            # We need log_probs associated with the actions sampled *during the actor update*
            # Note: log_probs were calculated based on the policy *before* the actor gradient step.
            # This is standard practice.
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
    
    def sample_actions(self, means, log_stds):
        """
        Sample actions using reparameterization trick
        """
        # 1. Input validation and debugging
        tf.debugging.check_numerics(means, message="NaNs detected in actor means before sampling")
        tf.debugging.check_numerics(log_stds, message="NaNs detected in actor log_stds before sampling")

        log_stds = tf.clip_by_value(log_stds, -20.0, 2.0)
        stds = tf.exp(log_stds)
        stds = tf.clip_by_value(stds, 1e-3, 2.0)  # Clip standard deviations for numerical stability
        normal = tfp.distributions.Normal(means, stds)
        x_t = normal.sample()
        tf.debugging.check_numerics(x_t, message="NaNs detected in sampled x_t")
        action = tf.tanh(x_t)
        epsilon = 1e-6
        action = tf.clip_by_value(action, -1.0 + epsilon, 1.0 - epsilon)
        tf.debugging.check_numerics(action, message="NaNs detected in action after tanh")

        # Compute log probability, using the formula for transformed distribution
        log_prob = normal.log_prob(x_t)
        log_prob = tf.reduce_sum(log_prob, axis=1)  #[batch_size]
        epsilon_for_jacobian = 1e-5  # slightly larger epsilon for numerical stability
        log_det_jacobian = tf.reduce_sum(tf.math.log(1.0 - tf.square(action) + epsilon_for_jacobian), axis=1)
        tf.debugging.check_numerics(log_det_jacobian, message="NaNs detected in log_det_jacobian")
        log_prob -= log_det_jacobian
        tf.debugging.check_numerics(log_prob, message="NaNs detected in log_prob after adjustments")
            
        return action, log_prob
    
    def _update_targets(self):
        """
        Soft update target networks
        """
        for target, source in [
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2)
        ]:
            for target_var, source_var in zip(
                target.trainable_variables, source.trainable_variables
            ):
                target_var.assign(
                    self.tau * source_var + (1 - self.tau) * target_var
                )

    def update(self, batch_size):
        """Updated update method with shape debugging"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
            
        try:
            batch = self.replay_buffer.sample(batch_size)
            metrics = self.train_step(batch)
            return metrics
            
        except Exception as e:
            print(f"Error in update method: {str(e)}")
            raise e
        
    def save_experience(self, state, action, reward, next_state, done):
        """Experience storage method"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def adapt_learning_rate(self, progress):
        """Learning rate adaptation"""
        lr = self.learning_rate * (1 - progress)
        self.actor_optimizer.learning_rate.assign(lr)
        self.critic_1_optimizer.learning_rate.assign(lr)
        self.critic_2_optimizer.learning_rate.assign(lr)
    
    def save(self, path):
        """Save the SAC agent's state"""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        print("Saving networks...")
        # Save networks
        self.actor.save(f"{path}_actor.keras")
        self.critic_1.save(f"{path}_critic1.keras")
        self.critic_2.save(f"{path}_critic2.keras")
        self.target_critic_1.save(f"{path}_target_critic1.keras")
        self.target_critic_2.save(f"{path}_target_critic2.keras")

        print("Saving parameters...")
        # Save replay buffer and parameters
        params_to_save = {
            "step_counter": self.step_counter,
            "best_val_reward": self.best_val_reward,
            "log_alpha": self.log_alpha.numpy(),
            "target_entropy": self.target_entropy,
            "replay_buffer_states": self.replay_buffer.states,
            "replay_buffer_actions": self.replay_buffer.actions,
            "replay_buffer_rewards": self.replay_buffer.rewards,
            "replay_buffer_next_states": self.replay_buffer.next_states,
            "replay_buffer_dones": self.replay_buffer.dones,
            "replay_buffer_position": self.replay_buffer.position
        }

        try:
            with open(f"{path}_params.pkl", "wb") as f:
                pickle.dump(params_to_save, f)
            print(f"Agent saved to {path}")
        except Exception as e:
            print(f"Error saving agent parameters: {str(e)}")

    def load(self, path):
        """Load the SAC agent's state"""
        try:
            print("Loading networks...")
            # Load networks
            self.actor = tf.keras.models.load_model(f"{path}_actor.keras")
            self.critic_1 = tf.keras.models.load_model(f"{path}_critic1.keras")
            self.critic_2 = tf.keras.models.load_model(f"{path}_critic2.keras")
            self.target_critic_1 = tf.keras.models.load_model(f"{path}_target_critic1.keras")
            self.target_critic_2 = tf.keras.models.load_model(f"{path}_target_critic2.keras")

            print("Loading parameters...")
            # Load parameters
            with open(f"{path}_params.pkl", "rb") as f:
                params = pickle.load(f)

            if not hasattr(self, 'replay_buffer'):
                self.replay_buffer = SACReplayBuffer(
                    capacity=int(1e7),
                    state_shape=self.observation_space.spaces,
                    action_dim=self.action_dim
                )

            # Restore agent attributes
            self.step_counter = params["step_counter"]
            self.best_val_reward = params["best_val_reward"]
            self.log_alpha.assign(params["log_alpha"])
            self.target_entropy = params["target_entropy"]
            
            print("Restoring replay buffer...")
            # Restore replay buffer
            self.replay_buffer.states = params["replay_buffer_states"]
            self.replay_buffer.actions = params["replay_buffer_actions"]
            self.replay_buffer.rewards = params["replay_buffer_rewards"]
            self.replay_buffer.next_states = params["replay_buffer_next_states"]
            self.replay_buffer.dones = params["replay_buffer_dones"]
            self.replay_buffer.position = params["replay_buffer_position"]

            print("Initializing optimizers...")
            # Initialize optimizers
            self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            self.critic_1_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            self.critic_2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            self.alpha_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            print(f"Successfully loaded agent from {path}")

        except Exception as e:
            print(f"Error loading agent: {str(e)}")
            raise

tradeable_tickers = ["TQQQ", "SQQQ","QQQ"]
feature_tickers = ["^IXIC", "^GSPC", "^DJI"]
tickers = feature_tickers + tradeable_tickers
start_date = '2010-10-18'
end_date = '2025-04-04'

bot_logger = bot_logger()  # Initialize your custom logger

data_processor = DataProcessor(start_date, end_date, bot_logger)

finnhub_client = finnhub.Client(api_key="coavp0pr01qro9kpjlb0coavp0pr01qro9kpjlbg")
# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Initialize the news fetcher
news_fetcher = NewsFetcher(
    finnhub_client=finnhub_client,
    sentiment_analyzer=sentiment_analyzer,
    start_date=start_date,
    end_date=end_date,
    bot_logger=bot_logger
)

# Fetch and preprocess the data
market_data_collector = MarketDataCollector(start_date, end_date, bot_logger)
economic_indicators, news_data, stock_data, actual_end_date = market_data_collector.fetch_data(tickers)

# process the data
data_3d = data_processor.process_data(economic_indicators, stock_data, news_data)

# Preprocess the data
preprocessor = DataPreprocessor()
# Preprocess the data
data_processed = preprocessor.fit_transform(data_3d)
feature_scalers = preprocessor.feature_scalers
ticker_scalers = preprocessor.ticker_scalers

# Set a global seed for overall reproducibility
global_seed = 42
np.random.seed(global_seed)
random.seed(global_seed)
tf.random.set_seed(global_seed)

# Define save path for the agent
SAVE_PATH = "TIT3"  # Base save path
BEST_MODEL_PATH3 = f"{SAVE_PATH}_best3"
CHECKPOINT_PATH3 = f"{SAVE_PATH}_checkpoint3"

def train_sac():
    total_steps = len(data_processed)
    train_size = int(0.8 * total_steps)
    
    train_data = data_processed[:train_size]
    val_data = data_processed[train_size:]

    print(f"Validation data shape: {val_data.shape}")
    print(f"Last day of validation data:")
    print(f"Features for last day: {val_data[-1]}")  # Check if last day has valid data
    
    preprocessor.save_scalers("my_scalers.pkl")

    print(f"Total steps: {total_steps}")
    print(f"Training steps: {len(train_data)}")
    print(f"Validation steps: {len(val_data)}")
    # Initialize TensorBoard logging
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'logs/SAC_{current_time}'
    summary_writer = tf.summary.create_file_writer(log_dir)# Initialize TensorBoard logging

    # Initialize environment and agent
    print("\nInitializing Training Environment:")
    train_env = TradingEnv(train_data, tradeable_tickers, feature_tickers, 
                          feature_scalers, ticker_scalers, 
                          initial_balance=10000,
                          transaction_cost=0.001,
                          window_size=10, 
                          bot_logger=bot_logger,
                          entropy_scale=0.1,
                          is_validation=False)
    
    print("\nInitializing Validation Environment:")
    val_env = TradingEnv(val_data, tradeable_tickers, feature_tickers, 
                         feature_scalers, ticker_scalers,
                         initial_balance=10000,
                         transaction_cost=0.001,
                         window_size=10, 
                         bot_logger=bot_logger,
                         entropy_scale=0.1,
                         is_validation=True)
    
    print(f"Training environment observation space: {train_env.observation_space}")
    print(f"Validation environment observation space: {val_env.observation_space}")
    
    # Try to load existing model, create new one if loading fails
    try:
        print("Attempting to load checkpoint model...")
        agent = SACAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            learning_rate=1e-6,
            load_path=CHECKPOINT_PATH3
        )
    except Exception as e:
        print(f"Could not load checkpoint model: {str(e)}")
        print("Creating new model...")
        agent = SACAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            learning_rate=1e-6,
            load_path=None
        )

    # Training parameters
    episodes = 200
    save_frequency = 10 
    batch_size = 2048
    min_buffer_size = 500
    updates_per_step = 1

    agent.min_replay_size = min_buffer_size

    # --- Profiler settings ---
    profiler_start_step = 100 # Start profiling after some steps
    profiler_duration_steps = 100  # How many steps to profile
    enable_profiling = True  # Flag to enable/disable profiling 
    is_profiling = False  # Track if profiling is active
    profiler_options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 2)

    
    # Early stopping
    patience = 1000
    patience_counter = 0
    best_val_reward = -np.inf
    best_val_performance = None
    has_started_training = False  # Flag to track training

    # Training metrics
    metrics_history = {
        'episode_rewards': [],
        'validation_rewards': [],
        # 'sharpe_ratios': [],
        # 'max_drawdowns': [],
        'actor_loss': [],        
        'critic_1_loss': [],     
        'critic_2_loss': [],      
        'alpha_loss': [],         
        'alpha': [],              
        'portfolio_values': []
    }

    def process_metrics(metrics_dict):
        """Helper function to process metrics for logging"""
        processed_metrics = {}
        for key, value in metrics_dict.items():
            try:
                if isinstance(value, (tf.Tensor, tf.Variable)):
                    processed_metrics[key] = float(value.numpy())
                elif isinstance(value, np.ndarray):
                    processed_metrics[key] = float(np.mean(value))
                else:
                    processed_metrics[key] = float(value)
            except Exception as e:
                print(f"Error processing metric {key}: {str(e)}")
                processed_metrics[key] = 0.0
        return processed_metrics

    for episode in range(episodes):
        print(f"\nStarting Episode {episode + 1}/{episodes}")
        val_reward = 0.0  # Initialize val_reward
        val_portfolio_values = []
        print("\nTraining Phase:")
        episode_start_time = time.time() 
        state = train_env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_metrics = defaultdict(list)
        portfolio_values = []  # Initialize portfolio_values list for this episode
        done = False

        while not done:
            # --- Start Profiler ---
            # Start profiling at a specific step count within an episode
            # Avoid profiling episode 0 or very early steps which might behave differently.
            if enable_profiling and episode == 1 and episode_steps == profiler_start_step and not is_profiling:
                print(f"\n>>> Starting Profiler for steps {profiler_start_step}-{profiler_start_step+profiler_duration_steps}...")
                tf.profiler.experimental.start(log_dir, options=profiler_options)
                is_profiling = True

            # Sample action from policy
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = train_env.step(action)
            portfolio_values.append(info['portfolio_value'])  # Track portfolio value
            
            agent.save_experience(state, action, reward, next_state, done)

            # Update networks if buffer is large enough
            if len(agent.replay_buffer) >= min_buffer_size:
                if not has_started_training:
                    print(f"\nStarting training at episode {episode + 1} with buffer size {len(agent.replay_buffer)}")
                    has_started_training = True
                    # Reset best validation reward when actual training starts
                    best_val_reward = -np.inf
            
            # 2. Update networks (learn) multiple times
            if len(agent.replay_buffer) >= min_buffer_size:
                try:
                    # print(f"Attempting update at buffer size: {len(agent.replay_buffer)}")
                    for _ in range(updates_per_step):
                        # This is the key part - we need to create a proper trace context
                        #step_count = episode_steps * updates_per_step + i  # Unique step ID
                        # with tf.profiler.experimental.Trace('train', step_num=step_count):
                        # Update the agent with the sampled batch        
                        metrics = agent.update(batch_size)
                        
                        if metrics is not None:
                            # Process metrics before storing
                            processed_metrics = process_metrics(metrics)
                            for k, v in processed_metrics.items():
                                episode_metrics[k].append(v)
                except Exception as e:
                    print(f"Error during update: {str(e)}")
                    print(f"Buffer size: {len(agent.replay_buffer)}")
                    print(f"State shape: {state['shared_features'].shape}")
                    print(f"Action shape: {action.shape}")
                    raise e
            
            state = next_state
            episode_reward += reward
            episode_steps += 1

            # --- Stop Profiler ---
            if is_profiling and episode_steps >= (profiler_start_step + profiler_duration_steps):
                print(f"\n>>> Stopping Profiler. Logs saved to: {log_dir}")
                tf.profiler.experimental.stop()
                is_profiling = False
        
        print(f"\nTraining Episode Complete:")
        print(f"Steps: {episode_steps}")
        print(f"Final Training Reward: {episode_reward:.2f}")
 
        # Calculate episode statistics
        if len(portfolio_values) > 1:  # Ensure we have enough values to calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            # sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
            # max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        print(f"\nValidation data details:")
        print(f"Validation steps: {len(val_data)}")
        print(f"First validation timestamp: {val_env.current_step}")
        print(f"Last validation timestamp: {val_env.num_dates - 1}")

        # Validation phase
        if has_started_training:
            print("\nStarting validation phase...")
            if len(val_data) > 0:  # Check if we have validation data
                val_reward, val_portfolio_values = evaluate_agent(agent, val_env)
                print(f"Validation completed. Reward: {val_reward:.2f}")
            else:
                print("No validation data available")
                val_reward = 0
                val_portfolio_values = []
        else:
            print("Agent hasn't started training yet. Skipping validation.")

                # Early stopping check
        """if has_started_training and val_reward >= best_val_reward:
            best_val_reward = val_reward
            best_val_performance = val_portfolio_values
            patience_counter = 0
            agent.save(BEST_MODEL_PATH3)
            print(f"New best validation reward: {best_val_reward:.2f}")
        elif has_started_training:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break"""
        
        # Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('Training/Episode_Reward', episode_reward, step=episode)
            # tf.summary.scalar('Training/Sharpe_Ratio', sharpe_ratio, step=episode)
            # tf.summary.scalar('Training/Max_Drawdown', max_drawdown, step=episode)
            tf.summary.scalar('Validation/Reward', val_reward, step=episode)
            
            # Log network metrics
            for k, v in episode_metrics.items():
                if v:  # Only if we have values
                    try:
                        avg_value = float(np.mean(v))
                        if k in metrics_history:  # Only append if key exists
                            metrics_history[k].append(avg_value)
                            tf.summary.scalar(f'Metrics/{k}', avg_value, step=episode)
                    except Exception as e:
                        print(f"Error processing metric {k}: {str(e)}")
                        if k in metrics_history:  # Only append if key exists
                            metrics_history[k].append(0.0)

        # Store basic metrics
        metrics_history['episode_rewards'].append(float(episode_reward))
        metrics_history['validation_rewards'].append(float(val_reward))
        # metrics_history['sharpe_ratios'].append(float(sharpe_ratio))
        # metrics_history['max_drawdowns'].append(float(max_drawdown))
        
        # Store portfolio values for this episode
        if portfolio_values:
            metrics_history['portfolio_values'].append(portfolio_values)

        # Periodic saving
        if has_started_training and (episode + 1) % save_frequency == 0:
            agent.save(CHECKPOINT_PATH3)
            print(f"Checkpoint saved at episode {episode + 1}")

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
        # Get final portfolio state
        final_portfolio_value = portfolio_values[-1] if portfolio_values else 0
        final_cash_balance = train_env.balance
        final_holdings = train_env.portfolio

        # Log progress with enhanced portfolio information
        print(f"\nEpisode {episode + 1}/{episodes} completed in {episode_duration:.2f}s")
        print(f"Training reward: {episode_reward:.2f}")
        print(f"Validation reward: {val_reward:.2f}")
        # print(f"Sharpe ratio: {sharpe_ratio:.4f}")
        # print(f"Max drawdown: {max_drawdown:.2%}")
        
        # Print portfolio details
        print("\nPortfolio Summary:")
        print(f"Total Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Cash Balance: ${final_cash_balance:.2f}")
        print("Holdings:")
        for i, ticker in enumerate(tradeable_tickers):
            if final_holdings[i] > 0:  # Only show non-zero positions
                current_price = train_env.unscaled_close_prices[train_env.current_step-1, i]
                position_value = final_holdings[i] * current_price
                print(f"  {ticker}: {final_holdings[i]:.6f} shares @ ${current_price:.2f} = ${position_value:.2f}")
        
        # Print metrics
        if episode_metrics:
            print("\nAverage losses:", {
                k: f"{np.mean(v):.4f}" for k, v in episode_metrics.items()
            })
        print(f"Buffer size: {len(agent.replay_buffer)}")
        print("-" * 80)

    # Final save
    if has_started_training:
        agent.save(CHECKPOINT_PATH3)

    # bot_logger.output_log(print_log=False, save_to_file="trading_log.csv")
    
    # Save metrics history to file
    with open(f'metrics_history_{current_time}.pkl', 'wb') as f:
        pickle.dump(metrics_history, f)
        
    return metrics_history

def evaluate_agent(agent, env, num_episodes=1):
    """Evaluate the agent without exploration"""
    total_reward = 0
    portfolio_values = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False 
        episode_reward = 0
        step_count = 0
        max_steps = env.num_dates - 1  

        while not done and step_count < max_steps:
            action = agent.select_action(state, evaluate=True)
            # fixed_action = np.array([0.5, -0.5, 0.1])
            # print(f"Validation Step: {step_count}, Action: {action}") 
            # print(f"Validation Step: {step_count}, Using Fixed Action: {fixed_action}")
            # action = fixed_action
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            portfolio_values.append(info['portfolio_value'])
            step_count += 1

        total_reward += episode_reward

    return total_reward / num_episodes, portfolio_values

if __name__ == "__main__": 
    metrics_history = train_sac()

