
"""Gymnasium-compatible trading environment.

NOTES
-----
* Imports are updated for the *gymnasium* fork (``import gymnasium as gym``).
* ``logger``: the constructor parameter is still called *bot_logger* for
  backward‑compatibility, but internally we never shadow the imported
  ``tradingbot.logger`` helper -- rename the import to ``_BotLogger`` and use
  ``self.bot_logger = bot_logger or _BotLogger()``.
* The file lives at ``tradingbot/env/trading_env.py`` and is re‑exported in
  ``tradingbot/env/__init__.py``.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
from tradingbot import cfg, get_logger
import logging

def _or_cfg(value, dotted_path: str):
    """Return *value* if not None, otherwise walk dotted path inside `cfg`."""
    if value is not None:
        return value
    node = cfg
    for attr in dotted_path.split("."):
        node = getattr(node, attr)
    return node

class TradingEnv(gym.Env):
    """A multi‑asset, continuous‑action trading environment."""

    metadata = {"render_modes": ["human"]}

    # ---------------------------------------------------------------------
    # ── INITIALISATION ────────────────────────────────────────────────────
    # ---------------------------------------------------------------------

    def __init__( self, data: np.ndarray, tradeable_tickers: list[str] | Tuple[str, ...], feature_tickers: list[str] | Tuple[str, ...],
        feature_scalers: Dict[str, Any], ticker_scalers: Dict[int, Dict[int, Any]], *, initial_balance: float | None = None, transaction_cost: float | None = None,
        window_size: int | None = None, bot_logger=None, is_validation: bool = False, seed: int | None = None, ):
        super().__init__()

        self.logger = bot_logger or get_logger(self.__class__.__name__)

        # Seeding for reproducibility within the environment instance
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
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

        self.tradeable_tickers = list(tradeable_tickers)
        self.feature_tickers   = list(feature_tickers)
        self.num_tradeable = len(tradeable_tickers)
        self.num_feature_tickers = len(feature_tickers)

        # --- Crucial Check: Ensure data dimensions match tickers ---
        expected_total_tickers = self.num_feature_tickers + self.num_tradeable
        if self.num_total_tickers_in_data != expected_total_tickers:
            raise ValueError(f"Data shape mismatch: data has {self.num_total_tickers_in_data} tickers, "
                            f"but expected {expected_total_tickers} ({self.num_feature_tickers} feature + {self.num_tradeable} tradeable).")
        # --- End Check ---

        self.initial_balance = _or_cfg(initial_balance, "training.initial_balance")
        self.transaction_cost = _or_cfg(transaction_cost, "training.transaction_cost") \
            if hasattr(cfg.training, "transaction_cost") else 0.001
        self.window_size = _or_cfg(window_size, "training.window_size")
        self.current_step = self.window_size # Start after the first window
        self.epsilon = 1e-9

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
        self._shared_idx_arr  = np.asarray(self.shared_feature_indices,  dtype=np.int64)


        # Indices for ticker-specific features (excluding shared features)
        self.ticker_feature_indices = [
            i for i in range(self.num_features) if i not in self.shared_feature_indices
        ]
        self.ticker_feature_indices.sort()
        self._ticker_idx_arr  = np.asarray(self.ticker_feature_indices, dtype=np.int64)

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

        lg = self.logger            # shorthand

        lg.debug("Data shape: %s", self.data.shape)
        lg.debug("Weekend feature index: %d", self.is_weekend_index)
        lg.debug("Holiday feature index: %d", self.is_holiday_index)
        lg.debug("Close-price index: %d", self.close_price_index)
        lg.debug("is_trading_day array shape: %s", self.is_trading_day.shape)
        lg.debug("Number of trading days: %d", int(np.sum(self.is_trading_day)))
        lg.debug("First 20 is_trading_day values: %s",
                self.is_trading_day[:20].astype(int).tolist())

        lg.debug(
            "[init] TradingEnv initialised – tradeable=%d, features=%d, "
            "window=%d, days=%d",
            self.num_tradeable, self.num_features, self.window_size, self.num_dates,
        )
        lg.debug("[init] Shared feature indices: %s", self.shared_feature_indices)
        lg.debug("[init] Ticker feature indices: %s", self.ticker_feature_indices)

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
            raise KeyError("NaN values detected in precomputed unscaled prices after inverse transform.")
            # Consider handling this, e.g., by filling NaNs or raising an error earlier during scaling checks.

        self.logger.info(
            "TradingEnv init – tradeable %d, features %d, window %d, "
            "dates %d, balance %.0f",
            self.num_tradeable, self.num_features, self.window_size,
            self.num_dates, self.initial_balance,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Handle seeding according to Gymnasium standard
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            # TF seeding should be handled per process outside

        self.balance   = float(self.initial_balance)
        self.portfolio = np.zeros(self.num_tradeable, dtype=np.float32)


        # Reset step to the start of the window
        self.current_step = self.window_size # Start after the first window

        # Ensure there are enough steps left for at least one trade
        if self.current_step >= self.num_dates:
            raise ValueError(f"Window size ({self.window_size}) is too large for the number of dates ({self.num_dates}). No trading possible.")

        self.trade_history: list = []
        # self.position_open_time = np.full(self.num_tradeable, -1) # If needed for PDT rules etc.
        # self.day_trade_history = deque(maxlen=5) # If needed
        # self.day_trades_today = 0 # If needed

        self.logger.info(
            "[reset] env reset – balance=%.2f, start_step=%d",
            self.balance, self.current_step,
        )

        # ------------------------------------------------------------ #
        #   First observation
        # ------------------------------------------------------------ #
        observation = self._get_observation()
        info        = self._get_info()

        # shape/type guard (debug only)
        if not self.observation_space.contains(observation):
            self.logger.debug(
                "Observation mismatch on reset - shapes: expected %s got %s",
                {k: sp.shape for k, sp in self.observation_space.spaces.items()},
                {k: v.shape for k, v in observation.items()},
            )
            raise ValueError("Reset observation does not match observation_space")

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
        """Return observation dict for the current step (exclusive)."""

        if self.current_step < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} steps, have {self.current_step}"
            )

        start_idx = self.current_step - self.window_size
        end_idx   = self.current_step
        if start_idx < 0 or end_idx > self.num_dates:
            raise IndexError(f"Window [{start_idx}:{end_idx}) outside data range")

        # fast Numba slice
        shared_w, ticker_w = TradingEnv.compute_observation_window(
            self.data,
            self._shared_idx_arr,
            self._ticker_idx_arr,
            start_idx,
            self.window_size,
            self.num_tradeable,
            self.num_feature_tickers,
        )

        obs = {
            "shared_features":   shared_w,
            "ticker_features":   ticker_w,
            "portfolio_holdings": self.portfolio.astype(np.float32),
            "cash_balance":      np.array([self.balance], dtype=np.float32),
        }

        # --- dev-time verification (DEBUG only) --------------------------
        if self.logger.isEnabledFor(logging.DEBUG):
            if not self.observation_space.contains(obs):
                self.logger.debug(
                    "Obs mismatch – expected %s got %s",
                    {k: sp.shape for k, sp in self.observation_space.spaces.items()},
                    {k: v.shape for k, v in obs.items()},
                )
                raise ValueError("Observation does not match observation_space")
        # -----------------------------------------------------------------

        return obs

    def _get_info(self):
        """Auxiliary info returned by Gymnasium step/reset."""
        prev_step = max(self.current_step - 1, 0)

        info = {
            "current_step":    self.current_step,
            "portfolio_value": self._get_portfolio_value(prev_step),
            "balance":         self.balance,
            "holdings":        self.portfolio.copy(),
        }

        # Uncomment for very fine-grained tracing
        # self.logger.debug("[info] %s", info)

        return info

    def _calculate_reward(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        """
        Calculates the reward for the *transition* that just occurred.
        Uses portfolio value *before* and *after* the step.
        """
        # --- Basic Portfolio Return ---
        # Avoid division by zero if previous value was 0
        if prev_portfolio_value is None or prev_portfolio_value <= self.epsilon:
            portfolio_return = 0.0
        else:
            portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
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


        # --- Sharpe Ratio (Optional, requires history) ---
        sharpe_reward = 0
        # Calculate over last 30 steps if history is sufficient
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
        reward = float(np.clip(reward, -1, 1))

        # --- Store Trade History (for potential later analysis or reward calculation) ---
        self.trade_history.append({
            "step":                self.current_step - 1,
            "return":              portfolio_return,
            "portfolio_value":     portfolio_value,
            "cash_balance":        self.balance,
            "holdings_vector":     self.portfolio.copy(),
            "reward":              reward,
            "raw_holding_rewards": holding_rewards,
            "raw_concentration_penalty": concentration_penalty,
            "raw_drawdown_penalty":      drawdown_penalty,
        })

        # ── Logging (DEBUG) ------------------------------------------------------
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[reward] step %d | prev=%.2f cur=%.2f "
                "ret=%.4f scaled_port=%.4f scaled_hold=%.4f "
                "sharpe=%.4f conc_pen=-%.4f draw_pen=-%.4f final=%.4f",
                self.current_step - 1,
                prev_portfolio_value,
                portfolio_value,
                portfolio_return,           
                scaled_portfolio_return,
                scaled_holding_rewards,
                sharpe_reward,
                concentration_penalty,
                drawdown_penalty,
                reward,
            )

            return reward

    def step(self, action):
        """
        Executes one time step within the environment.
        Follows the Gymnasium API: returns observation, reward, terminated, truncated, info.
        """
        # --- Input Validation ---
        if not self.action_space.contains(action):
            self.logger.warning(
                "Invalid action %s – clipping to space %s", action, self.action_space
            )
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
                self.logger.debug("Step %d is non-trading – no trades", self.current_step)
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
        # integrity check (debug only)
        if self.logger.isEnabledFor(logging.DEBUG) and \
        not self.observation_space.contains(observation):
            self.logger.debug("Obs mismatch post-step; raising")
            raise ValueError("Step observation does not match observation_space")

        self.logger.debug(
            "[step] %4d | act %s | r %.4f | val %.2f | bal %.2f | trunc=%s",
            self.current_step - 1,
            np.round(action, 3),
            reward,
            current_portfolio_value,
            self.balance,
            truncated,
        )


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
            self.logger.warning("Bad day_index %d in _execute_trades – skip", day_index)
            return

        # Get open prices for all tradeable tickers for the *current* day
        try:
            # Ensure we only take prices for tradeable tickers
            prices = self.unscaled_open_prices[day_index, :self.num_tradeable]
        except IndexError:
            if self.bot_logger: self.bot_logger.error(f"IndexError accessing open prices at day_index {day_index}.")
            self.logger.warning("IndexError accessing open prices at day_index %d", day_index)
            return

        # --- Input Validation ---
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            self.logger.error("NaN/Inf in actions %s – skip trades", actions)
            return
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)) or np.any(prices <= self.epsilon):
             # Identify problematic tickers
             bad_price_mask = np.isnan(prices) | np.isinf(prices) | (prices <= self.epsilon)
             bad_tickers = [self.tradeable_tickers[i] for i, bad in enumerate(bad_price_mask) if bad]
             self.logger.warning("Bad prices %s @ step %d – masking actions", bad_price_mask, day_index)
             # Mask actions for tickers with bad prices - don't trade them
             actions = np.where(bad_price_mask, 0, actions) # Set action to 0 if price is bad
             prices = np.where(bad_price_mask, 1, prices) # Set price to 1 to avoid division by zero later, action is 0 anyway
             # If all prices are bad, skip entirely
             if np.all(bad_price_mask):
                return


        # Use float64 for precision in calculations
        actions = actions.astype(np.float64)
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
                if self.logger.isEnabledFor(logging.DEBUG):
                    for i, global_idx in enumerate(sell_indices_global):
                        ticker = self.tradeable_tickers[global_idx]
                        qty = actual_sell_quantity[i]
                        price_val = actual_sell_prices[i]
                        proc = proceeds[i]
                        self.logger.debug("Sell %s %.4f @ %.2f – proceeds %.2f", self.tradeable_tickers[i], qty, prices[i], qty * prices[i])


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
                        if self.logger.isEnabledFor(logging.DEBUG):
                            for i, global_idx in enumerate(actual_buy_indices_global):
                                ticker = self.tradeable_tickers[global_idx]
                                qty = final_buy_quantity[i]
                                price_val = final_buy_prices[i]
                                c = final_buy_cost[i]

                                self.logger.debug("Buy  %s %.4f @ %.2f – cost %.2f",self.tradeable_tickers[i], qty, prices[i], qty * prices[i])


                        # Update balance and portfolio
                        self.balance -= np.sum(final_buy_cost) # Use the final calculated cost sum
                        portfolio[actual_buy_indices_global] += final_buy_quantity


        # --- Final State Update ---
        # Update the master portfolio state. Ensure no NaNs/Infs and non-negative.
        self.portfolio = np.maximum(0.0, portfolio).astype(np.float32) # Ensure non-negative and float32
        self.balance   = max(0.0, self.balance) # Ensure non-negative

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