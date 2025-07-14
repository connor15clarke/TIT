"""tradingbot.data.indicators – *v2*
-------------------------------------
Resilient wrapper around **pandas_ta** that sanitises the OHLCV input so the
library never receives ``None`` values (the root cause of the previous
``unsupported operand type(s) for -: 'float' and 'NoneType'`` error).

Public API is unchanged:
>>> ti = TechnicalIndicators()
>>> out = ti.compute_indicators_for_tickers(stock_df, dates)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta

from tradingbot import get_logger

__all__ = ["TechnicalIndicators"]

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _map_ta_cols(p: Dict[str, int]) -> Dict[str, str]:
    """Return the column names that *pandas_ta* will create given *p*."""
    return {
        "macd": f"MACD_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}",
        "macd_signal": f"MACDs_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}",
        "rsi": f"RSI_{p['rsi_length']}",
        "upper_band": f"BBU_{p['bbands_length']}_{p['bbands_std']}.0",
        "lower_band": f"BBL_{p['bbands_length']}_{p['bbands_std']}.0",
        "stoch_osc": f"STOCHk_{p['stoch_k']}_{p['stoch_d']}_{p['stoch_smooth_k']}",
        "obv": "OBV",
        "atr": f"ATR_{p['atr_length']}",
    }


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are numeric and *never* contain None/NaN."""
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    df = df.copy()
    # Force numeric with errors→NaN, then ffill/bfill, finally zeros
    df[ohlcv] = df[ohlcv].apply(pd.to_numeric, errors="coerce")
    df[ohlcv] = df[ohlcv].ffill().bfill().fillna(0.0)
    return df

# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #

class TechnicalIndicators:
    """Compute a fixed bundle of indicators and align them to *date_range*."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def compute_indicators_for_tickers(
        self,
        stock_data: pd.DataFrame,  # MultiIndex (Date, Ticker)
        date_range: pd.DatetimeIndex,
        collection_start_offset: int = 30,
        indicator_params: Dict[str, int] | None = None,
    ) -> Dict[str, Dict[str, List[float]]]:
        """Return ``{ticker: {indicator_name: [values…]}}`` aligned to *date_range*."""
        self.logger.info(
            "Computing indicators for %d tickers",
            stock_data.index.get_level_values("Ticker").nunique(),
        )

        # Defaults → user overrides
        defaults = {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_length": 14,
            "bbands_length": 20,
            "bbands_std": 2,
            "stoch_k": 14,
            "stoch_d": 3,
            "stoch_smooth_k": 3,
            "atr_length": 14,
        }
        p = {**defaults, **(indicator_params or {})}
        ta_cols = _map_ta_cols(p)
        keys = list(ta_cols)

        out: Dict[str, Dict[str, List[float]]] = {}
        tickers = stock_data.index.get_level_values("Ticker").unique()

        for tkr in tickers:
            df = stock_data.xs(tkr, level="Ticker")
            df = _clean_ohlcv(df)

            if len(df) < p["bbands_length"]:
                self.logger.warning("%s: not enough rows (%d), filling zeros", tkr, len(df))
                n_pad = max(0, len(date_range) - collection_start_offset)
                out[tkr] = {k: [0.0] * n_pad for k in keys}
                continue

            # ------------------------------------------------------------------ #
            # run pandas_ta safely
            # ------------------------------------------------------------------ #
            try:
                # MACD, RSI, Bollinger
                df.ta.macd(
                    fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"], append=True
                )
                df.ta.rsi(length=p["rsi_length"], append=True)
                df.ta.bbands(length=p["bbands_length"], std=p["bbands_std"], append=True)
                # Stochastic (%K) – grab one column from the returned DF
                stoch = df.ta.stoch(k=p["stoch_k"], d=p["stoch_d"], smooth_k=p["stoch_smooth_k"])
                if stoch is not None:
                    df[ta_cols["stoch_osc"]] = stoch[ta_cols["stoch_osc"]]
                # OBV & ATR
                df.ta.obv(append=True)
                df.ta.atr(length=p["atr_length"], append=True)
            except Exception as exc:  # pragma: no cover – unpredictable faults
                self.logger.error("%s: pandas_ta error – %s", tkr, exc)
                n_pad = max(0, len(date_range) - collection_start_offset)
                out[tkr] = {k: [0.0] * n_pad for k in keys}
                continue

            # Replace any residual NaN with zeros *after* computing
            df = df.fillna(0.0)

            # ------------------------------------------------------------------ #
            # align to date_range & slice after warm‑up window
            # ------------------------------------------------------------------ #
            cur = {k: [] for k in keys}
            for i, d in enumerate(date_range):
                if i < collection_start_offset:
                    continue
                row = df.reindex([d], method="ffill").iloc[0]
                for k in keys:
                    cur[k].append(float(row.get(ta_cols[k], 0.0)))
            out[tkr] = cur
            self.logger.debug("%s: collected %d points", tkr, len(cur["macd"]))

        self.logger.info("Indicator computation complete – output for %d tickers", len(out))
        if out:
            first_ticker_key = list(out.keys())[0]
            self.logger.debug(f"Final 'out' structure preview (for ticker: {first_ticker_key}):")
            if first_ticker_key in out:
                for indicator_name, values_list in out[first_ticker_key].items():
                    preview = str(values_list[:3]) + "..." if len(values_list) > 3 else str(values_list)
                    self.logger.debug(f"  {indicator_name}: length={len(values_list)}, first_3_values_preview={preview}")
        else:
            self.logger.info("Final 'out' is empty.")

        return out
