"""tradingbot.env
================
Light-weight wrapper that re-exports :class:`TradingEnv`.

Keeping the heavy-duty implementation inside *trading_env.py*
means importing *tradingbot.env* itself remains cheap and does
**not** touch TensorFlow, GPUs, or large data structures until
you actually access `TradingEnv`.
"""

from __future__ import annotations

from .trading_env import TradingEnv

__all__: list[str] = ["TradingEnv"]