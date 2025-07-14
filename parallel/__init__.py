"""tradingbot.parallel – helper to launch multi-process training."""
from __future__ import annotations
from tradingbot import get_logger

# from .run import train_sac_parallel

__all__: list[str] = ["train_sac_parallel"]

get_logger(__name__).debug("parallel package imported – no processes yet")