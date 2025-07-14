"""tradingbot.data package initialiser"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from tradingbot import cfg, get_logger
_logger = get_logger(__name__)

# --- auto‑discover & re‑export ------------------------------------------------

_submodules: Dict[str, str] = {
    "sentiment": "SentimentAnalyzer",
    "news": "NewsFetcher",
    "market": "MarketDataCollector",
    "indicators": "TechnicalIndicators",
    "time_features": "TimeFeatureGenerator",
    "preprocess": "DataProcessor",
}

__all__: List[str] = ["cfg", "get_logger"]  # will extend below

for _mod_name, _obj_name in _submodules.items():
    try:
        _mod: ModuleType = import_module(f"{__name__}.{_mod_name}")
        _obj: Any = getattr(_mod, _obj_name)
        globals()[_obj_name] = _obj  # re‑export at package level
        __all__.append(_obj_name)
    except Exception as exc:  # pragma: no cover — soft‑fail for optional deps
        _logger.warning("Could not import %s.%s: %s", _mod_name, _obj_name, exc)

if TYPE_CHECKING:
    # typing‑time imports for IDEs / mypy
    from .sentiment import SentimentAnalyzer as SentimentAnalyzer
    from .news import NewsFetcher as NewsFetcher
    from .market import MarketDataCollector as MarketDataCollector
    from .indicators import TechnicalIndicators as TechnicalIndicators
    from .time_features import TimeFeatureGenerator as TimeFeatureGenerator
    from .preprocess import DataProcessor as DataProcessor


