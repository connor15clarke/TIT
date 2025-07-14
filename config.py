"""tradingbot.config
-------------------
Static, central configuration -- import **once** and reuse everywhere::

    from tradingbot import cfg, get_logger
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=all, 1=info, 2=warning, 3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN banner
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml

# --------------------------------------------------------------------------- #
#  Section-level dataclasses
# --------------------------------------------------------------------------- #

__all__ = ["cfg", "Config", "load_yaml", "save_yaml","PathConfig"]

# ---------------------------------------------------------------------------
#                           Top‑level Sections
# ---------------------------------------------------------------------------

@dataclass
class GeneralConfig:
    PROJECT_NAME: str = "TradingBot"
    RANDOM_SEED: int = 42
    TORCH_DEVICE: str = os.getenv(
        "TRADINGBOT_TORCH_DEVICE",
        "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
    )

    global_seed: int = 42
    use_mixed_precision: bool = True  # Keras mixed‑float16 policy

@dataclass
class DataConfig:
    DATA_DIR: Path = Path(os.getenv("TRADINGBOT_DATA_DIR", "./data")).expanduser()
    CACHE_DIR: Path = Path(os.getenv("TRADINGBOT_CACHE_DIR", "./data/cache")).expanduser()
    """Dates, tickers and paths used in data collection."""

    # Inclusive start, exclusive end – matches pandas date_range default.
    start_date: str = "2024-10-18"
    end_date: str = "2025-05-23"

    # Index / benchmark tickers (not tradeable) that still feed features.
    feature_tickers: list[str] = field(
        default_factory=lambda: ["^IXIC", "^GSPC", "^DJI"]
    )

    # Leveraged + unleveraged products the RL agent *can* hold.
    tradeable_tickers: list[str] = field(
        default_factory=lambda: ["TQQQ", "SQQQ", "QQQ"]
    )

@dataclass
class APIConfig:
    """3rd‑party service credentials.

    Each default looks for an env‑var first.  This way CI / prod can set real
    secrets without touching code.
    """

    FINNHUB_KEY: str = os.getenv("FINNHUB_KEY", "coavp0pr01qro9kpjlb0coavp0pr01qro9kpjlbg")

@dataclass
class ModelsConfig:
    dense_hidden_1: int = 128
    dense_hidden_2: int = 64
    l2: float = 1e-5

@dataclass
class AgentConfig:
    learning_rate: float = 3e-5
    gamma: float = 0.99
    tau: float = 0.005
    replay_buffer_capacity: int = 50_000
    min_buffer_size: int = 128  # Min experiences before training starts
    log_dir_base: str = "logs/sac_agent"
    worker_id: int = 0  # Worker ID for logging

# -------------------------------------------------------------------
agent = AgentConfig()

@dataclass
class TrainingConfig:
    """Hyperparameters that steer the SAC agent."""

    episodes: int = 2000 # Training episodes
    batch_size: int = 64 # Gradient update batch size
    updates_per_step: int = 1
    initial_balance: float = 10_000.0 # Starting portfolio value
    transaction_cost: float = 0.001
    window_size: int = 10  # rolling history window fed to networks
    save_frequency: int = 100  # checkpoint every *n* steps

@dataclass
class ParallelConfig:
    num_workers: int = 1
    gpu_map: list[int] = field(        # ← explicit GPU per worker
        default_factory=lambda: [0]
    )
    steps_per_env: int = 5000 # Steps each worker trains for
    push_every:    int = 1000     
    polyak:        float = 0.05     
    sync_mode:     str   = "parameter_server"  # default hybrid

parallel = ParallelConfig()

@dataclass
class PathConfig:
    """Filesystem locations – single source of truth for I/O paths."""

    base_save_path: Path = Path("models/sac_parallel_run")
    log_dir_base: Path = Path("logs/sac_parallel_run")
    scaler_path: Path = Path("my_scalers_parallel.pkl")

@dataclass
class LoggingConfig:
    LEVEL: str = os.getenv("TRADINGBOT_LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path(os.getenv("TRADINGBOT_LOG_DIR", "./logs")).expanduser()
    LOG_FILE: str = "tradingbot.log"

    tensorboard_update_freq: int = 100      # Log training metrics every N gradient updates
    worker_log_freq: int = 100               # Log worker metrics every N episodes
    coordinator_log_freq: int = 30          # Log coordinator metrics every N seconds

# ---------------------------------------------------------------------------
#                        Composite Root Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    general:  GeneralConfig  = field(default_factory=GeneralConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    api:      APIConfig      = field(default_factory=APIConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models:   ModelsConfig   = field(default_factory=ModelsConfig)
    agent:    AgentConfig    = field(default_factory=AgentConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    path:     PathConfig      = field(default_factory=PathConfig) 

    # ------------- helpers -------------------------------------------------- #
    def update_from_mapping(self, mapping: Dict[str, Any]) -> None:
        """Recursively update fields from a dict (e.g., YAML overrides)."""
        for section_name, section_map in mapping.items():
            section = getattr(self, section_name, None)
            if section and isinstance(section_map, dict):
                for key, value in section_map.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def load_yaml(self, path: Path | str) -> None:
        """Load overrides from a YAML file."""
        path = Path(path)
        if path.exists():
            with path.open("r") as fh:
                data = yaml.safe_load(fh) or {}
            self.update_from_mapping(data)

    # Nice to have when debugging
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Live singleton – import this everywhere
# --------------------------------------------------------------------------- #

cfg = Config()

# automatic YAML override if env var points to a file
_yaml_override = os.getenv("TRADINGBOT_CONFIG")
if _yaml_override:
    cfg.load_yaml(_yaml_override)

# Ensure key directories exist
cfg.data.DATA_DIR.mkdir(parents=True, exist_ok=True)
cfg.data.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cfg.logging.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Convenience re-exports
__all__ = ["cfg"]

# ↓ Optional helper so other modules can do `from tradingbot.config import load_yaml`
def load_yaml(path: Path | str) -> None:  # noqa: D401
    """Load a YAML file into the global ``cfg`` instance."""
    cfg.load_yaml(path)