"""tradingbot.logger  – console + CSV handler"""
from __future__ import annotations
import logging, csv
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tradingbot.config import cfg

class _CSVFormatter(logging.Formatter):
    def format(self, record):          # noqa: D401
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f'{ts},{record.levelname},{record.name},"{record.getMessage()}"'

def _build_root() -> logging.Logger:
    root = logging.getLogger("tradingbot")
    if root.handlers:
        return root                    # already configured

    root.setLevel(cfg.logging.LEVEL.upper())

    # console
    con = logging.StreamHandler()
    con.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
                                       "%H:%M:%S"))
    root.addHandler(con)

    # CSV file
    csv_path = Path(cfg.logging.LOG_DIR) / (Path(cfg.logging.LOG_FILE).stem + ".csv")
    fh = RotatingFileHandler(csv_path, maxBytes=5_242_880, backupCount=5)
    fh.setFormatter(_CSVFormatter())
    root.addHandler(fh)

    return root

_root = _build_root()

def get_logger(name: str | None = None) -> logging.Logger:        # noqa: D401
    return _root.getChild(name) if name else _root
