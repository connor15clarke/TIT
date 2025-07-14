# tradingbot/cli.py
"""
TradingBot command-line interface
---------------------------------

Example usage
~~~~~~~~~~~~~
# parallel SAC training (all config comes from tradingbot.config)
python -m tradingbot.cli train-sac-parallel --save-dir runs/exp01

# quick deterministic evaluation of the most recent checkpoint
python -m tradingbot.cli eval --checkpoint runs/exp01/checkpoint
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any
import signal
import sys
import logging
import traceback

from tradingbot.config import cfg
from tradingbot.logger import get_logger
from tradingbot.env import TradingEnv
from tradingbot.agents import SACAgent
from tradingbot.parallel.run import train_sac_parallel

LOG = get_logger("cli")


# --------------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> None:
    # adopt user-supplied path or fall back to cfg
    save_dir: Path | None = Path(args.save_dir) if args.save_dir else cfg.path.base_save_path

    train_sac_parallel(save_dir)


# --------------------------------------------------------------------------- #
def _cmd_eval(args: argparse.Namespace) -> None:
    chk = Path(args.checkpoint).expanduser().resolve()
    if not chk.exists():
        raise FileNotFoundError(chk)

    LOG.info("Loading agent from %s", chk)
    env = TradingEnv.load_from_checkpoint(chk.parent / "scalers.pkl")   # helper you already wrote
    agent = SACAgent.load(chk, env.observation_space, env.action_space)

    state, _ = env.reset(seed=42)
    done = False
    ep_r = 0.0

    while not done:
        action = agent.select_action(state, deterministic=True)
        state, r, term, trunc, _ = env.step(action)
        done = term or trunc
        ep_r += r

    LOG.info("Episode return: %.4f", ep_r)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingbot", description="Stock-trading RL bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train-sac-parallel --------------------------------------------------
    t = sub.add_parser("train-sac-parallel", help="Run parallel SAC training")
    t.add_argument("--save-dir", type=str, default=None,
                   help="Directory to write model checkpoints (defaults to cfg.path.base_save_path)")
    t.set_defaults(func=_cmd_train)

    # --- eval ----------------------------------------------------------------
    e = sub.add_parser("eval", help="Evaluate a saved checkpoint for one episode")
    e.add_argument("--checkpoint", type=str, required=True,
                   help="Path to actor checkpoint (.h5)")
    e.set_defaults(func=_cmd_eval)

    return p


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)         # dispatch


if __name__ == "__main__":
    main()