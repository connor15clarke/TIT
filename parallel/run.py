# tradingbot/parallel/run.py
"""
Coordinator / launcher for parallel SAC training.

Key point
----------
1.  Two synchronisation modes, driven by cfg.parallel.sync_mode:
      • "posthoc" – each worker trains to completion; we just pick the
        checkpoint with the best validation reward (behaves like titparallelenv).
      • "parameter_server" – a lightweight ParameterServer process keeps a
        moving-average of the actor weights that workers push every
        N episodes (push code already exists in worker.py: it fires once,

        right before the worker terminates – it can emit more pushes if it
        calls pipe.send(("weights", ...)) inside the episode loop).

"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any
from datetime import datetime
from collections import defaultdict

import numpy as np

from tradingbot.config import cfg
from tradingbot.logger import get_logger

_LOG = get_logger("parallel.run")


# --------------------------------------------------------------------------- #
# 0. tiny helper – Polyak average                                             #
# --------------------------------------------------------------------------- #
def _polyak(target: dict[str, np.ndarray],
            source: dict[str, np.ndarray],
            tau: float = 1.0) -> dict[str, np.ndarray]:
    return {k: tau * source[k] + (1. - tau) * target[k] for k in target}


# --------------------------------------------------------------------------- #
# 1. Parameter-server process (optional)                                      #
# --------------------------------------------------------------------------- #
class ParameterServer(mp.Process):
    """
    Pure-NumPy parameter server.

    • Receives ("push", {name: np.ndarray}) from workers  
    • Polyak-averages into an internal dict (CPU RAM only)  
    • On ("stop", None) sends ("final", weights) back and exits
    """

    def __init__(self, q: mp.Queue) -> None:
        super().__init__(daemon=True)
        self.queue = q
        self.tau   = cfg.parallel.polyak

    # ---------- helpers ------------------------------------------------- #
    @staticmethod
    def _gpu_snapshot(where: str) -> None:
        """
        Log VRAM usage for every visible GPU (runs `nvidia-smi`).
        """
        import subprocess
        try:
            lines = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,nounits,noheader"],
                text=True,
            ).strip().splitlines()
            for idx, ln in enumerate(lines):
                used, total = map(int, ln.split(","))
                _LOG.info("%s – GPU-%d  %4d / %4d MiB", where, idx, used, total)
        except Exception:
            _LOG.info("%s – GPU usage unavailable", where)

    # ---------- main loop ----------------------------------------------- #

    def run(self) -> None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hard CPU fence
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     

        _LOG.info("Parameter-Server PID=%d  (CPU-only)", os.getpid())
        self._gpu_snapshot("  launch")

        weights: dict[str, np.ndarray] | None = None
        pushes = 0

        while True:
            tag, payload = self.queue.get()

            if tag == "push":
                pushes += 1
                if weights is None:                 # first push → copy
                    weights = {k: v.copy() for k, v in payload.items()}
                else:                               # Polyak average
                    for k, v in payload.items():
                        weights[k] = self.tau * v + (1.0 - self.tau) * weights[k]

            elif tag == "stop":
                self.queue.put(("final", weights or {}))
                break

        self._gpu_snapshot("   exit")
        _LOG.info("ParameterServer (CPU) exited after %d pushes", pushes)


# --------------------------------------------------------------------------- #
# 2. Build the shared 3-D tensor + scalers once                               #
# --------------------------------------------------------------------------- #
def _build_env_tensor() -> tuple[np.ndarray, dict, dict]:
    from tradingbot.data.market import MarketDataCollector
    from tradingbot.data.preprocess import DataProcessor, DataPreprocessor
    tickers = cfg.data.feature_tickers + cfg.data.tradeable_tickers
    log     = get_logger("data.pipeline")

    collector = MarketDataCollector(cfg.data.start_date, cfg.data.end_date,
                                    tickers)
    econ, news, stock, _ = collector.fetch_data(tickers)

    processor = DataProcessor(cfg.data.start_date, cfg.data.end_date,
                              tickers, log)
    raw = processor.process_data(econ, stock, news,
                                 window_size=cfg.training.window_size)

    pre = DataPreprocessor()
    tensor = pre.fit_transform(raw)
    pre.save_scalers(cfg.path.scaler_path)

    return tensor, pre.feature_scalers, pre.ticker_scalers


# --------------------------------------------------------------------------- #
# 3. Orchestrator                                                             #
# --------------------------------------------------------------------------- #
def train_sac_parallel(save_dir: str | Path | None = None) -> None:
    from tradingbot.data.market import MarketDataCollector
    from tradingbot.data.preprocess import DataProcessor, DataPreprocessor
    from tradingbot.parallel.worker import worker_process
    from tradingbot.env import TradingEnv
    import tensorflow as tf
    coordinator_start_time = time.perf_counter()

    # Setup TensorBoard for coordinator
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    coordinator_log_dir = Path(cfg.path.log_dir_base) / f"SAC_{ts}" / "coordinator"
    coordinator_log_dir.mkdir(parents=True, exist_ok=True)
    coordinator_writer = tf.summary.create_file_writer(str(coordinator_log_dir))
    
    _LOG.info(f"Coordinator TensorBoard logs → {coordinator_log_dir}")
    
    # Timing and metrics tracking
    phase_times = {}
    worker_metrics = defaultdict(list)
    
    # ------------------------------------------------------------------ #
    #  Phase 1: Data preprocessing
    # ------------------------------------------------------------------ #
    _LOG.info("Phase 1: Building environment tensor...")
    phase_start = time.perf_counter()

    tensor, f_scalers, t_scalers = _build_env_tensor()

    phase_times['data_preprocessing'] = time.perf_counter() - phase_start
    
    with coordinator_writer.as_default():
        tf.summary.scalar('Setup/Data_Preprocessing_Time_s', phase_times['data_preprocessing'], step=0)
        tf.summary.scalar('Setup/Data_Shape_Days', tensor.shape[0], step=0)
        tf.summary.scalar('Setup/Data_Shape_Features', tensor.shape[1], step=0)
        tf.summary.scalar('Setup/Data_Shape_Tickers', tensor.shape[2], step=0)

    # ------------------------------------------------------------------ #
    #  Phase 2: Environment setup
    # ------------------------------------------------------------------ #
    _LOG.info("Phase 2: Setting up environments...")
    phase_start = time.perf_counter()

    env_kwargs: dict[str, Any] = dict(
        data=tensor,
        tradeable_tickers=cfg.data.tradeable_tickers,
        feature_tickers=cfg.data.feature_tickers,
        feature_scalers=f_scalers,
        ticker_scalers=t_scalers,
        window_size=cfg.training.window_size,
        initial_balance=cfg.training.initial_balance,
        transaction_cost=cfg.training.transaction_cost,
    )

    # dummy env for spaces (needed by param-server)
    dummy_env = TradingEnv(**env_kwargs)
    obs_space, act_space = dummy_env.observation_space, dummy_env.action_space

    sync_mode = getattr(cfg.parallel, "sync_mode", "posthoc").lower()
    if sync_mode not in {"posthoc", "parameter_server"}:
        raise ValueError(f"Unknown sync_mode: {sync_mode}")

    ctx = mp.get_context("spawn")
    pipes: list[mp.connection.Connection] = []
    procs: list[mp.Process] = []

    # optional parameter-server
    ps_queue: mp.Queue | None = None
    if sync_mode == "parameter_server":
        ps_queue = ctx.Queue()
        ps_proc  = ParameterServer(ps_queue)
        ps_proc.start()
        _LOG.info("Parameter-Server PID=%d", ps_proc.pid)

    train_sz = int(0.8 * tensor.shape[0])
    train_slice, val_slice = tensor[:train_sz], tensor[train_sz:]

    phase_times['environment_setup'] = time.perf_counter() - phase_start
    
    with coordinator_writer.as_default():
        tf.summary.scalar('Setup/Environment_Setup_Time_s', phase_times['environment_setup'], step=0)
        tf.summary.scalar('Setup/Train_Data_Days', train_slice.shape[0], step=0)
        tf.summary.scalar('Setup/Val_Data_Days', val_slice.shape[0], step=0)

    shared_env_kwargs = dict(
        tradeable_tickers=cfg.data.tradeable_tickers,
        feature_tickers=cfg.data.feature_tickers,
        feature_scalers=f_scalers,
        ticker_scalers=t_scalers,
        window_size=cfg.training.window_size,
        initial_balance=cfg.training.initial_balance,
        transaction_cost=cfg.training.transaction_cost,
    )

    # launch worker processes
    for wid in range(cfg.parallel.num_workers):
        gpu_id = cfg.parallel.gpu_map[wid % len(cfg.parallel.gpu_map)]
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=worker_process,
            args=(child, train_slice, val_slice, shared_env_kwargs, wid, gpu_id),
            daemon=True,
        )
        p.start()
        pipes.append(parent)
        procs.append(p)
        _LOG.info("Spawned worker %d on GPU %d (pid=%d)", wid, gpu_id, p.pid)

    # ------------------------------------------------------------------ #
    #  Phase 4: Training coordination and monitoring
    # ------------------------------------------------------------------ #
    _LOG.info("Phase 4: Coordinating training...")
    training_start_time = time.perf_counter()
    finished, best_val = 0, -np.inf
    best_weights: dict[str, np.ndarray] | None = None
    total_episodes = 0
    total_returns = []
    worker_status = {i: {'episodes': 0, 'last_return': 0.0, 'best_val': -np.inf} 
                     for i in range(cfg.parallel.num_workers)}
    
    # Communication loop
    last_log_time = time.perf_counter()
    communication_times = []

    while finished < cfg.parallel.num_workers:
        loop_start = time.perf_counter()

        for i, pipe in enumerate(pipes[:]):
            if pipe.poll(0.05):
                tag, payload = pipe.recv()

                if tag == "episode_return":
                    episode_return = payload
                    total_episodes += 1
                    total_returns.append(episode_return)
                    
                    # Update worker status
                    worker_id = pipes.index(pipe)
                    worker_status[worker_id]['episodes'] += 1
                    worker_status[worker_id]['last_return'] = episode_return
                    
                    _LOG.debug("Worker %d episode return: %.4f (total episodes: %d)", 
                              worker_id, episode_return, total_episodes)

                elif tag == "push":
                # live update for parameter server
                    if sync_mode == "parameter_server":
                        ps_queue.put(("push", payload["weights"]))
                    
                    # Track worker validation performance
                    worker_id = pipes.index(pipe)
                    worker_val = payload.get("val", -np.inf)
                    worker_status[worker_id]['best_val'] = max(worker_status[worker_id]['best_val'], worker_val)

                elif tag == "done":
                    finished += 1
                    w, v = payload["weights"], payload["val"]
                    worker_id = pipes.index(pipe)
                    
                    _LOG.info("Worker %d finished with validation return: %.4f", worker_id, v)

                    if sync_mode == "posthoc":
                        if v > best_val:
                            best_val, best_weights = v, w
                    else:   # parameter-server
                        assert ps_queue
                        ps_queue.put(("push", w))

                    worker_status[worker_id]['best_val'] = v

                    # the worker has shut its end; close ours and stop polling it
                    pipe.close()
                    pipes.remove(pipe)
        
        # Periodic logging to TensorBoard (every 30 seconds)
        current_time = time.perf_counter()
        if current_time - last_log_time > 30.0:
            elapsed_training_time = current_time - training_start_time
            
            with coordinator_writer.as_default():
                step = int(elapsed_training_time)  # Use elapsed time as step
                
                # Overall progress
                tf.summary.scalar('Training/Total_Episodes', total_episodes, step=step)
                tf.summary.scalar('Training/Workers_Finished', finished, step=step)
                tf.summary.scalar('Training/Workers_Active', cfg.parallel.num_workers - finished, step=step)
                tf.summary.scalar('Training/Elapsed_Time_s', elapsed_training_time, step=step)
                
                # Episode statistics
                if total_returns:
                    recent_returns = total_returns[-50:]  # Last 50 episodes across all workers
                    tf.summary.scalar('Training/Recent_Mean_Return', np.mean(recent_returns), step=step)
                    tf.summary.scalar('Training/Recent_Std_Return', np.std(recent_returns), step=step)
                    tf.summary.scalar('Training/Best_Return_So_Far', np.max(total_returns), step=step)
                    tf.summary.scalar('Training/Worst_Return_So_Far', np.min(total_returns), step=step)
                
                # Worker-specific metrics
                active_workers = cfg.parallel.num_workers - finished
                if active_workers > 0:
                    avg_episodes_per_worker = sum(ws['episodes'] for ws in worker_status.values()) / cfg.parallel.num_workers
                    tf.summary.scalar('Training/Avg_Episodes_Per_Worker', avg_episodes_per_worker, step=step)
                    
                    # Individual worker progress
                    for wid, status in worker_status.items():
                        tf.summary.scalar(f'Workers/Worker_{wid}_Episodes', status['episodes'], step=step)
                        tf.summary.scalar(f'Workers/Worker_{wid}_Last_Return', status['last_return'], step=step)
                        tf.summary.scalar(f'Workers/Worker_{wid}_Best_Val', status['best_val'], step=step)
                
                # Communication performance
                if communication_times:
                    recent_comm_times = communication_times[-100:]  # Last 100 messages
                    tf.summary.scalar('Performance/Avg_Message_Time_ms', np.mean(recent_comm_times) * 1000, step=step)
                    tf.summary.scalar('Performance/Total_Messages', len(communication_times), step=step)
                
                # Throughput metrics
                if elapsed_training_time > 0:
                    tf.summary.scalar('Performance/Episodes_Per_Second', total_episodes / elapsed_training_time, step=step)
                    tf.summary.scalar('Performance/Episodes_Per_Minute', total_episodes / (elapsed_training_time / 60), step=step)
            
            last_log_time = current_time
        
        loop_time = time.perf_counter() - loop_start

    training_time = time.perf_counter() - training_start_time
    phase_times['training'] = training_time

    # ------------------------------------------------------------------ #
    #  Phase 5: Finalization
    # ------------------------------------------------------------------ #
    _LOG.info("Phase 5: Finalizing training...")
    finalization_start = time.perf_counter()
    if sync_mode == "parameter_server" and ps_queue is not None:
        ps_queue.put(("stop", None))
        _, best_weights = ps_queue.get()
        ps_proc.join()

    if best_weights is None:
        _LOG.warning("No weights returned – nothing saved.")
        return

    # save final model
    out_dir = Path(save_dir or cfg.path.base_save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    from tradingbot.agents import SACAgent
    final_agent = SACAgent(obs_space, act_space)
    for v in final_agent.actor.variables:
        v.assign(best_weights[v.name])
    final_agent.save(out_dir / "checkpoint")

    phase_times['finalization'] = time.perf_counter() - finalization_start
    total_time = time.perf_counter() - coordinator_start_time

    _LOG.info("Parallel training finished in %.1fs – model saved to %s",
            time.perf_counter() - total_time, out_dir)
    
    # ------------------------------------------------------------------ #
    #  Final TensorBoard logging
    # ------------------------------------------------------------------ #
    with coordinator_writer.as_default():
        final_step = int(total_time)
        
        # Final summary metrics
        tf.summary.scalar('Summary/Total_Time_s', total_time, step=final_step)
        tf.summary.scalar('Summary/Total_Episodes', total_episodes, step=final_step)
        tf.summary.scalar('Summary/Best_Validation_Return', best_val, step=final_step)
        tf.summary.scalar('Summary/Final_Mean_Return', np.mean(total_returns) if total_returns else 0, step=final_step)
        
        # Phase timing breakdown
        for phase_name, phase_time in phase_times.items():
            tf.summary.scalar(f'Phases/{phase_name}_time_s', phase_time, step=final_step)
            tf.summary.scalar(f'Phases/{phase_name}_percent', (phase_time / total_time) * 100, step=final_step)
        
        # Final worker statistics
        for wid, status in worker_status.items():
            tf.summary.scalar(f'Final_Workers/Worker_{wid}_Total_Episodes', status['episodes'], step=final_step)
            tf.summary.scalar(f'Final_Workers/Worker_{wid}_Final_Best_Val', status['best_val'], step=final_step)
        
        # Performance summary
        if total_time > 0:
            tf.summary.scalar('Summary/Episodes_Per_Hour', total_episodes / (total_time / 3600), step=final_step)
            tf.summary.scalar('Summary/Training_Efficiency', training_time / total_time, step=final_step)
        
        # Configuration logging for reproducibility
        tf.summary.scalar('Config/Num_Workers', cfg.parallel.num_workers, step=final_step)
        tf.summary.scalar('Config/Steps_Per_Env', cfg.parallel.steps_per_env, step=final_step)
        tf.summary.scalar('Config/Push_Every', cfg.parallel.push_every, step=final_step)
        tf.summary.scalar('Config/Batch_Size', cfg.training.batch_size, step=final_step)
        tf.summary.scalar('Config/Learning_Rate', cfg.agent.learning_rate, step=final_step)
        tf.summary.scalar('Config/Gamma', cfg.agent.gamma, step=final_step)
        tf.summary.scalar('Config/Tau', cfg.agent.tau, step=final_step)

    # Close TensorBoard writer
    coordinator_writer.close()

    _LOG.info("=" * 80)
    _LOG.info("TRAINING COMPLETED")
    _LOG.info("=" * 80)
    _LOG.info("Total time: %.1fs (%.1f minutes)", total_time, total_time / 60)
    _LOG.info("Total episodes: %d", total_episodes)
    _LOG.info("Best validation return: %.4f", best_val)
    _LOG.info("Model saved to: %s", out_dir)
    _LOG.info("TensorBoard logs: %s", coordinator_log_dir.parent)
    _LOG.info("=" * 80)
    
    # Print phase breakdown
    _LOG.info("Phase timing breakdown:")
    for phase_name, phase_time in phase_times.items():
        _LOG.info("  %-20s: %6.1fs (%4.1f%%)", phase_name.replace('_', ' ').title(), 
                 phase_time, (phase_time / total_time) * 100)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"📊 View results: tensorboard --logdir {coordinator_log_dir.parent}")
    print(f"💾 Model saved: {out_dir}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation: {best_val:.4f}")
    print("="*60)



# --------------------------------------------------------------------------- #

