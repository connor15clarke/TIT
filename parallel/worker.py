
# tradingbot/parallel/worker.py
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=all, 1=info, 2=warning, 3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN banner
import time
import math
from pathlib import Path
from typing import Any

import numpy as np

from tradingbot.config import cfg
from tradingbot.logger import get_logger

# --------------------------------------------------------------------------- #
#  quick aliases (all hyper-params live in cfg)
# --------------------------------------------------------------------------- #
EPISODES              = cfg.training.episodes
BATCH_SIZE            = cfg.training.batch_size
MIN_BUFFER            = cfg.agent.min_buffer_size
UPDATES_PER_STEP      = cfg.training.updates_per_step
SAVE_FREQ             = cfg.training.save_frequency
WINDOW_SIZE           = cfg.training.window_size
INITIAL_BALANCE       = cfg.training.initial_balance
LEARNING_RATE         = cfg.agent.learning_rate
BASE_SAVE_PATH        = cfg.path.base_save_path
LOG_DIR_BASE          = cfg.path.log_dir_base
GLOBAL_SEED           = cfg.general.global_seed
USE_MIXED_PRECISION   = cfg.general.use_mixed_precision
STEPS_PER_ENV         = cfg.parallel.steps_per_env
PUSH_EVERY            = cfg.parallel.push_every or STEPS_PER_ENV  

# --------------------------------------------------------------------------- #
def _pin_gpu(gpu_index: int) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    gpu_index %= len(gpus)
    tf.config.set_visible_devices(gpus[gpu_index], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

    LIMITS = {0: 8*1024, 1: 6*1024}   # MiB per physical GPU id
    mem_limit = LIMITS.get(gpu_index, 4096)   # fallback 4 GiB
    tf.config.set_logical_device_configuration(
        gpus[gpu_index],
        [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)]
    )


# --------------------------------------------------------------------------- #
def worker_process(
        pipe,
        train_data: np.ndarray,
        val_data:   np.ndarray,
        env_kwargs: dict[str, Any],
        worker_id:  int,
        gpu_id:     int, 
) -> None:
    """
    Full SAC training loop (episodes, warm-up, validation, checkpoints) that
    mirrors the old *titparallelenv* behaviour.

    Communication contract with the coordinator **unchanged**:
      • send ("episode_return", float) after every finished episode
      • send ("weights", {var.name: np.ndarray}) once before exit
    """
    # ------------------------------------------------------------------ #
    #  1) seed & GPU placement BEFORE importing TF/Keras
    # ------------------------------------------------------------------ #
    np.random.seed(GLOBAL_SEED + worker_id)


    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)     
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
        
    # TF import comes *after* the environment vars
    _pin_gpu(0)
    import tensorflow as tf
    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from tradingbot.env import TradingEnv
    from tradingbot.agents import SACAgent
    from tradingbot.logger import get_logger

    log = get_logger(f"worker-{worker_id}")
    log.info("TensorFlow %s – visible GPUs: %s",
             tf.__version__, tf.config.list_physical_devices("GPU"))
    
    # ------------------------------------------------------------------ #
    #  2) TensorBoard setup for worker-level metrics
    # ------------------------------------------------------------------ #
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    worker_log_dir = Path(LOG_DIR_BASE) / f"SAC_{ts}" / f"worker_{worker_id}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    worker_writer = tf.summary.create_file_writer(str(worker_log_dir))
    
    # Timing tracking dictionaries
    timing_stats = {
        'env_step_time': [],
        'action_selection_time': [],
        'env_reset_time': [],
        'gradient_updates_time': [],
        'validation_time': [],
        'checkpoint_save_time': [],
        'weight_push_time': [],
        'total_episode_time': []
    }

    # ------------------------------------------------------------------ #
    #  3) build dedicated env + agent
    # ------------------------------------------------------------------ #

    # paths for checkpoints
    ckpt_dir = Path(BASE_SAVE_PATH) / f"worker-{worker_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_chk = ckpt_dir / "checkpoint"
    ckpt_best = ckpt_dir / "best"

    ckpt_params = ckpt_chk.with_name(ckpt_chk.name + "_params.pkl")

    # Environment setup timing
    env_setup_start = time.perf_counter()

    train_env = TradingEnv(data=train_data,
                            **env_kwargs,              # already carries window_size, initial_balance, etc.
                            bot_logger=None,
                            seed=GLOBAL_SEED + worker_id,)

    val_env = TradingEnv(data=val_data,
                        **env_kwargs,
                        bot_logger=None,
                        is_validation=True,
                        seed=GLOBAL_SEED + 1_000 + worker_id,)

    agent = SACAgent(train_env.observation_space,
                     train_env.action_space,
                     load_path=ckpt_params if ckpt_params.exists() else None,
                     worker_id=worker_id,)
    
    agent.min_replay_size_to_train = MIN_BUFFER

    env_setup_time = time.perf_counter() - env_setup_start
    
    # Log environment setup time
    with worker_writer.as_default():
        tf.summary.scalar('Setup/Environment_Setup_Time_s', env_setup_time, step=0)

    # try resuming
    if ckpt_params.exists():
         agent.load(ckpt_chk)

    # ------------------------------------------------------------------ #
    #  4) main training loop
    # ------------------------------------------------------------------ #
    best_val: float = -np.inf
    env_steps: int = 0
    episode_count: int = 0
    push_counter: int = 0

    worker_start_time = time.perf_counter()
    log.info(f"Worker {worker_id} starting training for {STEPS_PER_ENV} steps")

    while env_steps < STEPS_PER_ENV:
        # --- 4.1 start a mini‑episode with timing ---------------------------------
        episode_start_time = time.perf_counter()
        episode_count += 1
        
        # Environment reset timing
        reset_start = time.perf_counter()
        state, _ = train_env.reset()
        reset_time = time.perf_counter() - reset_start
        timing_stats['env_reset_time'].append(reset_time)
        
        ep_return = 0.0
        episode_steps = 0

        while True:
            step_start_time = time.perf_counter()
            
            # Action selection timing
            action_start = time.perf_counter()
            action = agent.select_action(state)
            action_time = time.perf_counter() - action_start
            timing_stats['action_selection_time'].append(action_time)

            # Environment step timing
            env_step_start = time.perf_counter()
            next_state, reward, term, trunc, _ = train_env.step(action)
            env_step_time = time.perf_counter() - env_step_start
            timing_stats['env_step_time'].append(env_step_time)

            if reward is None or (isinstance(reward, float) and math.isnan(reward)):
               reward = 0.0
            done = term or trunc

            agent.save_experience(state, action, reward, next_state, done)
            state = next_state
            ep_return += float(reward)
            env_steps += 1

            # gradient updates ---------------------------------------
            if agent.replay_buffer.size >= MIN_BUFFER:
                updates_start = time.perf_counter()
                for _ in range(UPDATES_PER_STEP):
                    agent.update(batch_size=BATCH_SIZE)
                updates_time = time.perf_counter() - updates_start
                timing_stats['gradient_updates_time'].append(updates_time)
            
            total_step_time = time.perf_counter() - step_start_time

            if done or env_steps >= STEPS_PER_ENV:
                pipe.send(("episode_return", float(ep_return)))
                break  # end mini‑episode

        episode_total_time = time.perf_counter() - episode_start_time
        timing_stats['total_episode_time'].append(episode_total_time)

        # --------------------- validation + checkpoints ----------------
        if agent.replay_buffer.size >= MIN_BUFFER:
            if env_steps % SAVE_FREQ == 0:
                checkpoint_start = time.perf_counter()
                agent.save(ckpt_chk)
                checkpoint_time = time.perf_counter() - checkpoint_start
                timing_stats['checkpoint_save_time'].append(checkpoint_time)

            # one deterministic episode on *unseen* slice
            # Validation timing
            validation_start = time.perf_counter()
            v_state, _ = val_env.reset()
            v_ret = 0.0
            v_done = False
            v_steps = 0
            
            while not v_done and v_steps < 1000:
                v_action = agent.select_action(v_state, evaluate=True)
                v_state, r, term, trunc, _ = val_env.step(v_action)
                if r is None or (isinstance(r, float) and math.isnan(r)):
                    r = 0.0
                v_done = term or trunc
                v_ret += float(r)
                v_steps += 1
            validation_time = time.perf_counter() - validation_start
            timing_stats['validation_time'].append(validation_time)

            if v_ret > best_val:
                best_val = v_ret
                agent.save(ckpt_best)
        
        # --- 4.2 push weights if quota met -----------------------------
        if env_steps >= STEPS_PER_ENV or env_steps >= (push_counter + 1) * PUSH_EVERY:
            push_start = time.perf_counter()
            weights = {v.name: v.numpy() for v in agent.actor.variables}
            pipe.send(("push", {"weights": weights, "val": best_val}))
            push_time = time.perf_counter() - push_start
            timing_stats['weight_push_time'].append(push_time)
            push_counter += 1
            push_counter += 1

        # --- 4.3 Log episode metrics to TensorBoard every 10 episodes -------
        if episode_count % 10 == 0:
            with worker_writer.as_default():
                step = env_steps
                
                # Episode metrics
                tf.summary.scalar('Episode/Return', ep_return, step=step)
                tf.summary.scalar('Episode/Length', episode_steps, step=step)
                tf.summary.scalar('Episode/Best_Validation', best_val, step=step)
                tf.summary.scalar('Episode/Buffer_Size', agent.replay_buffer.size, step=step)
                tf.summary.scalar('Episode/Total_Episodes', episode_count, step=step)
                
                # Timing metrics (averages of last 10 measurements)
                for timing_key, times in timing_stats.items():
                    if len(times) >= 10:
                        recent_times = times[-10:]
                        tf.summary.scalar(f'Timing/{timing_key}_mean_ms', 
                                        np.mean(recent_times) * 1000, step=step)
                        tf.summary.scalar(f'Timing/{timing_key}_total_s', 
                                        np.sum(recent_times), step=step)
                
                # Progress tracking
                progress = env_steps / STEPS_PER_ENV
                tf.summary.scalar('Progress/Steps_Completed', env_steps, step=step)
                tf.summary.scalar('Progress/Percent_Complete', progress * 100, step=step)
                
                # Throughput metrics
                elapsed_time = time.perf_counter() - worker_start_time
                if elapsed_time > 0:
                    tf.summary.scalar('Performance/Steps_Per_Second', env_steps / elapsed_time, step=step)
                    tf.summary.scalar('Performance/Episodes_Per_Minute', episode_count / (elapsed_time / 60), step=step)

        log.debug("w%02d | ep %d | ret %.3f | val %.3f | steps %d/%d | eps_time %.3fs",
                  worker_id, episode_count, ep_return, best_val, env_steps, STEPS_PER_ENV, 
                  episode_total_time)

    # ------------------------------------------------------------------ #
    #  5) Final logging and cleanup
    # ------------------------------------------------------------------ #
    total_worker_time = time.perf_counter() - worker_start_time
    
    # Log final summary statistics
    with worker_writer.as_default():
        final_step = env_steps
        tf.summary.scalar('Summary/Total_Training_Time_s', total_worker_time, step=final_step)
        tf.summary.scalar('Summary/Total_Episodes', episode_count, step=final_step)
        tf.summary.scalar('Summary/Final_Buffer_Size', agent.replay_buffer.size, step=final_step)
        tf.summary.scalar('Summary/Best_Validation_Return', best_val, step=final_step)
        
        # Final timing summaries
        for timing_key, times in timing_stats.items():
            if times:
                tf.summary.scalar(f'Summary/{timing_key}_total_time_s', np.sum(times), step=final_step)
                tf.summary.scalar(f'Summary/{timing_key}_mean_ms', np.mean(times) * 1000, step=final_step)
                tf.summary.scalar(f'Summary/{timing_key}_count', len(times), step=final_step)
        
        # Performance summary
        tf.summary.scalar('Summary/Average_Steps_Per_Second', env_steps / total_worker_time, step=final_step)
        tf.summary.scalar('Summary/Average_Episodes_Per_Hour', episode_count / (total_worker_time / 3600), step=final_step)

        log.debug("w%02d | ep %d | ret %.3f | val %.3f | steps %d/%d",
                  worker_id, episode_count, ep_return, best_val, env_steps, STEPS_PER_ENV)

    # ------------------------------------------------------------------ #
    #  6) flush final weights back to coordinator
    # ------------------------------------------------------------------ #
    final_push_start = time.perf_counter()
    weights = {v.name: v.numpy() for v in agent.actor.variables}
    pipe.send(("done", {"weights": weights, "val": best_val}))
    final_push_time = time.perf_counter() - final_push_start
    worker_writer.close()
    pipe.close()
    log.info("Worker %d finished – best validation return = %.4f, total time = %.2fs", 
             worker_id, best_val, total_worker_time)