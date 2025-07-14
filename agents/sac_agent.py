# agents/sac_agent.py
from __future__ import annotations
import os, pickle, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=all, 1=info, 2=warning, 3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN banner
from datetime import datetime
from typing import Any, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gymnasium import spaces
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras import layers  
from tensorflow.keras.optimizers import Adam


from tradingbot import cfg, get_logger
from tradingbot.models import (build_actor, build_critic, expand_dims_channel, SACReplayBuffer,)

_LOG = get_logger("agents.sac")

if cfg.general.use_mixed_precision:
    mixed_precision.set_global_policy("mixed_float16")
    _LOG.info("Mixed‑precision policy 'mixed_float16' activated")

# ------------------------------------------------------------------ #
#  cfg helpers
# ------------------------------------------------------------------ #

LR   = cfg.agent.learning_rate
GAMMA = cfg.agent.gamma
TAU   = cfg.agent.tau
BUFFER_CAP = cfg.agent.replay_buffer_capacity
MIN_RB = cfg.agent.min_buffer_size # minimum replay buffer size to start training
LOG_DIR_BASE = cfg.path.log_dir_base

# ------------------------------------------------------------------ #
#  utils                                                             #
# ------------------------------------------------------------------ #
_fp32 = tf.float32

def _cast_fp32(x: tf.Tensor) -> tf.Tensor:  # keep this tiny for tf.function
    return tf.cast(x, _fp32)

def _to_model_inputs(state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Rename keys from replay‑buffer format to Keras Functional API inputs."""
    return {
        "shared_features_input":    state["shared_features"],
        "ticker_features_input":    state["ticker_features"],
        "portfolio_holdings_input": state["portfolio_holdings"],
        "cash_balance_input":       state["cash_balance"],
    }

class SACAgent:
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, *, load_path: str | None = None, worker_id: int = 0, inference_only: bool = False):
        import tensorflow as tf
        import tensorflow_probability as tfp
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("observation_space must be a Gymnasium Dict")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("action_space must be a Gymnasium Box")
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.worker_id = worker_id
        self.is_inference_only = inference_only # Store the flag


        # --------------- common scalar hyperparameters ------------------ #
        self.gamma        = GAMMA
        self.tau          = TAU
        self.lr           = LR
        self.action_dim   = int(np.prod(action_space.shape))
        self.min_rb_size  = MIN_RB

        # --------------- temperature (α) -------------------------------- #
        self.log_alpha      = tf.Variable(0.0, trainable=True, dtype=_fp32)
        self.target_entropy = -float(self.action_dim)
        
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        # --------------- replay buffer ---------------------------------- #
        if not self.is_inference_only:
            self.replay_buffer = SACReplayBuffer(
                capacity=BUFFER_CAP,
                state_shape_dict=observation_space.spaces,
                action_dim=self.action_dim,
            )
        else:
            self.replay_buffer = None # Don't create a buffer in inference mode
            
        # TensorBoard setup with worker-specific logging
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(LOG_DIR_BASE, f"SAC_{ts}", f"worker_{worker_id}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        _LOG.info("TensorBoard logs → %s", self.log_dir)

        # Performance tracking variables
        self.timing_stats = {
            'action_selection_time': [],
            'gradient_update_time': [],
            'experience_save_time': [],
            'target_update_time': [],
            'total_step_time': []
        }
        
        # Training metrics
        self.episode_returns = []
        self.episode_lengths = []
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.total_gradient_updates = 0

        # --------------- networks & opts -------------------------------- #

        if load_path and self._check_model_files_exist(load_path):
            _LOG.info("Loading agent from %s", load_path)
            try:
                self.load(load_path)
                return
            except Exception as e:
                _LOG.warning("Failed to load (%s). Re-initialising.", e)

        self._initialize_new_model(observation_space.spaces)

    def _check_model_files_exist(self, path_base):
        required_suffixes = [
            "_actor.keras", "_critic1.keras", "_critic2.keras",
            "_target_critic1.keras", "_target_critic2.keras", "_params.pkl"
        ]
        return all(os.path.exists(f"{path_base}{suffix}") for suffix in required_suffixes)
    
    def _wrap_opt(self, opt: Adam):
        """Wrap optimizer for mixed precision if enabled."""
        if cfg.general.use_mixed_precision:
            return mixed_precision.LossScaleOptimizer(opt)
        return opt

    def _initialize_new_model(self, space_dict: Dict[str, spaces.Space]) -> None:
        # ---------- build networks ------------------------------------- #
        self.actor        = build_actor(space_dict, self.action_dim)
        self.critic_1     = build_critic(space_dict, self.action_dim, name="critic_1")
        self.critic_2     = build_critic(space_dict, self.action_dim, name="critic_2")
        self.target_c1    = build_critic(space_dict, self.action_dim, name="t_c1")
        self.target_c2    = build_critic(space_dict, self.action_dim, name="t_c2")
        self.target_c1.set_weights(self.critic_1.get_weights())
        self.target_c2.set_weights(self.critic_2.get_weights())

        # ---------- mixed‑precision opts -------------------------------- #
        base = lambda: tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
        self.actor_opt = self._wrap_opt(base())
        self.c1_opt    = self._wrap_opt(base())
        self.c2_opt    = self._wrap_opt(base())
        self.alpha_opt = self._wrap_opt(base())

        _LOG.info("New SAC agent initialised – networks & optimisers ready.")

    def select_action(self, state: dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        """Return a clipped NumPy action for an *unbatched* state dict."""
        start_time = time.perf_counter()
        # 1) convert to tensors and add batch dim
        state_tf = {k: tf.expand_dims(tf.convert_to_tensor(v, tf.float32), 0)
                    for k, v in state.items()}

        # 2) remap keys to the model’s input names
        model_inputs = _to_model_inputs(state_tf)        # {'shared_features_input': …, …}

        # 3) forward pass
        means, log_std = self.actor(model_inputs, training=False)

        if evaluate:
            action = tf.tanh(means)
        else:
            action, _ = self._sample_actions_and_log_probs(means, log_std)

        # 4) clip + sanity-check
        action_np = np.clip(action.numpy().squeeze(0),
                            self.action_space.low,
                            self.action_space.high)

        if np.isnan(action_np).any() or np.isinf(action_np).any():
            print("WARNING: NaN/Inf detected in final action. Clipping.")
            action_np = np.nan_to_num(
                action_np,
                nan=0.0,
                posinf=self.action_space.high[0],
                neginf=self.action_space.low[0],
            )
        # Track timing
        action_time = time.perf_counter() - start_time
        self.timing_stats['action_selection_time'].append(action_time)

        return action_np

    @tf.function
    def _sample_actions_and_log_probs(self, mean, log_std):
        log_std = tf.clip_by_value(log_std, -5.0, 2.0)
        std = tf.exp(tf.cast(log_std, tf.float32))           # fp32 here
        std = tf.cast(std, mean.dtype)                       # back to fp16

        # 2. re-parameterise
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        action = tf.tanh(mean + std * noise)

        # 3. NaN/Inf guard (should now be redundant but keeps env log clean)
        action = tf.where(tf.math.is_finite(action), action,
                        tf.zeros_like(action))

    @tf.function
    def _train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Per‑batch training using explicit loss‑scaling."""
        # -------------- unpack & cast ---------------------------------- #
        s      = _to_model_inputs(batch["state"])
        s2     = _to_model_inputs(batch["next_state"])
        a      = _cast_fp32(batch["action"])
        r      = _cast_fp32(batch["reward"])
        done   = _cast_fp32(batch["done"])

        # -------------- targets ---------------------------------------- #
        m2, ls2 = self.actor(s2, training=False)
        a2, lp2 = self._sample_actions_and_log_probs(m2, ls2)
        lp2     = _cast_fp32(lp2)

        q1_t = _cast_fp32(self.target_c1({**s2, "action_input": a2}, training=False))
        q2_t = _cast_fp32(self.target_c2({**s2, "action_input": a2}, training=False))
        alpha = tf.exp(self.log_alpha)
        q_tar = tf.minimum(q1_t, q2_t) - alpha * lp2
        q_tar = r + self.gamma * (1.0 - done) * q_tar

        # =============================================================== #
        # 1) Critic‑1 update                                             #
        # =============================================================== #
        with tf.GradientTape() as tape:
            q1 = _cast_fp32(self.critic_1({**s, "action_input": a}, training=True))
            loss_c1 = tf.reduce_mean(tf.square(q1 - q_tar))
        
        g1 = tape.gradient(loss_c1, self.critic_1.trainable_variables)
        self.c1_opt.apply_gradients(zip(g1, self.critic_1.trainable_variables))

        # =============================================================== #
        # 2) Critic‑2 update                                             #
        # =============================================================== #
        with tf.GradientTape() as tape:
            q2 = _cast_fp32(self.critic_2({**s, "action_input": a}, training=True))
            loss_c2 = tf.reduce_mean(tf.square(q2 - q_tar))
        
        g2 = tape.gradient(loss_c2, self.critic_2.trainable_variables)
        self.c2_opt.apply_gradients(zip(g2, self.critic_2.trainable_variables))

        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean(
                tf.square(q1 - q_tar) + tf.square(q2 - q_tar)
            )

        # =============================================================== #
        # 3) Actor update                                                #
        # =============================================================== #
        with tf.GradientTape() as tape:
            m, ls = self.actor(s, training=True)
            a_new, lp = self._sample_actions_and_log_probs(m, ls)
            lp       = _cast_fp32(lp)
            q1_new   = _cast_fp32(self.critic_1({**s, "action_input": a_new}, training=False))
            q2_new   = _cast_fp32(self.critic_2({**s, "action_input": a_new}, training=False))
            q_new    = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(alpha * lp - q_new)
        ga = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(ga, self.actor.trainable_variables))

        # =============================================================== #
        # 4) Temperature (α) update                                      #
        # =============================================================== #
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * (lp + self.target_entropy))
        galpha = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(galpha, [self.log_alpha]))

        # -------- soft target update ----------------------------------- #
        tau = tf.cast(self.tau, _fp32)
        for tw, w in zip(self.target_c1.weights, self.critic_1.weights):
            tw.assign(tau * w + (1.0 - tau) * tw)
        for tw, w in zip(self.target_c2.weights, self.critic_2.weights):
            tw.assign(tau * w + (1.0 - tau) * tw)

        self.step_counter.assign_add(1)
        self._update_targets()   # τ-weighted Polyak averaging
        
        return dict(
            critic_loss=critic_loss, 
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            alpha=tf.exp(self.log_alpha),
            q1_mean=tf.reduce_mean(q1),
            q2_mean=tf.reduce_mean(q2),
            reward_mean=tf.reduce_mean(r)
        )

    def _update_targets(self):
        start_time = time.perf_counter()
        
        tau = tf.cast(self.tau, tf.float32)  # ensures dtype match under mixed precision
        for target_w, source_w in zip(self.target_c1.trainable_variables, self.critic_1.trainable_variables):
            target_w.assign(self.tau * source_w + (1.0 - self.tau) * target_w)
        for target_w, source_w in zip(self.target_c2.trainable_variables, self.critic_2.trainable_variables):
            target_w.assign(self.tau * source_w + (1.0 - self.tau) * target_w)
        # Track timing
        update_time = time.perf_counter() - start_time
        self.timing_stats['target_update_time'].append(update_time)

    def update(self, batch_size: int) -> dict[str, float]:
        start_time = time.perf_counter()
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        metrics = self._train_step({
            "state":      s,
            "action":     a,
            "reward":     r,
            "next_state": s2,
            "done":       d,
        }) 

        # Track timing and metrics
        update_time = time.perf_counter() - start_time
        self.timing_stats['gradient_update_time'].append(update_time)
        self.total_gradient_updates += 1
        
        # Log to TensorBoard every 100 updates
        if self.total_gradient_updates % 100 == 0:
            self._log_training_metrics(metrics)
        # convert to Python scalars for logs
        return {k: float(v.numpy()) for k, v in metrics.items()}

    def save_experience(self, state, action, reward, next_state, done):
        start_time = time.perf_counter()
        
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Track timing
        save_time = time.perf_counter() - start_time
        self.timing_stats['experience_save_time'].append(save_time)
        
        # Track episode metrics
        self.current_episode_return += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode metrics
            with self.summary_writer.as_default():
                tf.summary.scalar('Episode/Return', self.current_episode_return, 
                                step=len(self.episode_returns))
                tf.summary.scalar('Episode/Length', self.current_episode_length, 
                                step=len(self.episode_returns))
                tf.summary.scalar('Episode/Buffer_Size', self.replay_buffer.size, 
                                step=len(self.episode_returns))
            
            # Reset episode tracking
            self.current_episode_return = 0.0
            self.current_episode_length = 0

    def _log_training_metrics(self, metrics: dict):
        """Log training metrics to TensorBoard"""
        with self.summary_writer.as_default():
            step = self.total_gradient_updates
            
            # Training losses
            tf.summary.scalar('Training/Critic_Loss', metrics['critic_loss'], step=step)
            tf.summary.scalar('Training/Actor_Loss', metrics['actor_loss'], step=step)
            tf.summary.scalar('Training/Alpha_Loss', metrics['alpha_loss'], step=step)
            tf.summary.scalar('Training/Alpha_Value', metrics['alpha'], step=step)
            tf.summary.scalar('Training/Q1_Mean', metrics['q1_mean'], step=step)
            tf.summary.scalar('Training/Q2_Mean', metrics['q2_mean'], step=step)
            tf.summary.scalar('Training/Reward_Mean', metrics['reward_mean'], step=step)
            
            # Replay buffer stats
            tf.summary.scalar('Buffer/Size', self.replay_buffer.size, step=step)
            tf.summary.scalar('Buffer/Utilization', 
                            self.replay_buffer.size / self.replay_buffer.capacity, step=step)
            
            # Performance timing (average of last 100 measurements)
            if len(self.timing_stats['gradient_update_time']) >= 100:
                recent_times = self.timing_stats['gradient_update_time'][-100:]
                tf.summary.scalar('Performance/Gradient_Update_Time_ms', 
                                np.mean(recent_times) * 1000, step=step)
            
            if len(self.timing_stats['action_selection_time']) >= 100:
                recent_times = self.timing_stats['action_selection_time'][-100:]
                tf.summary.scalar('Performance/Action_Selection_Time_ms', 
                                np.mean(recent_times) * 1000, step=step)
            
            if len(self.timing_stats['target_update_time']) >= 100:
                recent_times = self.timing_stats['target_update_time'][-100:]
                tf.summary.scalar('Performance/Target_Update_Time_ms', 
                                np.mean(recent_times) * 1000, step=step)
            
            # Episode statistics (if we have recent episodes)
            if len(self.episode_returns) >= 10:
                recent_returns = self.episode_returns[-10:]
                tf.summary.scalar('Performance/Recent_Episode_Return_Mean', 
                                np.mean(recent_returns), step=step)
                tf.summary.scalar('Performance/Recent_Episode_Return_Std', 
                                np.std(recent_returns), step=step)

    def log_performance_summary(self):
        """Log a summary of performance statistics"""
        with self.summary_writer.as_default():
            step = self.total_gradient_updates
            
            # Overall timing statistics
            for timing_key, times in self.timing_stats.items():
                if times:
                    tf.summary.scalar(f'Summary/{timing_key}_mean_ms', 
                                    np.mean(times) * 1000, step=step)
                    tf.summary.scalar(f'Summary/{timing_key}_std_ms', 
                                    np.std(times) * 1000, step=step)
                    tf.summary.scalar(f'Summary/{timing_key}_total_s', 
                                    np.sum(times), step=step)
            
            # Episode statistics
            if self.episode_returns:
                tf.summary.scalar('Summary/Total_Episodes', len(self.episode_returns), step=step)
                tf.summary.scalar('Summary/Mean_Episode_Return', np.mean(self.episode_returns), step=step)
                tf.summary.scalar('Summary/Best_Episode_Return', np.max(self.episode_returns), step=step)
                tf.summary.scalar('Summary/Worst_Episode_Return', np.min(self.episode_returns), step=step)

    def save(self, path_base):
        try:
             dir_name = os.path.dirname(path_base)
             if dir_name:
                 os.makedirs(dir_name, exist_ok=True)

             print(f"Saving agent networks to {path_base}...")
             self.actor.save(f"{path_base}_actor.keras")
             self.critic_1.save(f"{path_base}_critic1.keras")
             self.critic_2.save(f"{path_base}_critic2.keras")
             self.target_c1.save(f"{path_base}_target_c1.keras")
             self.target_c2.save(f"{path_base}_target_c2.keras")

             print(f"Saving agent parameters to {path_base}_params.pkl...")
             params_to_save = {
                 "step_counter": self.step_counter.numpy(),
                 "log_alpha": self.log_alpha.numpy(),
                 "target_entropy": self.target_entropy,
                 "replay_buffer_states": self.replay_buffer.states,
                 "replay_buffer_actions": self.replay_buffer.actions,
                 "replay_buffer_rewards": self.replay_buffer.rewards,
                 "replay_buffer_next_states": self.replay_buffer.next_states,
                 "replay_buffer_dones": self.replay_buffer.dones,
                 "replay_buffer_position": self.replay_buffer.position,
                 "replay_buffer_size": self.replay_buffer.size,
             }
             with open(f"{path_base}_params.pkl", "wb") as f:
                 pickle.dump(params_to_save, f)
             print(f"Agent saved successfully to {path_base}")
        except Exception as e:
             print(f"Error saving agent to {path_base}: {e}")

    def load(self, path_base):
        if not self._check_model_files_exist(path_base):
            raise FileNotFoundError(f"Cannot load agent. Required files not found at base path: {path_base}")
        try:
            print(f"Loading agent networks from {path_base}...")
            # expand_dims_channel is needed for ConvLSTM2D Lambda layer
            custom_objects = {'expand_dims_channel': expand_dims_channel}
            self.actor = tf.keras.models.load_model(f"{path_base}_actor.keras", custom_objects=custom_objects, safe_mode=False)
            self.critic_1 = tf.keras.models.load_model(f"{path_base}_critic1.keras", custom_objects=custom_objects, safe_mode=False)
            self.critic_2 = tf.keras.models.load_model(f"{path_base}_critic2.keras", custom_objects=custom_objects, safe_mode=False)
            self.target_c1 = tf.keras.models.load_model(f"{path_base}_target_c1.keras", custom_objects=custom_objects, safe_mode=False)
            self.target_c2 = tf.keras.models.load_model(f"{path_base}_target_c2.keras", custom_objects=custom_objects, safe_mode=False)

            print(f"Loading agent parameters from {path_base}_params.pkl...")
            with open(f"{path_base}_params.pkl", "rb") as f:
                params = pickle.load(f)

            self.step_counter.assign(params["step_counter"])
            self.log_alpha.assign(params["log_alpha"])
            self.alpha.assign(tf.exp(self.log_alpha))
            self.target_entropy = params["target_entropy"]

            if "replay_buffer_position" in params:
                 print("Restoring replay buffer state...")
                 self.replay_buffer.states = params["replay_buffer_states"]
                 self.replay_buffer.actions = params["replay_buffer_actions"]
                 self.replay_buffer.rewards = params["replay_buffer_rewards"]
                 self.replay_buffer.next_states = params["replay_buffer_next_states"]
                 self.replay_buffer.dones = params["replay_buffer_dones"]
                 self.replay_buffer.position = params["replay_buffer_position"]
                 self.replay_buffer.size = params["replay_buffer_size"]
            else:
                 print("Replay buffer state not found in saved params. Starting with empty buffer.")

            print("Initializing optimizers...")
            # Re-initialize optimizers (state is not typically saved with model.save for Adam directly)
            base_actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_critic_1_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_critic_2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            base_alpha_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            self.actor_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_actor_optimizer)
            self.critic_1_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_critic_1_optimizer)
            self.critic_2_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_critic_2_optimizer)
            self.alpha_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_alpha_optimizer)


            print(f"Agent loaded successfully from {path_base}")
        except Exception as e:
            print(f"Error loading agent from {path_base}: {e}")
            raise e