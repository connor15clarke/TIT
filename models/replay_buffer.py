# models/replay_buffer.py
import numpy as np
import tensorflow as tf

from tradingbot import cfg, get_logger

_LOG = get_logger("models.replay_buffer")


def _cfg_capacity(default: int = 1_000_000) -> int:
    try:
        return cfg.agent.replay_buffer_capacity
    except AttributeError:
        return default

class SACReplayBuffer:
    def __init__(self, capacity: int | None, state_shape_dict, action_dim: int):
        self.capacity = capacity or _cfg_capacity()
        self.position = 0
        self.size = 0 

        expected_keys = ['shared_features', 'ticker_features', 'portfolio_holdings', 'cash_balance']
        if not all(key in state_shape_dict for key in expected_keys):
             raise ValueError(f"state_shape_dict missing one or more expected keys: {expected_keys}")

        self.states = {}
        self.next_states = {}
        for key, shape_obj in state_shape_dict.items():
             shape = shape_obj.shape
             dtype = shape_obj.dtype
             self.states[key] = np.zeros((capacity, *shape), dtype=dtype)
             self.next_states[key] = np.zeros((capacity, *shape), dtype=dtype)

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        _LOG.info("Replay buffer ready – capacity %d entries", self.capacity)

    def push(self, state, action, reward, next_state, done):
        for key in self.states.keys():
            if key not in state or key not in next_state:
                 print(f"Warning: Key '{key}' not found in state or next_state during push.")
                 continue
            self.states[key][self.position] = state[key]
            self.next_states[key][self.position] = next_state[key]

        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} elements, only {self.size} available.")

        batch_indices = np.random.choice(self.size, batch_size, replace=False)

        batch_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.states.items()
        }
        batch_next_states = {
            k: tf.convert_to_tensor(v[batch_indices], dtype=tf.float32)
            for k, v in self.next_states.items()
        }

        batch_actions = tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.float32)
        batch_rewards = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        batch_dones = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size