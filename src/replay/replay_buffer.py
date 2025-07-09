import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, sequence_length, batch_size, observation_shape, action_dim, device, oversample_ends=False):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        self.oversample_ends = oversample_ends

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self._next_idx = 0
        self._size = 0

    def add(self, obs, action, reward, done):
        self.observations[self._next_idx] = obs
        self.actions[self._next_idx] = action
        self.rewards[self._next_idx] = reward
        self.dones[self._next_idx] = done

        self._next_idx = (self._next_idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self):
        if self._size < self.sequence_length:
            return None
        
        valid_starts = []
        for i in range(self._size - self.sequence_length + 1):
            end_idx = i + self.sequence_length
            if self._next_idx > i and self._next_idx <= end_idx and self._size == self.capacity:
                continue
            
            if not np.any(self.dones[i : end_idx - 1]):
                 valid_starts.append(i)

        if not valid_starts:
            return None

        if self.oversample_ends:
            end_indices = [i for i in valid_starts if self.dones[i + self.sequence_length - 1]]
            if len(end_indices) > 0:
                # Sample half from ends, half from all valid starts
                num_end_samples = min(len(end_indices), self.batch_size // 2)
                end_sample_indices = np.random.choice(end_indices, num_end_samples, replace=False)
                
                remaining_starts = list(set(valid_starts) - set(end_sample_indices))
                num_regular_samples = self.batch_size - num_end_samples
                
                if len(remaining_starts) > 0:
                    regular_sample_indices = np.random.choice(remaining_starts, num_regular_samples, replace=True) # Can replace if not enough unique starts
                    start_indices = np.concatenate([end_sample_indices, regular_sample_indices])
                else:
                    start_indices = np.random.choice(end_indices, self.batch_size, replace=True)
            else:
                start_indices = np.random.choice(valid_starts, self.batch_size, replace=True)
        else:
            if len(valid_starts) < self.batch_size:
                start_indices = np.random.choice(valid_starts, self.batch_size, replace=True)
            else:
                start_indices = np.random.choice(valid_starts, self.batch_size, replace=False)


        batch_obs = np.zeros((self.batch_size, self.sequence_length, *self.observation_shape), dtype=np.uint8)
        batch_actions = np.zeros((self.batch_size, self.sequence_length, self.action_dim), dtype=np.float32)
        batch_rewards = np.zeros((self.batch_size, self.sequence_length), dtype=np.float32)
        batch_dones = np.zeros((self.batch_size, self.sequence_length), dtype=np.bool_)

        for i, start_idx in enumerate(start_indices):
            indices = (np.arange(start_idx, start_idx + self.sequence_length)) % self.capacity
            batch_obs[i] = self.observations[indices]
            batch_actions[i] = self.actions[indices]
            batch_rewards[i] = self.rewards[indices]
            batch_dones[i] = self.dones[indices]

        return {
            'observations': torch.tensor(batch_obs, dtype=torch.float32, device=self.device) / 255.0 - 0.5,
            'actions': torch.tensor(batch_actions, device=self.device),
            'rewards': torch.tensor(batch_rewards, device=self.device),
            'dones': torch.tensor(batch_dones, dtype=torch.bool, device=self.device)
        }