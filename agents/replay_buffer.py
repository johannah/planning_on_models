import random
import torch
import numpy as np
import pickle
from IPython import embed

class Sample(object):
    def __init__(self, state, action, reward, next_state, end):
        assert type(state) == type(next_state), '%s (true) vs %s (expected)' % (type(state), type(next_state))

        self._state = (state * 255.0).astype(np.uint8)
        self._next_state = (next_state * 255.0).astype(np.uint8)
        self.action = action
        self.reward = reward
        self.end = end

    @property
    def state(self):
        return self._state.astype(np.float32) / 255.0

    @property
    def next_state(self):
        return self._next_state.astype(np.float32) / 255.0

    def __repr__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, End: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.end))
        return info


class ReplayBuffer(object):
    def __init__(self, max_size, random_seed=293):
        self.max_size = max_size
        self.samples = []
        self.oldest_idx = 0
        self.random_state = np.random.RandomState(random_seed)

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return int(to_evict)

    def append(self, state, action, reward, next_state, end):
        assert len(self.samples) <= self.max_size
        new_sample = Sample(state, action, reward, next_state, end)
        if len(self.samples) == self.max_size:
            avail_slot = self._evict()
            self.samples[avail_slot] = new_sample
        else:
            self.samples.append(new_sample)

    def ready(self, min_size):
        return min_size < len(self.samples)

    def sample(self, batch_size):
        """Simpliest uniform sampling (w/o replacement) to produce a batch.
        """
        assert batch_size < len(self.samples), 'not enough samples to sample from'
        return self.random_state.choice(self.samples, batch_size)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0

    def save_buffer(self, filename):
        print("SKIPPPING !!!!!! starting save of buffer of size %s to: %s"%(len(self.samples), filename))
        #try:
        #    fp = open(filename, 'wb')
        #    pickle.dump(self.samples, fp)
        #    fp.close()
        #    print("successfully saved data buffer")

        #except Exception as e:
        #    print("save buffer fail", e)
        #    embed()


def samples_to_tensors(samples, DEVICE):
    num_samples = len(samples)

    states_shape = (num_samples, ) + samples[0].state.shape
    states = np.zeros(states_shape, dtype=np.float32)
    next_states = np.zeros(states_shape, dtype=np.float32)

    rewards = np.zeros(num_samples, dtype=np.float32)
    actions = np.zeros(num_samples, dtype=np.int64)
    non_ends = np.zeros(num_samples, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i] = s.state
        next_states[i] = s.next_state
        rewards[i] = s.reward
        actions[i] = s.action
        non_ends[i] = 0.0 if s.end else 1.0

    states = torch.from_numpy(states).to(DEVICE)
    actions = torch.from_numpy(actions).to(DEVICE)
    rewards = torch.from_numpy(rewards).to(DEVICE)
    next_states = torch.from_numpy(next_states).to(DEVICE)
    non_ends = torch.from_numpy(non_ends).to(DEVICE)

    return states, actions, rewards, next_states, non_ends
