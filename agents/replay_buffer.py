import torch
import numpy as np
import pickle
from IPython import embed

def to_simple_storage_state(state):
    return state

def from_simple_storage_state(state):
    return np.array(state)

def to_uint8_storage_state(state):
    return (np.array(state)*255).astype(np.uint8)

def from_uint8_storage_state(state):
    return (np.array(state)/255.).astype(np.float32)

class ReplayBuffer(object):
    def __init__(self, max_buffer_size=1e6, history_size=4, min_sampling_size=1000,
                 num_masks=0, bernoulli_probability=1.0, device='cpu',
                 random_seed=293, to_storage_function=to_uint8_storage_state,
                 from_storage_function=from_uint8_storage_state):
        self.from_storage_function = from_storage_function
        self.to_storage_function = to_storage_function
        self.max_buffer_size = max_buffer_size
        # min_sampling_size used to prevent starting training on too few samples
        # in replay buffer (may cause overfitting)
        assert(min_sampling_size < max_buffer_size), 'invalid configurationn - min sampling size should be smaller than the buffer'
        self.min_sampling_size = min_sampling_size
        self.history_size = history_size
        self.num_masks = num_masks
        self.states = []
        self.rewards = []
        self.actions = []
        self.ongoings = []
        self.episode_cnts = []
        self.state_indexes = []
        # only use masks if needed
        self.device = device
        self.masks = []
        self.episode_num = 0
        self.need_new_init = True
        self.episode_boundaries = []
        self.bernoulli_probability = bernoulli_probability
        self.random_state = np.random.RandomState(random_seed)

    def add_init_state(self, full_state):
        # this initial state should be the entire history_size state
        # indexing is done by the next_state only, so this init_state is never
        # referenced in indexing
        for state in full_state:
            self.states.append(self.to_storage_function(state))
        self.need_new_init = False
        self.episode_num+=1
        self.episode_boundaries.append(len(self.states))

    def add_experience(self, next_state, action, reward, finished):
        assert len(self.states) >= self.history_size, 'did not add initial states before adding experience'
        assert self.need_new_init == False
        # this state can be indexed - must be before state - index is to
        # last needed index for "state"
        self.state_indexes.append(len(self.states))
        self.states.append(self.to_storage_function(next_state))
        self.rewards.append(reward)
        self.actions.append(action)
        self.ongoings.append(int(not finished))
        if finished:
            self.need_new_init = True
        if self.num_masks > 0:
            self.masks.append(self.random_state.binomial(1, self.bernoulli_probability, self.num_masks).astype(np.uint8))
        self.evict()

    def ready(self, batch_size):
        compare = max(batch_size, self.min_sampling_size)
        return compare < len(self.state_indexes)

    def evict(self):
        """
         we should evict frames when the replay buffer is too big, however,
         we have to do this carefully to make sure that we dont hit edge cases
        """
        if len(self.state_indexes)>self.max_buffer_size:
            # history_size+1 states are required for each index in the buffer
            oldest_index = self.state_indexes.pop(0)
            # remove the oldest state required for this index only
            # this will be index-self.history_size
            # if oldest_index == 4, then states will pop the true frame 0
            self.states.pop(0)
            self.rewards.pop(0)
            self.actions.pop(0)
            self.ongoings.pop(0)
            if self.num_masks > 0:
                self.masks.pop(0)
            self.state_indexes = [x-1 for x in self.state_indexes]

    def sample(self, batch_indexes, pytorchify):
        """ the index is counted at last needed index of  "state"
         if states == [0,1,2,3,4,5,6]
         then avilable indexes are [3,4,5]
         if index=5, state will be grabbed from states[2,3,4,5]
         next_state will be grabbed from states[3,4,5,6], assuming history_size=4
        """

        assert(min(self.state_indexes) >= self.history_size)
        assert(max(self.state_indexes) < len(self.states))
        state_indexes = [self.state_indexes[i] for i in batch_indexes]
        _all_states = self.from_storage_function([self.states[i-(self.history_size):i+1] for i in state_indexes])
        _states = _all_states[:,:self.history_size]
        _next_states = _all_states[:,1:]
        _rewards = np.array([self.rewards[i] for i in batch_indexes])
        _actions = np.array([self.actions[i] for i in batch_indexes])
        _ongoings = np.array([self.ongoings[i] for i in batch_indexes])
        if pytorchify:
            _states = torch.FloatTensor(_states).to(self.device)
            _next_states = torch.FloatTensor(_next_states).to(self.device)
            _rewards = torch.FloatTensor(_rewards).to(self.device)
            _actions = torch.LongTensor(_actions).to(self.device)
            # which type should finished be ?
            _ongoings = torch.FloatTensor(_ongoings).to(self.device)
        return_vals = [_states, _actions, _rewards, _next_states, _ongoings]

        if self.num_masks > 0:
            _masks = np.array([self.masks[i] for i in batch_indexes])
            if pytorchify:
                _masks = torch.FloatTensor(_masks).to(self.device)
            return_vals.append(_masks)
        return_vals.append(batch_indexes)
        return return_vals

    def sample_random(self, batch_size, pytorchify=True):
        assert self.ready(batch_size), 'not enough samples in replay buffer'
        batch_indexes = self.random_state.randint(0, len(self.rewards), batch_size)
        return self.sample(batch_indexes, pytorchify)

    def sample_ordered(self, start_index, batch_size=32, pytorchify=True):
        assert self.ready(batch_size), 'not enough samples in replay buffer'
        assert(start_index < len(self.rewards)-1), 'start index too high'
        batch_indexes = np.arange(start_index, min(start_index+batch_size, len(self.rewards)), dtype=np.int)
        return self.sample(batch_indexes, pytorchify)

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()



def test_buffer():
    replay_buffer = ReplayBuffer(max_buffer_size=10, history_size=4, min_sampling_size=2,
                                  num_masks=0, bernoulli_probability=1.0, device='cpu', random_seed=293,
                                 to_storage_function=to_simple_storage_state,
                                 from_storage_function=from_simple_storage_state)

    print('indexes', replay_buffer.state_indexes)
    print('states',replay_buffer.states)
    print('avail states', [replay_buffer.states[i] for i in replay_buffer.state_indexes])
    # episode 1
    replay_buffer.add_init_state([0,1,2,3])
    replay_buffer.add_experience(next_state=4, action=1, reward=1, finished=False)
    replay_buffer.add_experience(next_state=5, action=1, reward=0, finished=False)
    replay_buffer.add_experience(next_state=6, action=1, reward=1, finished=False)
    replay_buffer.add_experience(next_state=7, action=1, reward=1, finished=False)
    replay_buffer.add_experience(next_state=8, action=1, reward=1, finished=True)
    # episode 2
    print('indexes', replay_buffer.state_indexes)
    print('states',replay_buffer.states)
    print('avail states', [replay_buffer.states[i] for i in replay_buffer.state_indexes])
    print('-----------')
    replay_buffer.add_init_state([10,11,12,13])
    replay_buffer.add_experience(next_state=14, action=2, reward=1, finished=False)
    replay_buffer.add_experience(next_state=15, action=2, reward=1, finished=False)
    replay_buffer.add_experience(next_state=16, action=2, reward=0, finished=False)
    replay_buffer.add_experience(next_state=17, action=2, reward=1, finished=False)
    replay_buffer.add_experience(next_state=18, action=2, reward=1, finished=False)
    print('indexes', replay_buffer.state_indexes)
    print('states',replay_buffer.states)
    print('avail states', [replay_buffer.states[i] for i in replay_buffer.state_indexes])
    print('-----------')
    s,a,r,ns,f,bi = replay_buffer.sample_ordered(0,3)
    # episode 3
    replay_buffer.add_init_state([20,21,22,23])
    replay_buffer.add_experience(next_state=24, action=3, reward=0, finished=True)
    print('indexes', replay_buffer.state_indexes)
    print('states',replay_buffer.states)
    print('avail states', [replay_buffer.states[i] for i in replay_buffer.state_indexes])
    print('-----------')
    # episode 4
    replay_buffer.add_init_state([30,31,32,33])
    replay_buffer.add_experience(next_state=34, action=4, reward=0, finished=False)
    replay_buffer.add_experience(next_state=35, action=4, reward=0, finished=False)
    replay_buffer.add_experience(next_state=37, action=4, reward=0, finished=False)
    replay_buffer.add_experience(next_state=38, action=4, reward=0, finished=True)
    print('indexes', replay_buffer.state_indexes)
    print('states',replay_buffer.states)
    print('avail states', [replay_buffer.states[i] for i in replay_buffer.state_indexes])
    print('-----------')
    s2,a2,r2,ns2,f2,bi2 = replay_buffer.sample_ordered(8,3,False)
    assert ns2[-1][-1] == 38
    assert ns2[-1][0] == 34
    assert s2[-1][-1] == 37
    assert s2[-1][0] == 33


if __name__ == '__main__':
    test_buffer()


