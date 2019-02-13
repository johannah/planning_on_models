import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import pickle
import os
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
        self.device = device
        # min_sampling_size used to prevent starting training on too few samples
        # in replay buffer (may cause overfitting)
        assert(min_sampling_size < max_buffer_size), 'invalid configurationn - min sampling size should be smaller than the buffer'
        self.min_sampling_size = min_sampling_size
        self.history_size = history_size
        self.pop_offset = 0
        self.num_masks = num_masks
        self.episodes = []
        self.rewards = []
        self.actions = []
        self.ongoings = []
        self.episodes = []
        self.episode_indexes = []
        self.episode_steps = []
        # only use masks if needed
        self.masks = []
        self.episode_num = -1
        self.evicting_episode = -1
        self.episode_step_cnt = 0
        self.need_new_init = True
        self.episode_lengths = []
        self.bernoulli_probability = bernoulli_probability
        self.random_state = np.random.RandomState(random_seed)
        self.mask_state = np.random.RandomState(random_seed+10)

    def add_init_state(self, full_state):
        # this initial state should be the entire history_size state
        # indexing is done by the next_state only, so this init_state is never
        # referenced in indexing
        self.episode_num+=1
        #print("INIT episode", self.episode_num)
        self.episode_step_cnt = 0
        self.episodes.append([])
        for state in full_state:
            self.episodes[self.episode_num].append(self.to_storage_function(state))
            self.episode_step_cnt+=1
        self.need_new_init = False

    def add_experience(self, next_state, action, reward, finished):
        assert self.need_new_init == False
        # this state can be indexed - must be before state - index is to
        # last needed index for "state"
        self.episode_steps.append(self.episode_step_cnt)
        self.episode_step_cnt+=1
        self.episodes[self.episode_num].append(self.to_storage_function(next_state))
        self.episode_indexes.append(self.episode_num)
        self.rewards.append(reward)
        self.actions.append(action)
        self.ongoings.append(int(not finished))
        if finished:
            self.episode_lengths.append(self.episode_step_cnt)
            self.need_new_init = True
            print("finished episode", self.episode_num)
        if self.num_masks > 1:
            self.masks.append(self.mask_state.binomial(1, self.bernoulli_probability, self.num_masks).astype(np.uint8))
        self.evict()

    def ready(self, batch_size):
        # is our replay buffer bigger than the min sampling size?
        # is it also bigger than the batch size requested
        compare = max(batch_size, self.min_sampling_size)
        return compare < len(self.rewards)

    def evict(self):
        """
         we should evict frames when the replay buffer is too big, however,
         we have to do this carefully to make sure that we dont hit edge cases
        """
        if len(self.rewards)>self.max_buffer_size:
            self.rewards.pop(0)
            self.actions.pop(0)
            self.ongoings.pop(0)
            if self.num_masks > 1:
                self.masks.pop(0)
            evicting_episode = self.episode_indexes.pop(0)
            episode_step = self.episode_steps.pop(0)
            if evicting_episode != self.evicting_episode:
                # started evicting a new episode - keep track
                #print("NEW EVICT EP", evicting_episode)
                self.evicting_episode = evicting_episode
                self.episode_pop_count = -1
            else:
                self.episode_pop_count-=1
            assert(len(self.episode_steps) == len(self.episode_indexes))
            self.episodes[evicting_episode].pop(0)
            if len(self.episodes[evicting_episode])==self.history_size:
                # finished
                #print('end of evicting episode', evicting_episode)
                self.episodes[evicting_episode] = 0

    def sample(self, batch_indexes, pytorchify):
        eps = [self.episode_indexes[i] for i in batch_indexes]
        orig_epi = epi = [self.episode_steps[i] for i in batch_indexes]
        for idx, e in enumerate(eps):
            if e == self.evicting_episode:
                epi[idx] += self.episode_pop_count
        _list_all_states = [self.from_storage_function(self.episodes[ep][i-self.history_size:i+1]) for (ep,i)  in zip(eps, epi)]
        _all_states = np.array(_list_all_states)
        try:
            assert(len(_all_states.shape) == 4)
            _states = _all_states[:,:self.history_size]
        except Exception as e:
            print("bad sample", e)
            embed()
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

        if self.num_masks > 1:
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
        st = time.time()
        print("Buffer loading from %s. This may take some time" %filename)
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        print("buffer finished loading from %s" %(filename))

    def save(self, filename):
        print("replay buffer saving: %s" %filename)
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        print("FINISHED replay buffer saving")

    def plot_buffer(self, buffer_filepath, plot_end_minus=1000, make_gif=True, make_gif_min=-2000, overwrite_imgs=False):
        img_filepath = buffer_filepath.replace('.pkl', '_plot')
        if not os.path.exists(img_filepath):
            print("making img directory: %s" %img_filepath)
            os.makedirs(img_filepath)

        if plot_end_minus > 0:
            start_index = len(self.rewards)-plot_end_minus
            start_index = max(0,start_index)
        else:
            start_index = 0
        batch_size = len(self.rewards)-start_index
        if self.num_masks > 1:
            states, actions, rewards, next_states, ongoings, _, batch_index = self.sample_ordered(start_index, batch_size=batch_size, pytorchify=False)
        else:
            states, actions, rewards, next_states, ongoings, batch_index = self.sample_ordered(start_index, batch_size=batch_size, pytorchify=False)

        episode_reward = 0
        for ss in range(len(ongoings)):
            step = self.episode_steps[batch_index[ss]]
            episode = self.episode_indexes[batch_index[ss]]
            ipath = os.path.join(img_filepath, "E%05d_%05d.png" %(episode, step))
            episode_reward += rewards[ss]
            if not os.path.exists(ipath) or overwrite_imgs:
                title = 'EP:%05d ST%05d A%s R:%03d' %(episode, step, actions[ss], episode_reward)
                if not bool(ongoings[ss]):
                    title+= " FINISHED"
                plt.figure()
                plt.title(title)
                plt.imshow(states[ss][-1])
                plt.savefig(ipath)
                plt.close()
            if not bool(ongoings[ss]):
                print(" episode: %s rewards: %s "%(episode, episode_reward))
                if make_gif and episode_reward > make_gif_min:
                    print("making gif")
                    spath = os.path.join(img_filepath, 'E%05d*.png' %episode)
                    gpath = os.path.join(img_filepath, '_E%05d_R%04d.gif' %(episode, episode_reward))
                    os.system("convert %s %s " %(spath,gpath))
                episode_reward = 0

class SimpleReplayBuffer(object):
    def __init__(self, max_buffer_size=1e6, history_size=4, min_sampling_size=1000,
                 num_masks=0, bernoulli_probability=1.0, device='cpu',
                 random_seed=293, to_storage_function=to_uint8_storage_state,
                 from_storage_function=from_uint8_storage_state):
        self.from_storage_function = from_storage_function
        self.to_storage_function = to_storage_function
        self.max_buffer_size = max_buffer_size
        self.device = device
        # min_sampling_size used to prevent starting training on too few samples
        # in replay buffer (may cause overfitting)
        assert(min_sampling_size < max_buffer_size), 'invalid configurationn - min sampling size should be smaller than the buffer'
        self.min_sampling_size = min_sampling_size
        self.history_size = history_size
        self.num_masks = num_masks
        self.states = []
        self.episode_num = -1
        self.evicting_episode = -1
        self.bernoulli_probability = bernoulli_probability
        self.random_state = np.random.RandomState(random_seed)
        self.mask_state = np.random.RandomState(random_seed+10)

    def add_experience(self, state, action, reward, next_state, finished):
        # this state can be indexed - must be before state - index is to
        # last needed index for "state"
        if self.num_masks > 1:
            mask = self.mask_state.binomial(1, self.bernoulli_probability, self.num_masks).astype(np.uint8)
            self.states.append((state,action,reward,next_state,not finished,mask))
        else:
            self.states.append((state,action,reward,next_state,not finished))
        if finished:
            self.episode_num+=1
            print("finished episode", self.episode_num)
        self.evict()

    def ready(self, batch_size):
        compare = max(batch_size, self.min_sampling_size)
        return compare < len(self.states)

    def evict(self):
        """
         we should evict frames when the replay buffer is too big, however,
         we have to do this carefully to make sure that we dont hit edge cases
        """
        if len(self.states)>self.max_buffer_size:
            self.states.pop(0)

    def sample(self, batch_indexes, pytorchify):
        """ TODO check this
         the index is counted at last needed index of  "state"
         if states == [0,1,2,3,4,5,6]
         then avilable indexes are [3,4,5]
         if index=5, state will be grabbed from states[2,3,4,5]
         next_state will be grabbed from states[3,4,5,6], assuming history_size=4
        """
        batch = [self.states[i] for i in batch_indexes]
        _states = np.array([item[0] for item in batch])
        _actions = np.array([item[1] for item in batch])
        _rewards = np.array([item[2] for item in batch])
        _next_states = np.array([item[3] for item in batch])
        _ongoings = np.array([item[4] for item in batch])

        if pytorchify:
            _states = torch.FloatTensor(_states).to(self.device)
            _next_states = torch.FloatTensor(_next_states).to(self.device)
            _rewards = torch.FloatTensor(_rewards).to(self.device)
            _actions = torch.LongTensor(_actions).to(self.device)
            # which type should finished be ?
            _ongoings = torch.FloatTensor(_ongoings).to(self.device)
        return_vals = [_states, _actions, _rewards, _next_states, _ongoings]

        if self.num_masks > 1:
            _masks = [item[5] for item in batch]
            if pytorchify:
                _masks = torch.FloatTensor(_masks).to(self.device)
            return_vals.append(_masks)
        return_vals.append(batch_indexes)
        return return_vals

    def sample_random(self, batch_size, pytorchify=True):
        assert self.ready(batch_size), 'not enough samples in replay buffer'
        batch_indexes = self.random_state.randint(0, len(self.states), batch_size)
        return self.sample(batch_indexes, pytorchify)

    def sample_ordered(self, start_index, batch_size=32, pytorchify=True):
        #assert self.ready(batch_size), 'not enough samples in replay buffer'
        #assert(start_index < len(self.rewards)-1), 'start index too high'
        batch_indexes = np.arange(start_index, min(start_index+batch_size, len(self.states)), dtype=np.int)
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

def test_against_simple():
    from env import Environment

    BUFFER_SIZE = 20
    HISTORY_SIZE = 4
    MIN_HISTORY_TO_LEARN = 10
    BERNOULLI_PROBABILITY = 1.0
    DEVICE = 'cpu'
    N_ENSEMBLE = 1
    RANDOM_SEED = 39
    BATCH_SIZE = 12

    replay_buffer = ReplayBuffer(max_buffer_size=BUFFER_SIZE,
                           history_size=HISTORY_SIZE,
                           min_sampling_size=MIN_HISTORY_TO_LEARN,
                           num_masks=N_ENSEMBLE,
                           bernoulli_probability=BERNOULLI_PROBABILITY,
                           device=DEVICE, random_seed=RANDOM_SEED)

    simple_replay_buffer = SimpleReplayBuffer(max_buffer_size=BUFFER_SIZE,
                           history_size=HISTORY_SIZE,
                           min_sampling_size=MIN_HISTORY_TO_LEARN,
                           num_masks=N_ENSEMBLE,
                           bernoulli_probability=BERNOULLI_PROBABILITY,
                           device=DEVICE, random_seed=RANDOM_SEED)

    env = Environment('roms/breakout.bin')
    rs = np.random.RandomState(2949)
    finished = True
    for i in range(100000):
        if finished:
            state = env.reset()
            replay_buffer.add_init_state(state)
        action = rs.randint(0,4)
        next_state, reward, finished = env.step(action)
        replay_buffer.add_experience(next_state[-1], action, reward, finished)
        simple_replay_buffer.add_experience(state, action, reward, next_state, finished)
        state = next_state
        if replay_buffer.ready(BATCH_SIZE):
            rb_states, rb_actions, rb_rewards, rb_next_states, rb_ongoings, rb_batch_indexes = replay_buffer.sample_random(BATCH_SIZE, pytorchify=False)
            srb_states, srb_actions, srb_rewards, srb_next_states, srb_ongoings, srb_batch_indexes = simple_replay_buffer.sample_random(BATCH_SIZE, pytorchify=False)
            assert rb_states.sum() == srb_states.sum()
            assert rb_next_states.sum() == srb_next_states.sum()
            assert rb_actions.sum() == srb_actions.sum()
            assert rb_rewards.sum() == srb_rewards.sum()
            assert rb_ongoings.sum() == srb_ongoings.sum()
            assert rb_batch_indexes.sum() == srb_batch_indexes.sum()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-t", '--test', action='store_true', default=False)
    parser.add_argument("-b", '--buffer_loadpath', default='')
    parser.add_argument("-p", '--plot', action='store_true', default=True)
    parser.add_argument("-o", '--overwrite_imgs', action='store_true', default=False, help='overwrite images if they already exist')
    parser.add_argument("-mg", '--make_gif', action='store_true', default=True, help='make a gif of each episode')
    parser.add_argument("-em", '--plot_end_minus', type=int, default=1000, help='limit plotting to the last x steps')
    parser.add_argument("-gm", '--make_gif_min', type=int, default=-2000, help='only make a gif if the episode reward is greater than this number')
    args = parser.parse_args()
    if args.test:
        test_against_simple()
    else:
        if args.buffer_loadpath != '':
            rbuffer = ReplayBuffer()
            rbuffer.load(args.buffer_loadpath)
            rbuffer.plot_buffer(args.buffer_loadpath, plot_end_minus=args.plot_end_minus, make_gif=args.make_gif, make_gif_min=args.make_gif_min, overwrite_imgs=args.overwrite_imgs)

