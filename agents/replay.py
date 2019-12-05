import numpy as np
import time
from IPython import embed

def find_component_proportion(data, unique):
    if unique == []:
        unique = list(set(data))
    component_percentages = []
    num = data.shape[0]
    for u in unique:
        num_u = np.where(data == u)[0].shape[0]
        print('val', u, 'num', num_u)
        if num_u:
            component_percentages.append(num_u/float(num))
        else:
            component_percentages.append(0.0)
    return component_percentages

# This function was mostly pulled from
# https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32, num_heads=1,
                 bernoulli_probability=1.0, load_file='', sample_only=False, seed=393, use_pred_frames=False):
                # bernoulli_probability=1.0, latent_frame_height=0, latent_frame_width=0, load_file='', sample_only=False, seed=393):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
            num_heads: integer number of heads needed in mask
            bernoulli_probability: bernoulli probability that an experience will go to a particular head
           # latent_frame_height/width: size of latent representations, if 0, then no latents are stored

        """
        self.sample_only = sample_only
        self.unique_available = False
        if load_file != '':
            self.load_buffer(load_file)
        else:
            self.bernoulli_probability = bernoulli_probability
            assert(self.bernoulli_probability > 0)
            self.size = size
            self.frame_height = frame_height
            self.frame_width = frame_width
            self.agent_history_length = agent_history_length
            self.count = 0
            self.current = 0
            self.num_heads = num_heads
            # Pre-allocate memory
            self.actions = np.empty(self.size, dtype=np.int32)
            self.rewards = np.empty(self.size, dtype=np.float32)
            # store actual frames in true frames
            self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
            self.use_pred_frames = use_pred_frames
            if self.use_pred_frames:
                self.pred_frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
            else:
                self.pred_frames = self.frames
            self.terminal_flags = np.empty(self.size, dtype=np.bool)
            self.masks = np.empty((self.size, self.num_heads), dtype=np.bool)

            if self.num_heads == 1:
                assert(self.bernoulli_probability == 1.0)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.pred_states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.pred_new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)

        self.indices = np.empty(batch_size, dtype=np.int32)
        self.random_state = np.random.RandomState(seed)

    def percentages_rewards(self, unique_rewards=[]):
        return find_component_proportion(self.rewards, unique_rewards)

    def percentages_actions(self, unique_actions=[]):
        return find_component_proportion(self.actions, unique_actions)

    def num_actions(self):
        return len(set(self.actions))

    def num_rewards(self):
        return len(set(self.rewards))

    def num_examples(self):
        return self.count

    def save_buffer(self, filepath):
        st = time.time()
        print("starting save of buffer to %s"%filepath, st)
        np.savez(filepath,
                 frames=self.frames, pred_frames=self.pred_frames,
                 actions=self.actions, rewards=self.rewards,
                 terminal_flags=self.terminal_flags, masks=self.masks,
                 count=self.count, current=self.current, sizze=self.size,
                 agent_history_length=self.agent_history_length,
                 frame_height=self.frame_height, frame_width=self.frame_width,
                 num_heads=self.num_heads, bernoulli_probability=self.bernoulli_probability,
                 )
        print("finished saving buffer", time.time()-st)

    def load_buffer(self, filepath):
        st = time.time()
        print("starting load of buffer from %s"%filepath, st)
        npfile = np.load(filepath)
        self.frames = npfile['frames']
        if 'pred_frames' in npfile.keys():
            self.pred_frames = npfile['pred_frames']
        else:
            self.pred_frames = self.frames

        self.actions = npfile['actions']
        self.rewards = npfile['rewards']
        self.terminal_flags = npfile['terminal_flags']
        self.masks = npfile['masks']
        self.count = npfile['count']
        self.current = npfile['current']
        self.agent_history_length = npfile['agent_history_length']
        self.frame_height = npfile['frame_height']
        self.frame_width = npfile['frame_width']
        self.num_heads = npfile['num_heads']
        self.bernoulli_probability = npfile['bernoulli_probability']
        #self.latent_frames = npfile['latent_frames']

        if self.num_heads == 1:
            assert(self.bernoulli_probability == 1.0)
        try:
            #self.latent_frame_height = npfile['latent_frame_height']
            #self.latent_frame_width = npfile['latent_frame_width']
            self.size = npfile['size']
        except:
            #self.latent_frame_height = self.latent_frames.shape[1]
            #self.latent_frame_width = self.latent_frames.shape[2]
            self.size = self.frames.shape[0]
        print("finished loading buffer", time.time()-st)
        print("loaded buffer current is", self.current)

    def add_experience(self, action, frame, reward, terminal, pred_frame=None):#, latent_frame=''):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        if self.use_pred_frames:
            self.pred_frames[self.current, ...] = pred_frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        mask = self.random_state.binomial(1, self.bernoulli_probability, self.num_heads)
        self.masks[self.current] = mask
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        frames = self.frames[index-self.agent_history_length+1:index+1, ...]
        # if not use_pred_states, this will be a copy of frames
        pframes =  self.pred_frames[index-self.agent_history_length+1:index+1, ...]
        return frames, pframes

    def _get_valid_indices(self, batch_size):
        if batch_size != self.indices.shape[0]:
             self.indices = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            while True:
                index = self.random_state.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # dont add if there was a terminal flag in previous
                # history_length steps
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self, batch_size):
        """
        Returns a minibatch of batch_size
        """
        if batch_size != self.states.shape[0]:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)
            self.pred_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.pred_new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)


        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices(batch_size)

        for i, idx in enumerate(self.indices):
            # This seems correct to me
            # when adding experience - every input frame is the "next frame",
            # the action that got us to this frame, and the reward received
            self.states[i], self.pred_states[i] = self._get_state(idx - 1)
            self.new_states[i], self.pred_new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.masks[self.indices], self.pred_states, self.pred_new_states

    def reset_unique(self):
        self.unique_indexes = np.arange(self.count)
        self.random_state.shuffle(self.unique_indexes)
        self.unique_index = 0
        self.unique_available = True

    def _get_unique_valid_indices(self, batch_size):
        unique_indices = []
        for i in range(batch_size):
            while self.unique_index < len(self.unique_indexes):
                index = self.unique_indexes[self.unique_index]
                self.unique_index+=1
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # dont add if there was a terminal flag in previous
                # history_length steps
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            unique_indices.append(index)

        if self.unique_index >= len(self.unique_indexes)-1:
            self.unique_available = False

        return np.array(unique_indices, np.int32)

    def get_unique_minibatch(self, batch_size):
        """
        Returns a unique minibatch of batch_size -
        self.reset_unique() must be called before utilizing this function or if
        self.unique_available == False

        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        unique_indices = self._get_unique_valid_indices(batch_size)
        batch_size = unique_indices.shape[0]

        if batch_size != self.states.shape[0]:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)
            self.pred_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.pred_new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)


        for i, idx in enumerate(unique_indices):
            # This seems correct to me
            # when adding experience - every input frame is the "next frame",
            # the action that got us to this frame, and the reward received
            self.states[i], self.pred_states[i] = self._get_state(idx - 1)
            self.new_states[i], self.pred_new_states[i] = self._get_state(idx)
        return self.states, self.actions[unique_indices], self.rewards[unique_indices], self.new_states, self.terminal_flags[unique_indices], self.masks[unique_indices], unique_indices

def test_replay_values():
    pass


