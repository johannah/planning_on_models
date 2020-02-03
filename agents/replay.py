import numpy as np
import time
from copy import deepcopy
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
                 bernoulli_probability=1.0, load_file='', seed=393, use_pred_frames=False,
                 maxpool=False, trim_before=0, trim_after=0, kernel_size=0, reduction_function=np.max):
                # bernoulli_probability=1.0, latent_frame_height=0, latent_frame_width=0, load_file='', seed=393):
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
        if maxpool:
            self.maxpool = True
        else:
            self.maxpool = False
        self.trim_before = trim_before
        self.trim_after = trim_after
        self.kernel_size = kernel_size
        self.reduction_function = reduction_function
        self.unique_available = False
        self.use_pred_frames = use_pred_frames
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
            self.rewards = np.empty(self.size, dtype=np.int32)
            # store actual frames in true frames
            self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
            if self.use_pred_frames:
                self.pred_frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
            else:
                self.pred_frames = self.frames
            self.terminal_flags = np.zeros(self.size, dtype=np.bool)
            self.end_flags = np.zeros(self.size, dtype=np.bool)
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
                 terminal_flags=self.terminal_flags,
                 end_flags=self.end_flags,
                 masks=self.masks,
                 count=self.count, current=self.current, size=self.size,
                 agent_history_length=self.agent_history_length,
                 frame_height=self.frame_height, frame_width=self.frame_width,
                 num_heads=self.num_heads, bernoulli_probability=self.bernoulli_probability,
                 maxpool=self.maxpool,
                 trim_before=self.trim_before,
                 trim_after=self.trim_after,
                 kernel_size=self.kernel_size,
                 reduction_function=str(self.reduction_function),
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

        try:
            self.size = npfile['size']
        except:
            self.size = npfile['sizze']

        self.actions = npfile['actions']
        self.rewards = npfile['rewards']
        self.terminal_flags = npfile['terminal_flags']
        self.end_flags = npfile['end_flags']
        self.masks = npfile['masks']
        self.count = npfile['count']
        self.current = npfile['current']
        self.agent_history_length = npfile['agent_history_length']
        self.frame_height = npfile['frame_height']
        self.frame_width = npfile['frame_width']
        self.num_heads = npfile['num_heads']
        self.bernoulli_probability = npfile['bernoulli_probability']
        if 'maxpool' in npfile.keys():
            self.maxpool = npfile['maxpool']
            self.trim_before = npfile['trim_before']
            self.trim_after = npfile['trim_after']
            self.kernel_size = npfile['kernel_size']
            #self.reduction_function = eval(npfile['reduction_function'])
            print("not loading reduction function")
            self.reduction_function = np.max
        else:
            self.maxpool = False
        self.unique_available = False
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

    # TODO add current q value
    def add_experience(self, action, frame, reward, terminal, end, pred_frame=None):#, latent_frame=''):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84) frame of an Atari game in grayscale - is the "next state" when moving through a game
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            if self.maxpool:
                frame = self.online_shrink_frame_size(frame)
            else:
                print('maxpool issue')
                embed()
                raise ValueError('Dimension of frame is wrong!', frame.shape)
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        if self.use_pred_frames:
            self.pred_frames[self.current, ...] = pred_frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.end_flags[self.current] = end
        mask = self.random_state.binomial(1, self.bernoulli_probability, self.num_heads)
        self.masks[self.current] = mask
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        return frame

    def _get_state(self, index):
        try:
            if self.count < self.agent_history_length - 1:
                raise ValueError("The replay memory is empty!")
            if index <= self.agent_history_length and self.count >= self.size:
                print("WE HIT THE BOUNDARY CONDITION")
                #if index < self.agent_history_length - 1:
                #    raise ValueError("Index must be min 3")
                index_frames = []
                for j in range(self.agent_history_length):
                    ii = index - self.agent_history_length + 1 + j
                    index_frames.append(ii)
                frames = self.frames[index_frames, ...]
                pframes = self.pred_frames[index_frames, ...]
            else:
                frames = self.frames[index-self.agent_history_length+1:index+1, ...]
                # if not use_pred_states, this will be a copy of frames
                pframes =  self.pred_frames[index-self.agent_history_length+1:index+1, ...]
        except Exception as e:
            print('get_indices', e)
            embed()
        if frames.shape[0] == 0:
            print('frames wrong size')
            embed()
        return frames, pframes

    def is_valid_index(self, index, last_state_allowed=False):
        if self.count < self.agent_history_length-1:
            print("INVALID INDEX: count:%s less than %s" %(self.count, self.agent_history_length-1))
            return False
        if index > self.count:
            return False
        #if index < self.agent_history_length-1:
        #    return False
        # Jan 2020 - not sure why this flag was in there - doesnt allow me to
        # get last state though - which is needed
        if not last_state_allowed:
            if index >= self.current and index - self.agent_history_length <= self.current:
                print('INVALID index:%s >= current:%s'%(index, self.current))
                print('and index buff:%s <= current %s'%(index - self.agent_history_length,
                                                         self.current))
                return False
        if self.end_flags[index-self.agent_history_length+1:index].any():
            #print('INVALID index:%s - end in state'%index)
            return False
        return True

    def _get_valid_indices(self, batch_size):
        if batch_size != self.indices.shape[0]:
             self.indices = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            valid = False
            while not valid:
                index = self.random_state.randint(self.agent_history_length, min([self.count, self.size]) - 1)
                valid = self.is_valid_index(index)
            self.indices[i] = index

    def get_last_state(self):
        # not handling rollovers of replay buffer correctly
        last_count = self.current - 1
        if self.is_valid_index(last_count, last_state_allowed=True):
            next_states, _ = self._get_state(last_count)
        else:
            print('----------------------------------unable to get valid index', last_count)
            embed()
            next_states = self.states[0]*0
        if next_states.shape[0] != self.agent_history_length:
            print("BAD GET_LAST STATE")
            embed()
        return next_states

    def get_last_n_states(self, num_steps):
        # terminal_flags[self.current-1] will be True if it was the end of the
        # episode
        if num_steps > self.size:
            print('limiting last n states to %s from %s due to size of replay buffer'%(num_steps,  self.size - self.agent_history_length))
            num_steps = self.size - self.agent_history_length
        index = self.current-1
        get_indexes = []
        for i in range(num_steps):
            if self.is_valid_index(index, last_state_allowed=True):
                get_indexes.append(index)
            index-=1
        # change order
        get_indexes = get_indexes[::-1]
        #TODO - finish getting arrays
        states = np.empty((len(get_indexes), self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        next_states = np.empty((len(get_indexes), self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)

        for i, idx in enumerate(get_indexes):
            # This seems correct to me
            # when adding experience - every input frame is the "next frame",
            # the action that got us to this frame, and the reward received
            states[i], _ = self._get_state(idx-1)
            next_states[i], _ = self._get_state(idx)
        return states, self.actions[get_indexes], self.rewards[get_indexes], next_states, self.terminal_flags[get_indexes], self.masks[get_indexes], get_indexes

    def get_last_episode(self):
        # NOTE THIS WORKS ON terminal_flags - not end_episode
        # terminal_flags[self.current-1] will be True if it was the end of the
        # episode
        get_indexes = [self.current-1]
        # self.current-2 will be False unless it was a very tiny episode
        last_index = self.current-2
        while self.is_valid_index(last_index):
            # step back in time, quit when index reaches previous episode - no
            # idea how this will work across buffer boundaries -
            get_indexes.append(last_index)
            last_index-=1
            if len(get_indexes) >= self.size - self.agent_history_length:
                break
        # change order
        get_indexes = get_indexes[::-1]
        #TODO - finish getting arrays
        states = np.empty((len(get_indexes), self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        next_states = np.empty((len(get_indexes), self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)

        for i, idx in enumerate(get_indexes):
            # This seems correct to me
            # when adding experience - every input frame is the "next frame",
            # the action that got us to this frame, and the reward received
            states[i], _ = self._get_state(idx-1)
            next_states[i], _ = self._get_state(idx)
        return states, self.actions[get_indexes], self.rewards[get_indexes], next_states, self.terminal_flags[get_indexes], self.masks[get_indexes], get_indexes

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
            #print('trying', idx)
            # This seems correct to me
            # when adding experience - every input frame is the "next frame",
            # the action that got us to this frame, and the reward received
            self.states[i], self.pred_states[i] = self._get_state(idx - 1)
            self.new_states[i], self.pred_new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.masks[self.indices]
        #return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.masks[self.indices], self.pred_states, self.pred_new_states

    def init_unique(self):
        """ must call this to before using reset_unique - this will change when adding to buffer """
        self.unique_indexes = self._find_all_valid_indices()
        print('found %s unique_indexes in buffer' %len(self.unique_indexes))

    def reset_unique(self):
        if 'unique_indexes' not in self.__dict__.keys():
            self.init_unique()
        self.random_state.shuffle(self.unique_indexes)
        self.unique_index = 0
        self.unique_available = True

    def _find_all_valid_indices(self):
        """ find all indices which can be used for data """
        unique_indexes = []
        for index in range(self.count):
            if index < self.agent_history_length:
                continue
            if index >= self.current and index - self.agent_history_length <= self.current:
                continue
            # dont add if there was a terminal flag in previous
            # history_length steps
            if self.end_flags[index - self.agent_history_length:index].any():
                continue
            unique_indexes.append(index)
        return np.array(unique_indexes, np.int32)

    def _get_unique_valid_indices(self, batch_size):
        unique_indices = []
        index_indices = []
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
            index_indices.append(self.unique_index-1)
            unique_indices.append(index)

        if self.unique_index >= len(self.unique_indexes)-1:
            self.unique_available = False

        return np.array(unique_indices, np.int32), np.array(index_indices, np.int32)


    def online_shrink_frame_size(self, frame):
        frame = frame[None]
        if self.trim_before > 0:
            frame = trim_array(frame, self.trim_before)
        frame = pool_2d(frame, self.kernel_size, self.reduction_function)
        if self.trim_after > 0:
            frame = trim_array(frame, self.trim_after)
        return frame[0]

    def shrink_frame_size(self, kernel_size=(2,2), reduction_function=np.max, trim_before=0, trim_after=0,  batch_size=32):
        _, oh, ow = self.frames.shape
        if trim_before > 0:
            self.frames = trim_array(self.frames, trim_before)
            self.pred_frames = trim_array(self.pred_frames, trim_before)

        self.frames = pool_2d(self.frames, kernel_size, reduction_function)
        self.pred_frames = pool_2d(self.pred_frames, kernel_size, reduction_function)
        if trim_after > 0:
            self.frames = trim_array(self.frames, trim_after)
            self.pred_frames = trim_array(self.pred_frames, trim_after)
        _,self.frame_height,self.frame_width = self.frames.shape
        print('resized frames from %sx%s to %sx%s'%(oh,ow,self.frame_height,self.frame_width))
        self.states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.pred_states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.pred_new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)



    def get_unique_minibatch(self, batch_size):
        """
        Returns a unique minibatch of batch_size -
        self.reset_unique() must be called before utilizing this function or if
        self.unique_available == False

        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # index indices is the value that should be used for acn
        unique_indices, index_indices = self._get_unique_valid_indices(batch_size)
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
        return self.states, self.actions[unique_indices], self.rewards[unique_indices], self.new_states, self.terminal_flags[unique_indices], self.masks[unique_indices], unique_indices, index_indices

def trim_array(array, trim):
   return array[:,trim:-trim, trim:-trim]

def pool_2d(arr, kernel_size, reduction_function, debug_plot=False, debug_name=""):
    # takes in arr of bs, h, w
    # returns bs, h, w where h and w are reduced
    # reduction function MUST SUPPORT AXIS ARGUMENT
    assert arr.shape[1] % kernel_size[0] == 0
    assert arr.shape[2] % kernel_size[1] == 0

    if debug_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.matshow(arr[0])
        plt.axis("off")
        plt.savefig("plt_{}_0.png".format(debug_name))
        plt.close()

    arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2] // kernel_size[1], kernel_size[1])
    arr = np.transpose(arr, (0, 2, 3, 1))
    arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3] // kernel_size[0], kernel_size[0])
    arr = np.transpose(arr, (0, 3, 1, 4, 2))
    # now bs, h, w, sp1_h, sp2_w

    if debug_plot:
        f, axarr = plt.subplots(arr.shape[1], arr.shape[2])
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                axarr[i, j].matshow(arr[0, i, j, :, :])
                axarr[i, j].axis("off")
        plt.savefig("plt_{}_1.png".format(debug_name))
        plt.close()

    arr = reduction_function(arr, axis=-1)
    arr = reduction_function(arr, axis=-1)
    if debug_plot:
        plt.matshow(arr[0])
        plt.axis("off")
        plt.savefig("plt_{}_2.png".format(debug_name))
        print("Plotted debug plots")
    return arr


