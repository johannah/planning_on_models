import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from IPython import embed
from imageio import imwrite

def make_random_subset_buffers(dataset_path, buffer_path, train_max_examples=100000, kernel_size=(2,2), trim_before=0, trim_after=0):
    sys.path.append('../agents')
    from replay import ReplayMemory
    # keep max_examples < 100000 to enable knn search
    # states [top of image:bottom of image,:]
    # in breakout - can safely reduce size to be 40x40 of the given image
    # try to get an even number of each type of reward

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    buffer_name = os.path.split(buffer_path)[1]
    buffers = {}
    paths = {}
    for phase in ['valid', 'train']:
        if phase == 'valid':
            max_examples = int(0.15*train_max_examples)
        else:
            max_examples = train_max_examples
        small_name = buffer_name.replace('.npz', '_random_subset_%06d_%sx%stb%sta%s_%s.npz' %(max_examples, kernel_size[0], kernel_size[1], trim_before, trim_after, phase))
        small_path = os.path.join(dataset_path, small_name)
        paths[phase] = small_path
        if os.path.exists(small_path):
            print('loading small buffer path')
            print(small_path)
            sbuffer = ReplayMemory(load_file=small_path)
            sbuffer.init_unique()
            buffers[phase] = sbuffer

    # if we dont have both train and valid - make completely new train/valid set
    if not len(buffers.keys()) == 2:
        print('creating new train/valid buffers')
        load_buffer = ReplayMemory(load_file=buffer_path)
        orig_states = []
        small_states = []
        for index in range(10,400):
            if load_buffer.is_valid_index(index):
                s,_ = load_buffer._get_state(index)
                orig_states.append(s[-1])
                small_states.append(load_buffer.online_shrink_frame_size(s[-1], trim_before, kernel_size, trim_after))
        bdir = small_path.replace('.npz', '')
        if not os.path.exists(bdir):
            os.makedirs(bdir)
        image_path = os.path.join(bdir, 'step_%03d.png')
        movie_path = os.path.join(bdir, 'movie.mp4')
        for index in range(len(orig_states)):
            f,ax = plt.subplots(1,2)
            ax[0].matshow(orig_states[index])
            ax[1].matshow(small_states[index])
            plt.savefig(image_path%index)
            plt.close()
        cmd = "ffmpeg -n -r 10 -i %s -c:v libx264 -pix_fmt yuv420p %s"%(os.path.abspath(image_path),os.path.abspath(movie_path))
        print(cmd)
        os.system(cmd)

        if max(list(kernel_size)+[trim_before, trim_after]) > 1:
            load_buffer.shrink_frame_size(kernel_size=kernel_size,
                                          reduction_function=np.max,
                                          trim_before=trim_before, trim_after=trim_after)


        #for r in range(states.shape[0]):
        #    imwrite('mp%s.png'%r, states[r,-1])

        load_buffer.reset_unique()
        # history_length + 1 for every random example
        frame_multiplier = (load_buffer.agent_history_length+1)
        #frame_multiplier = 2
        total_frames_needed = int((max_examples*1.15)*frame_multiplier)+1
        # not sure why we weren't allowing overlapping frames
        #total_frames_needed = int((max_examples*1.15))
        if load_buffer.count < total_frames_needed%load_buffer.size:
           raise ValueError('load buffer is not large enough (%s) to collect number of examples (%s)'%(load_buffer.count, total_frames_needed))
        print('loading prescribed buffer path.... this may take a while')
        print(buffer_path)
        for phase in ['valid', 'train']:
            if phase == 'valid':
                max_examples = int(0.15*train_max_examples)
            else:
                max_examples = train_max_examples
            print('creating small %s buffer with %s examples'%(phase, max_examples))
            # actions for breakout:
            # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            frames_needed = max_examples*frame_multiplier
            sbuffer = ReplayMemory(frames_needed,
                                   frame_height=load_buffer.frame_height, frame_width=load_buffer.frame_width,
                                   agent_history_length=load_buffer.agent_history_length)

            num_examples = 0
            while num_examples < max_examples:
                batch = load_buffer.get_unique_minibatch(1)
                states, actions, rewards, next_states, real_terminal_flags, _, unique_indices, index_indices = batch
                bs,num_hist,h,w = states.shape
                # action is the action that was used to get from state to next state
                #    t-3, t-2, t-1, t-1, t
                #  s-4, s-3, s-2, s-1
                #     s-3, s-2, s-1, s

                past_indices = np.arange(unique_indices-(num_hist), unique_indices+1)
                for batch_idx in range(bs):
                    # get t-4 thru t=0
                    # size is bs,5,h,w
                    all_states = np.hstack((states[:,0:1], next_states))
                    for ss in range(num_hist+1):
                        # only use batch size 1 in minibatch
                        # frame is "next state" in replay buffer
                        frame = all_states[batch_idx,ss]
                        action = load_buffer.actions[past_indices[ss]]
                        reward = load_buffer.rewards[past_indices[ss]]
                        if ss == num_hist:
                            # this is the observed state and the only one we will
                            # use a true action/reward for
                            #action = actions[batch_idx]
                            #reward = rewards[batch_idx]
                            terminal_flag = True
                            end_flag = True
                            num_examples += 1
                            if not num_examples % 5000:
                                print('added %s examples to %s buffer'%(num_examples, phase))
                        else:
                            # use this to debug and assert that all actions/rewards
                            # in sampled minibatch of sbuffer are < 99
                            terminal_flag = False
                            end_flag = False

                        sbuffer.add_experience(action, frame, reward, terminal_flag, end_flag)
            sbuffer.rewards = sbuffer.rewards.astype(np.int32)
            sbuffer.init_unique()
            sbuffer.save_buffer(paths[phase])
            buffers[phase] = sbuffer
    return buffers, paths

def test_make_random_subset_buffers():
    # on raza
    train_data_path = '/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_00/breakout_S014342_N0002813995_train.npz'
    dataset_path = '../../dataset/breakout'
    buffers = make_random_subset_buffers(dataset_path, train_data_path, train_max_examples=5000)
    return buffers

if __name__ == '__main__':
    test_make_random_subset_buffers()

