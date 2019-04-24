import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
from IPython import embed
from glob import glob
from torch.utils.data import Dataset
import os, sys
from imageio import imread, imwrite, mimwrite, imsave, mimsave
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_ubyte
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

oy = 15
# height - oyb
ox = 9
chicken_color = 240
chicken_color1 = 244
chicken_color2 = 246
stripes = 214
road = 142
staging = 170
input_ysize = input_xsize = 80
orig_ysize = 210
orig_xsize = 160
# i dont think this changes
extra_chicken = np.array([[77, 78, 78, 78, 79], [54, 52, 53, 54, 54]])
base_chicken = np.array([[77, 78, 78, 78, 79], [20, 21, 21, 20, 20]])

def prepare_img_small(obs):
    # turn to gray
    cropped = obs[oy:(orig_ysize-oy),ox:]
    gimg = img_as_ubyte(rgb2gray(cropped))
    gimg[stripes == gimg] = 0
    gimg[road == gimg] = 0
    gimg[staging == gimg] = 0
    sgimg = img_as_ubyte(resize(gimg, (int(input_ysize/2), int(input_xsize/2)), order=0))
    sgimg[(extra_chicken[0]/2).astype(np.int), (extra_chicken[1]/2).astype(np.int)] = 0
    our_chicken = np.where(sgimg == chicken_color)
    sgimg[our_chicken[0], our_chicken[1]] = 0
    n,c = np.unique(sgimg, return_counts=True)
    stray_chickens = n[n>240]
    stray_chickens = stray_chickens[stray_chickens<250]
    y,x = list(our_chicken[0]), list(our_chicken[1])

    # remove spurious color from moving chicken
    for sc in stray_chickens:
        oc = np.where(sgimg == sc)
        if not len(y):
            y.extend(oc[0])
            x.extend(oc[1])
        sgimg[oc[0], oc[1]] = 0

    if not len(y):
        print("COULDNT FIND CHICKEN IN OBSERVED IMAGE")
        y = (base_chicken[0]/2).astype(np.int)
        x = (base_chicken[1]/2).astype(np.int)

    our_chicken = (np.array(y), np.array(x))
    # there are 14 more 252 color when chicken moves -
    # there are 14 more 252 color when chicken moves -
    #our_chicken2 = np.where(sgimg == chicken_color2)
    #sgimg[our_chicken2[0], our_chicken2[1]] = 0
    return our_chicken, sgimg


def prepare_img(obs):
    # turn to gray
    cropped = obs[oy:(orig_ysize-oy),ox:]
    gimg = img_as_ubyte(rgb2gray(cropped))
    gimg[stripes == gimg] = 0
    gimg[road == gimg] = 0
    gimg[staging == gimg] = 0
    sgimg = img_as_ubyte(resize(gimg, (input_ysize, input_xsize), order=0))
    sgimg[extra_chicken[0], extra_chicken[1]] = 0
    our_chicken = np.where(sgimg == chicken_color)
    sgimg[our_chicken[0], our_chicken[1]] = 0
    n,c = np.unique(sgimg, return_counts=True)
    stray_chickens = n[n>240]
    stray_chickens = stray_chickens[stray_chickens<250]
    y,x = list(our_chicken[0]), list(our_chicken[1])

    # remove spurious color from moving chicken
    for sc in stray_chickens:
        oc = np.where(sgimg == sc)
        if not len(y):
            y.extend(oc[0])
            x.extend(oc[1])
        sgimg[oc[0], oc[1]] = 0

    if not len(y):
        print("COULDNT FIND CHICKEN IN OBSERVED IMAGE")
        y = base_chicken[0]
        x = base_chicken[1]

    our_chicken = (np.array(y), np.array(x))
    # there are 14 more 252 color when chicken moves -
    # there are 14 more 252 color when chicken moves -
    #our_chicken2 = np.where(sgimg == chicken_color2)
    #sgimg[our_chicken2[0], our_chicken2[1]] = 0
    return our_chicken, sgimg

def undo_img_scaling(sgimg, our_chicken):
    sgimg[our_chicken] = chicken_color
    rec = np.zeros((orig_ysize, orig_xsize))
    outimg = img_as_ubyte(resize(sgimg, (orig_ysize-(oy*2), orig_xsize-ox), order=0))
    rec[oy:(orig_ysize-oy),ox:] = outimg
    return rec

def transform_freeway(obs):
    h,w,c = obs.shape
    o = obs[25:(h-26),9:,:]
    cr = img_as_ubyte(rgb2gray(resize(o, (80,80), order=0)))
    return cr

def remove_background(gimg):
    # background has two colors - 142 is the asphalt and 214 is the lines
    gimg[gimg==road] = 0
    gimg[gimg==stripes] = 0
    return gimg

def remove_chicken(gimg):
    # remove chicken from background removed image
    chicken = np.where(gimg==chicken_color)
    gimg[chicken] = 0
    return gimg, chicken

class FroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None, max_pixel_used=254.0, min_pixel_used=0.0):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, '*.png')
        self.max_pixel_used = max_pixel_used
        self.min_pixel_used = min_pixel_used
        ss = sorted(glob(search_path))
        self.indexes = [s for s in ss if 'gen' not in s]
        print("found %s files in %s" %(len(self.indexes), search_path))

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = self.indexes[idx]
        image = imread(img_name)
        if len(image.shape) == 2:
            image = image[:,:,None].astype(np.float32)
        if self.transform is not None:
            # bt 0 and 1
            image = (self.transform(image)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)

        return image,img_name


class FreewayForwardDataset(Dataset):
    def __init__(self,  data_file, number_condition=4,
                        steps_ahead=1, limit=None, batch_size=300,
                        max_pixel_used=254.0, min_pixel_used=0.0, augment_file="None",
                        rdn_num=3949):

        self.rdn = np.random.RandomState(rdn_num)
        if augment_file is not "None":
            self.do_augment = True
        else:
            self.do_augment = False

        # index right now is by the oldest observation needed to compute
        # prediction
        self.data_file = os.path.abspath(data_file)
        self.augment_data_file = os.path.abspath(augment_file)
        self.num_condition = int(number_condition)
        assert(self.num_condition>0)
        self.steps_ahead = int(steps_ahead)
        assert(self.steps_ahead>=0)
        self.max_pixel_used = max_pixel_used
        self.min_pixel_used = min_pixel_used
        self.data = np.load(self.data_file)['arr_0']
        if self.do_augment:
            self.augmented_data = np.load(self.augment_data_file)['arr_0']
        _,self.data_h,self.data_w = self.data.shape
        self.num_examples = self.data.shape[0]-(self.steps_ahead+self.num_condition)
        # index by observation number ( last sample of conditioning )
        # if i ask for frame 3 - return w/ steps_ahead=1
        # x = data[0,1,2,3]
        # y = data[4]
        self.index_array = np.arange(self.num_condition-1, self.data.shape[0]-self.steps_ahead)

    def __max__(self):
        return max(self.index_array)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        try:
            n = idx.shape[0]
        except:
            n = 1
        dx = []
        add_range = np.arange(-(self.num_condition-1), 1)
        # old way which doesnt allow augmentation
        #for i in add_range:
        #    i_idx = idx+i
        #    dx.append(self.data[i_idx])
        #dx = np.array(dx).swapaxes(1,0)

        if self.do_augment:
            # choose the augmented data for the most recent observations some of
            # the time
            start_augment_idx = self.rdn.choice(add_range, n, replace=True)
        else:
            # always choose the real data
            start_augment_idx = np.zeros((n))

        for nidx, start in enumerate(idx):
            this_sample = []
            for i in add_range:
                i_idx = start+i
                if i > start_augment_idx[nidx]:
                    pp = 'pred'
                    this_sample.append(self.augmented_data[i_idx])
                else:
                    pp = 'real'
                    this_sample.append(self.data[i_idx])
                #print(start, i_idx, i,pp, start_augment_idx[nidx])
            dx.append(this_sample)

        dy = self.data[idx+self.steps_ahead][:,None]
        x = (torch.FloatTensor(dx)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)
        y = (torch.FloatTensor(dy)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)
        return x,y

def find_component_proportion(data):
    unique = list(set(data))
    component_percentages = []
    num = data.shape[0]
    for u in unique:
        num_u = np.where(data == u)[0].shape[0]
        component_percentages.append(num_u/float(num))
    return num, unique, component_percentages

class ForwardLatentDataset(Dataset):
    def __init__(self,  data_file, batch_size=128, seed=9):
        self.random_state = np.random.RandomState(seed)
        self.batch_size = batch_size
        # index next observation, will need
        self.data_filename = os.path.abspath(data_file)
        self.data_file = np.load(self.data_filename)
        # output of vqvae embedding
        self.prev_latents = self.data_file['prev_latents']
        self.prev_actions = self.data_file['prev_actions']
        self.prev_rewards = self.data_file['prev_rewards']
        self.prev_values = self.data_file['prev_values']
        self.latents = self.data_file['latents']
        self.next_latents = self.data_file['next_latents']
        self.rewards = self.data_file['rewards']
        self.values = self.data_file['values']
        self.actions = self.data_file['actions']
        _, self.unique_rewards, self.percentages_rewards = find_component_proportion(self.rewards)
        _, self.action_space, self.percentages_actions = find_component_proportion(self.actions)
        # should probably add terminals
        self.n_actions = len(self.action_space)
        self.num_examples,self.data_h,self.data_w = self.latents.shape
        self.index_array = list(np.arange(0, self.num_examples, dtype=np.int))

    def reset_batch(self):
        self.unique_index_array = deepcopy(self.index_array)

    def get_data(self, indexes, reset=False):
        # action corresponds to the action that moves s_t to s_t+1
        #return torch.FloatTensor(self.latents[indexes]),
        #torch.LongTensor(self.actions[indexes]),
        #torch.LongTensor(self.rewards[indexes]), torch.FloatTensor(self.values[indexes]),
        #torch.LongTensor(self.next_latents[indexes]), reset, indexes
        # S0  S1  S2**S3 - prev
        #               A0
        #     S1  S2  S3**S4  - state
        #                   A1
        #          S2  S3  S4**S5  - next
        #                          A3
        data = (self.prev_latents[indexes], self.prev_actions[indexes],
                self.prev_rewards[indexes], self.prev_values[indexes],
                self.latents[indexes], self.actions[indexes],
                self.rewards[indexes], self.values[indexes],
                self.next_latents[indexes], reset, indexes)
        return data

    def get_minibatch(self):
        indexes = self.random_state.choice(self.index_array, self.batch_size)
        return self.get_data(indexes)

    def get_unique_minibatch(self):
        reset = False
        indexes = self.random_state.choice(self.unique_index_array, self.batch_size, replace=False)
        # remove used indexes
        not_in = np.logical_not(np.isin(self.unique_index_array, indexes))
        self.unique_index_array = self.unique_index_array[not_in]
        if len(self.unique_index_array) < self.batch_size:
            self.reset_batch()
            reset = True
        return self.get_data(indexes, reset)


class AtariDataset(Dataset):
    def __init__(self,  data_file, number_condition=4, steps_ahead=1,
                        limit=None, batch_size=128,
                        norm_by=255.0,
                        seed=39):

        self.random_state = np.random.RandomState(seed)
        self.batch_size = batch_size
        # index next observation, will need
        self.data_file_name = os.path.abspath(data_file)
        self.num_condition = int(number_condition)
        assert(self.num_condition>0)
        self.steps_ahead = int(steps_ahead)
        assert(self.steps_ahead>=0)
        self.norm_by = float(norm_by)
        self.data_file = np.load(self.data_file_name)
        # TODO! Rewards should be normalized
        self.rewards = self.data_file['rewards'].astype(np.int16)
        self.values = self.data_file['values'].astype(np.float32)
        self.unique_rewards = list(set(self.rewards))
        self.terminals = self.data_file['terminals'].astype(np.int16)
        self.actions = self.data_file['actions'].astype(np.int16)
        self.action_space = sorted(list(set(self.actions)))
        self.n_actions = len(self.action_space)
        self.episodic_reward=self.data_file['episodic_reward']
        _, self.unique_rewards, self.percentages_rewards = find_component_proportion(self.rewards)
        _, self.action_space, self.percentages_actions = find_component_proportion(self.actions)
        self.frames = self.data_file['frames']
        #self.num_examples,self.data_h,self.data_w = self.data_file['frames'].shape
        self.num_examples,self.data_h,self.data_w = self.frames.shape

        self.index_array = list(np.arange(self.num_condition, self.num_examples, dtype=np.int))
        # ending indexes cannot be selected
        to_remove = []
        for index in self.index_array:
            if np.sum(self.terminals[index-self.num_condition:index])>0:
                to_remove.append(index)
        #print('removing episode start indices', to_remove)
        for pl, index in enumerate(to_remove):
            self.index_array.remove(index)
        self.index_array = np.array(self.index_array)
        self.relative_indexes = np.arange(len(self.index_array))
        self.reset_batch()
        # location of start and ends of episodes in "relative indexes"
        self.ends = list(np.where(self.terminals[self.index_array[self.relative_indexes]] == 1)[0])[:-1]
        #self.starts = [e+1 for e in self.ends]
        # dont start at very beginning because need-1 states in generating
        self.starts = [e+1 for e in self.ends]
        self.starts.insert(0,0)
        self.ends.append(len(self.relative_indexes))
        self.episode_indexes = [x for x in range(len(self.starts))]

    def plot_dataset(self):
        ddir = self.data_file_name.replace('.npz', '_imgs')
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        for episode_index in range(len(self.starts)):
            data, episode_index, episode_reward = self.get_episode_by_index(episode_index, diff=True)
            states, actions, rewards, values, pred_states, terminals, is_new_epoch, relative_indexes = data
            gif_name = os.path.join(ddir, 'EI%05d_R%02d_N%05d.gif'%(episode_index, episode_reward, states.shape[0]))
            print('plotting %s'%gif_name)
            mimsave(gif_name, states[:,-1])
            for i in range(1,min(states.shape[0]-1, 300)):
                png_name = os.path.join(ddir, 'EI%05d_R%02d_N%05d.png'%(episode_index, episode_reward, i))
                if not os.path.exists(png_name):
                    f,ax = plt.subplots(1,2)
                    ax[0].set_title("%05d A%d" %(i,actions[i]))
                    ax[0].imshow(states[i-1,-1])
                    ax[1].imshow(states[i,-1])
                    plt.savefig(png_name)
                    plt.close()
        embed()

    def get_episode_by_index(self, episode_index, limit=0, diff=False):
        relative_indexes = np.arange(self.starts[episode_index], self.ends[episode_index], dtype=np.int)
        if limit > 0:
            print("limiting episode to %s steps" %limit)
            relative_indexes = relative_indexes[:limit]
        episode_reward = self.episodic_reward[episode_index]
        if not diff:
            data = self.get_data(relative_indexes)
        else:
            data = self.get_framediff_data(relative_indexes)
        return (data, episode_index, episode_reward)

    def get_entire_episode(self, diff=False, limit=-10, min_reward=-99):
        ep_reward = -9999
        while ep_reward < min_reward:
            episode_index = self.random_state.choice(self.episode_indexes)
            print('grabbing episode %s [%s:%s] of reward %s' %(episode_index,
                                  self.starts[episode_index], self.ends[episode_index],
                                  self.episodic_reward[episode_index]))
            ep_reward = self.episodic_reward[episode_index]
        return self.get_episode_by_index(episode_index, limit=limit, diff=diff)

    def reset_batch(self):
        self.unique_index_array = deepcopy(self.relative_indexes)

    def __getstates__(self, index):
        # determine if this is beginning of episode
        # index refers to the next_state index
        #frames = self.data_file['frames'][index-self.num_condition:index+1].astype(np.float32)/255.0
        frames = self.frames[index-self.num_condition:index+1]
        try:
            assert (np.sum(self.terminals[index-self.num_condition:index]) == 0)
        except:
            print("terminals")
            embed()
        state = frames[:-1]
        next_state = frames[1:]
        # next state is one frame ahead, so the observed frame in state is -1
        # and the observed frame in next_state is -2, by index. predicted (step
        # ahead) frame is next_state[-1]
        try:
            assert (np.sum(state[-1]) == np.sum(next_state[-2]))
            assert (np.sum(state[-2]) == np.sum(next_state[-3]))
        except:
            print("assert dataset")
            embed()

        return state, next_state

    def get_data(self, relative_indexes, reset=False):
        # action corresponds to the action that moves s_t to s_t+1
        indexes = self.index_array[relative_indexes]
        batch_size = len(relative_indexes)
        mb_states = np.zeros((batch_size, self.num_condition, self.data_h, self.data_w), np.float32)
        mb_next_states = np.zeros((batch_size, self.num_condition, self.data_h, self.data_w), np.float32)
        for i, idx in enumerate(indexes):
            # todo - proper norm
            st, nst = self.__getstates__(idx)
            #print(i, idx, st.sum())
            mb_states[i] = st.astype(np.float32)/255.0
            mb_next_states[i] = nst.astype(np.float32)/255.0
        # these assertions should be true - commented out to save time
        return mb_states, self.actions[indexes], self.rewards[indexes], self.values[indexes], mb_next_states, self.terminals[indexes], reset, relative_indexes

    def get_minibatch(self):
        relative_indexes = self.random_state.choice(self.relative_indexes, self.batch_size)
        return self.get_data(relative_indexes)

    def get_unique_minibatch(self):
        reset = False
        relative_indexes = self.random_state.choice(self.unique_index_array, self.batch_size, replace=False)
        # remove used indexes
        not_in = np.logical_not(np.isin(self.unique_index_array, relative_indexes))
        self.unique_index_array = self.unique_index_array[not_in]
        if len(self.unique_index_array) < self.batch_size:
            self.reset_batch()
            reset = True
        return self.get_data(relative_indexes, reset)

    def get_entire_episode(self, diff=False, limit=-10, min_reward=-99):
        ep_reward = -9999
        while ep_reward < min_reward:
            episode_index = self.random_state.choice(self.episode_indexes)
            print('grabbing episode %s [%s:%s] of reward %s' %(episode_index,
                                  self.starts[episode_index], self.ends[episode_index],
                                  self.episodic_reward[episode_index]))
            ep_reward = self.episodic_reward[episode_index]

        relative_indexes = np.arange(self.starts[episode_index], self.ends[episode_index], dtype=np.int)
        if limit > 0:
            print("limiting episode to %s steps" %limit)
            relative_indexes = relative_indexes[:limit]
        episode_reward = self.episodic_reward[episode_index]
        if not diff:
            data = self.get_data(relative_indexes)
        else:
            data = self.get_framediff_data(relative_indexes)
        return (data, episode_index, episode_reward)

    def get_framediff_data(self, relative_indexes, reset=False):
        indexes = self.index_array[relative_indexes]
        batch_size = len(relative_indexes)
        mb_states = np.zeros((batch_size, self.num_condition, self.data_h, self.data_w), np.float32)
        mb_pred_states = np.zeros((batch_size, 2, self.data_h, self.data_w), np.float32)
        for i, idx in enumerate(indexes):
            # todo - proper norm
            st, nst = self.__getstates__(idx)
            # use nst so the action is coherent
            nst = nst.astype(np.float32)/255.0
            mb_states[i] =  nst
            # reconstruct the previously observed
            mb_pred_states[i,0] = nst[-1]
            # reconstruct the frame difference between observed frame and the
            # previous frame
            mb_pred_states[i,1] = nst[-2]-nst[-1]
            # the action is the action which brings us from nst[-2] to nst[-1] (
            # (or st[-1] to nst[-1])
        #return torch.FloatTensor(mb_states), torch.LongTensor(self.actions[indexes]), torch.LongTensor(self.rewards[indexes]), torch.FloatTensor(self.values[indexes]), torch.FloatTensor(mb_pred_states), torch.LongTensor(self.terminals[indexes]), reset, relative_indexes
        # TODO - reinstate self.mb_states to save time
        # TODO - also fix all the code i broke by making this not a tensor
        return mb_states, self.actions[indexes], self.rewards[indexes], self.values[indexes], mb_pred_states, self.terminals[indexes], reset, relative_indexes

    def get_framediff_minibatch(self):
        relative_indexes = self.random_state.choice(self.relative_indexes, self.batch_size)
        return self.get_framediff_data(relative_indexes)

# used to be Dataloader, but overloaded
class AtariDataLoader():
    def __init__(self, train_load_function, test_load_function,
                 batch_size, random_number=394):

        self.done = False
        self.test_done = False

        self.batch_size = batch_size
        self.test_loader = test_load_function
        self.train_loader = train_load_function
        self.last_test_batch_idx = self.test_loader.index_array.min()+1
        self.last_batch_idx = self.train_loader.index_array.min()+1

        self.train_rdn = np.random.RandomState(random_number)
        self.test_rdn = np.random.RandomState(random_number)

        self.reset_batch()
        self.num_batches = len(self.train_loader)/self.batch_size

    def reset_batch(self):
        self.unique_index_array = np.arange(self.train_loader.index_array.shape[0], dtype=np.int)

    def validation_data(self):
        batch_choice = self.test_rdn.choice(self.test_loader.index_array, self.batch_size, replace=False)
        vx,vy = self.test_loader[batch_choice]
        return vx,vy,batch_choice

    def validation_ordered_batch(self):
        batch_choice = np.arange(self.last_test_batch_idx, self.last_test_batch_idx+self.batch_size)
        self.last_test_batch_idx += self.batch_size
        batch_choice = batch_choice[batch_choice<max(self.test_loader.index_array)]
        batch_choice = batch_choice[batch_choice>min(self.test_loader.index_array)]
        if batch_choice.shape[0] <= 1:
            self.test_done = True
        x,y = self.test_loader[batch_choice]
        return x,y,batch_choice

    def ordered_batch(self):
        batch_choice = np.arange(self.last_batch_idx, self.last_batch_idx+self.batch_size)
        self.last_batch_idx += self.batch_size
        batch_choice = batch_choice[batch_choice<max(self.train_loader.index_array)]
        batch_choice = batch_choice[batch_choice>min(self.train_loader.index_array)]
        if batch_choice.shape[0] <= 1:
            self.done = True
        x,y = self.train_loader[batch_choice]
        return x,y,batch_choice

    def next_batch(self):
        batch_choice = self.train_rdn.choice(self.train_loader.index_array, self.batch_size, replace=False)
        x,y = self.train_loader[batch_choice]
        return x,y,batch_choice

    def next_unique_batch(self):
        reset = False
        batch_indexes = self.train_rdn.choice(self.unique_index_array, self.batch_size, replace=False)
        # absolute indexes are different than the indexing indexes
        batch_choices = self.train_loader.index_array[batch_indexes]
        x,y = self.train_loader[batch_choices]
        # remove used indexes
        not_in = np.logical_not(np.isin(self.unique_index_array, batch_indexes))
        self.unique_index_array = self.unique_index_array[not_in]
        if len(self.unique_index_array) < self.batch_size:
            self.reset_batch()
            reset = True
        return x,y,batch_choices,reset


class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)




#
#z_q_x_mean = 0.16138
#z_q_x_std = 0.7934
#
#class VqvaeDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=None):
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, '*z_q_x.npy')
#        self.indexes = sorted(glob(search_path))
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            raise
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        data_name = self.indexes[idx]
#        data = np.load(data_name).ravel().astype(np.float32)
#        # normalize v_q
#        data = (data-z_q_x_mean)/z_q_x_std
#        # normalize for embedding space
#        #data = 2*((data/512.0)-0.5)
#        return data,data_name
#
#class EpisodicVaeFroggerDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
#        # what really matters is the seed - only generated one game per seed
#        #seed_00334_episode_00029_frame_00162.png
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, search)
#        self.indexes = sorted(glob(search_path))
#        print("will use transform:%s"%transform)
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            sys.exit()
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        dname = self.indexes[idx]
#        d = np.load(open(dname, 'rb'))
#        mu = d['mu'].astype(np.float32)[:,best_inds]
#        sig = d['sigma'].astype(np.float32)[:,best_inds]
#        if self.transform == 'pca':
#            if not idx:
#                print("tranforming dataset using pca")
#            mu_scaled = mu-vae_mu_mean
#            mu_scaled = (np.dot(mu_scaled, V.T)/Xpca_std).astype(np.float32)
#        elif self.transform == 'std':
#            mu_scaled= ((mu-vae_mu_mean)/vae_mu_std).astype(np.float32)
#        else:
#            mu_scaled = mu
#            sig_scaled = sig
#
#        return mu_scaled,mu,sig_scaled,sig,dname
#
#
#class EpisodicDiffFroggerDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
#        # what really matters is the seed - only generated one game per seed
#        #seed_00334_episode_00029_frame_00162.png
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, search)
#        self.indexes = sorted(glob(search_path))
#        dparams = np.load('vae_diff_params.npz')
#        self.mu_diff_mean = dparams['mu_diff_mean'][best_inds]
#        self.mu_diff_std = dparams['mu_diff_std'][best_inds]
#        self.sig_diff_mean = dparams['sig_diff_mean'][best_inds]
#        self.sig_diff_std = dparams['sig_diff_std'][best_inds]
#        print("will use transform:%s"%transform)
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            sys.exit()
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        if idx == 0:
#            print("loading first file")
#        dname = self.indexes[idx]
#        d = np.load(open(dname, 'rb'))
#        mu = d['mu'].astype(np.float32)[:,best_inds]
#        sig = d['sigma'].astype(np.float32)[:,best_inds]
#        mu_diff = np.diff(mu,n=1,axis=0)
#        sig_diff = np.diff(sig,n=1,axis=0)
#        if self.transform == 'std':
#            if not idx:
#                print("performing transform std")
#            mu_diff_scaled= ((mu_diff-self.mu_diff_mean)/self.mu_diff_std).astype(np.float32)
#            sig_diff_scaled= ((sig_diff-self.sig_diff_mean)/self.sig_diff_std).astype(np.float32)
#            # how to unscale
#            #mu_diff_unscaled = ((mu_scaled*self.mu_diff_std)+self.mu_diff_mean).astype(np.float32)
#            #sig_diff_unscaled = ((sig_scaled*self.sig_diff_std)+self.sig_diff_mean).astype(np.float32)
#        else:
#            if not idx:
#                print("performing no data transform")
#            mu_diff_scaled = mu_diff
#            sig_diff_scaled = sig_diff
#        return mu_diff_scaled,mu_diff,mu,sig_diff_scaled,sig_diff,sig,dname
#
#
class EpisodicVqVaeFroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=-1, search='seed*episode*.npz'):
        # what really matters is the seed - only generated one game per seed
        #seed_00334_episode_00029_frame_00162.png
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, search)
        self.indexes = sorted(glob(search_path))
        print("will use transform:%s"%transform)
        print("found %s files in %s" %(len(self.indexes), search_path))
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        dname = self.indexes[idx]
        d = np.load(open(dname, 'rb'))
        latents = d['latents']
        return latents,dname



class AtariActionDataset(Dataset):
    def __init__(self,  data_file, number_condition=4,
                        steps_ahead=1, limit=None, batch_size=300,
                        augment_file="None",
                        rdn_num=3949):

        self.rdn = np.random.RandomState(rdn_num)
        if augment_file is not "None":
            self.do_augment = True
        else:
            self.do_augment = False

        # index right now is by the oldest observation needed to compute
        # prediction
        self.data_file = os.path.abspath(data_file)
        self.augment_data_file = os.path.abspath(augment_file)
        self.num_condition = int(number_condition)
        assert(self.num_condition>0)
        self.steps_ahead = int(steps_ahead)
        assert(self.steps_ahead>=0)
        self.data = np.load(self.data_file)
        if self.do_augment:
            self.augmented_data = np.load(self.augment_data_file)
        num_examples,self.data_h,self.data_w = self.data['states'].shape
        self.num_examples = self.num_examples-(self.steps_ahead+self.num_condition)
        # index by observation number ( last sample of conditioning )
        # if i ask for frame 3 - return w/ steps_ahead=1
        # x = data[0,1,2,3], action
        # y = data[1,2,3,4], reward
        self.index_array = np.arange(self.num_condition-1, self.data.shape[0]-self.steps_ahead)

    def __max__(self):
        return max(self.index_array)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        try:
            n = idx.shape[0]
        except:
            n = 1
        dx = []
        add_range = np.arange(-(self.num_condition-1), 1)
        # old way which doesnt allow augmentation
        #for i in add_range:
        #    i_idx = idx+i
        #    dx.append(self.data[i_idx])
        #dx = np.array(dx).swapaxes(1,0)

        if self.do_augment:
            # choose the augmented data for the most recent observations some of
            # the time
            start_augment_idx = self.rdn.choice(add_range, n, replace=True)
        else:
            # always choose the real data
            start_augment_idx = np.zeros((n))

        for nidx, start in enumerate(idx):
            this_sample = []
            for i in add_range:
                i_idx = start+i
                if i > start_augment_idx[nidx]:
                    pp = 'pred'
                    this_sample.append(self.augmented_data[i_idx])
                else:
                    pp = 'real'
                    this_sample.append(self.data[i_idx])
                #print(start, i_idx, i,pp, start_augment_idx[nidx])
            dx.append(this_sample)

        dy = self.data[idx+self.steps_ahead][:,None]
        x = (torch.FloatTensor(dx)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)
        y = (torch.FloatTensor(dy)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)
        return x,y




