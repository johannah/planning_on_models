import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import configparser
from glob import glob
from copy import deepcopy
import pickle

from skvideo.io import vwrite
from env import Environment
from replay import ReplayMemory

from IPython import embed
import torch

from imageio import imwrite
from skvideo.io import vwrite
from skimage.transform import resize

"""
TODO - create error checking for all of the variables in config
copy code to working directory
plot entire episode observed frames - could do this in a sep script
"""

class ConfigHandler():
    """
    class to wrap configs and setup the housekeeping materials needed to run an experiment
    """
    def __init__(self, config_file, device,
                 restart_last_run=False, restart_run=''):


        self.device = device
        self.output_base = '../../'
        self.model_savedir = 'model_savedir'
        self.random_buffer_dir = os.path.join(self.output_base, self.model_savedir, 'random_buffers')
        self.start_time = time.time()

        self._load_config(config_file)
        self._get_output_name(restart_last_run, restart_run)
        # TODO - when reloading data - cp does happen
        os.system('cp %s %s'%(config_file, self.output_dir))
        self._find_dependent_constants()

    def _get_output_name(self, restart_last_run, restart_run):
        # TODO - if nothing is inside of previous directory - we should use it
        if restart_run != '':
            restart_run_add = os.path.join(self.output_base, self.model_savedir, restart_run)
            if os.path.exists(restart_run):
                self.output_dir = restart_run
            elif os.path.exists(restart_run_add):
                self.output_dir = restart_run_add
            else:
                print("unable to find restart")
                raise Exception
        elif restart_last_run:
            output_search = os.path.join(self.output_base, self.model_savedir, self.cfg['RUN']['name']+'_*')
            self.output_dir = sorted(glob(output_search))[-1]
        else:
            run_num = 0
            self.output_dir = os.path.join(self.output_base, self.model_savedir, self.cfg['RUN']['name']+'_%s_%02d'%(self.cfg['RUN']['train_seed'], run_num))
            # make sure this has not been run before
            while os.path.exists(self.output_dir):
                run_num +=1
                self.output_dir = os.path.join(self.output_base, self.model_savedir, self.cfg['RUN']['name']+'_%s_%02d'%(self.cfg['RUN']['train_seed'], run_num))
        output_dirs = [self.random_buffer_dir, self.output_dir]
        self.create_dirs(output_dirs)
        print("using output dir: %s" %self.output_dir)

    def create_dirs(self, dirs):
        for dd in dirs:
            if not os.path.exists(dd):
                os.makedirs(dd)

    def _load_config(self, config_file):
        self.config_file = config_file
        cfg = configparser.ConfigParser()
        cfg.read(self.config_file)
        self.cfg = dict(cfg)
        for k in cfg.keys():
            self.cfg[k] = dict(cfg[k])
        self._make_correct_types()

    def _make_correct_types(self):
        # all of the values are read as strings
        for section in self.cfg.keys():
            # strings are ok
            for key in self.cfg[section].keys():
                try:
                    val = self.cfg[section][key]
                    if section == 'REP' and key == 'reduction_function':
                        self.cfg[section][key] = eval(val)
                    elif section == 'REP' and key == 'kernel_size':
                        self.cfg[section][key] = eval(val)
                    # correctly load filepath
                    elif '.pt' in val:
                        self.cfg[section][key] = eval(val)
                    # handle float
                    elif '.' in val:
                        self.cfg[section][key] = float(val)
                    # handle int
                    else:
                        self.cfg[section][key] = int(val)
                except Exception:
                    # handle list
                    if ',' in val:
                        list_val = [int(x) for x in self.cfg['ENV'][key][1:-1].split(',')]
                        self.cfg[section][key] = list_val

    def _find_dependent_constants(self):
        self.cfg['ENV']['max_steps'] = self.cfg['ENV']['max_frames']/self.cfg['ENV']['frame_skip']
        self.cfg['PLOT']['fake_acts']= [self.cfg['PLOT']['random_head']for x in range(int(self.cfg['DQN']['n_ensemble']))]
        self.cfg['ENV']['num_rewards'] = len(self.cfg['ENV']['reward_space'])
        self.norm_by = float(self.cfg['ENV']['norm_by'])
        self.num_rewards = len(self.cfg['ENV']['reward_space'])
        self.frame_height = self.cfg['ENV']['obs_width']
        self.frame_width = self.cfg['ENV']['obs_width']
        self.history_length = self.cfg['ENV']['history_size']
        self.maxpool = False
        self.trim_before = 0
        self.trim_after = 0
        self.reduction_fn = None
        self.kernel_size = None
        if 'REP' in self.cfg.keys():
            if 'mp_height' in self.cfg['REP'].keys():
                if self.cfg['REP']['mp_height'] > 0:
                    # if maxpooling is done to output of env.py -> then this is what
                    # goes in the replay buffer. we used maxpool downsample in atari to
                    # get small enough frames, while preserving important info in the
                    # frames
                    self.frame_height = self.cfg['REP']['mp_width']
                    self.frame_width = self.cfg['REP']['mp_width']
                    self.trim_before = self.cfg['REP']['trim_before']
                    self.trim_after = self.cfg['REP']['trim_after']
                    self.kernel_size = self.cfg['REP']['kernel_size']
                    self.maxpool = True
                    self.reduction_fn = eval(self.cfg['REP']['reduction_function'])

    def get_random_buffer_path(self, phase, seed):
        assert phase in ['train', 'eval']
        game = os.path.split(self.cfg['ENV']['game'])[1].split('.')[0]
        n = self.cfg['DQN']['num_pure_random_steps_%s'%phase]
        buffer_size = self.cfg['RUN']['%s_buffer_size'%phase]
        filename = '%s_B%06dS%06d_N%05dRAND_%s.npz' %(game, buffer_size,  seed, n, phase)
        filepath = os.path.join(self.random_buffer_dir, filename)
        base_config = os.path.split(self.config_file)[1].replace('.npz', '_')
        out_config = os.path.join(self.random_buffer_dir, (filename + base_config))
        cmd = 'cp %s %s' %(self.config_file, out_config)
        os.system(cmd)
        # TODO load the comparison
        return filepath

    def get_checkpoint_basepath(self, num_train_steps):
        # always use training steps for reference
        game = os.path.split(self.cfg['ENV']['game'])[1].split('.')[0]
        train_seed = self.cfg['RUN']['train_seed']
        filename = '%s_S%06d_N%010d' %(game, train_seed, num_train_steps)
        filepath = os.path.join(self.output_dir, filename)
        return filepath

    def create_environment(self, seed):
        return Environment(rom_file=self.cfg['ENV']['game'],
                   frame_skip=self.cfg['ENV']['frame_skip'],
                   num_frames=self.cfg['ENV']['history_size'],
                   no_op_start=self.cfg['ENV']['max_no_op_frames'],
                   seed=seed,
                   obs_height=self.cfg['ENV']['obs_height'],
                   obs_width=self.cfg['ENV']['obs_width'],
                   dead_as_end=self.cfg['ENV']['dead_as_end'],
                   max_episode_steps=self.cfg['ENV']['max_episode_steps'])

    def search_for_latest_replay_buffer(self, phase):
        latest = sorted(glob(os.path.join(self.output_dir, '*.npz')))
        if len(latest):
            return latest[-1]
        else:
            return ""

    def create_empty_memory_buffer(self, seed, buffer_size):
        return ReplayMemory(size=buffer_size,
                               frame_height=self.frame_height,
                               frame_width=self.frame_width,
                               agent_history_length=self.history_length,
                               batch_size=self.cfg['DQN']['batch_size'],
                               num_heads=self.cfg['DQN']['n_ensemble'],
                               bernoulli_probability=self.cfg['DQN']['bernoulli_probability'],
                               seed=seed,
                               use_pred_frames=self.cfg['DQN']['use_pred_frames'],
                               # details needed for online max pooling
                               maxpool=self.maxpool,
                               trim_before=self.trim_before,
                               trim_after=self.trim_after,
                               kernel_size=self.kernel_size,
                               reduction_function=self.reduction_fn,
                             )

    def load_memory_buffer(self, phase, load_previously_saved=True):
        """
         phase: string should be "train" or "eval" to indicate which memory buffer to load

         function will load latest experience in the model_savedir/name or create a random replay buffer of specified size to start from
        """
        assert phase in ['train', 'eval']
        buffer_size = self.cfg['RUN']['%s_buffer_size'%phase]
        seed = self.cfg['RUN']['%s_seed'%phase]
        init_empty_with_random=self.cfg['DQN']['load_random_%s_buffer'%phase]
        self.num_random_steps = self.cfg['DQN']['num_pure_random_steps_%s'%phase]
        if load_previously_saved:
            buffer_path = self.search_for_latest_replay_buffer(phase)
            if buffer_path != "":
                print("loading buffer from past experience:%s"%buffer_path)
                return ReplayMemory(load_file=buffer_path)
        if not init_empty_with_random:
            # no buffer file was found, and we want an empty buffer
            print("creating empty replay buffer")
            return self.create_empty_memory_buffer(seed, buffer_size)

        #####################################################
        # from here on - we assume we need random values
        # load a presaved random buffer if it is available
        #random_buffer_path = self.get_random_buffer_path(phase, seed)
        #if os.path.exists(random_buffer_path):
        #    print("loading random replay buffer:%s"%random_buffer_path)
        #    return ReplayMemory(load_file=random_buffer_path)
        #else:
        # no buffer file was found, and we want an empty buffer
        #print('did not find saved replay buffers')
        #print('cannot find a suitable random replay buffers... creating one - this will take some time')
        # did not find any checkpoints - load random buffer
        empty_memory_buffer = self.create_empty_memory_buffer(seed, buffer_size)
        #env = self.create_environment(seed)
        # save the random buffer
        #random_memory_buffer.save_buffer(random_buffer_path)
        return empty_memory_buffer

class StateManager():
    def __init__(self):
        self.reward_space = [-1, 0, 1]
        self.latent_representation_function = None
        pass

    def create_new_state_instance(self, config_handler, phase):
        self.ch = config_handler
        self.save_time = time.time()-100000
        self.phase = phase
        self.step_number = 0
        self.end_step_number = -1
        self.episode_number = 0
        self.seed = self.ch.cfg['RUN']['%s_seed'%self.phase]
        self.random_state = np.random.RandomState(self.seed)
        self.heads = np.arange(self.ch.cfg['DQN']['n_ensemble'])
        self.episodic_reward = []
        self.episodic_reward_avg = []
        self.episodic_step_count = []
        self.episodic_step_ends = []
        self.episodic_loss = []
        self.episodic_times = []
        self.episodic_eps = []

        self.env = self.ch.create_environment(self.seed)
        self.memory_buffer = self.ch.load_memory_buffer(self.phase)
        # TODO should you load the count from the memory buffer - ?
        self.step_number = self.memory_buffer.count
        self.setup_eps()

    def setup_eps(self):
        if self.phase == 'train':
            self.eps_init = self.ch.cfg['DQN']['eps_init']
            self.eps_final = self.ch.cfg['DQN']['eps_final']
            self.eps_annealing_steps = self.ch.cfg['DQN']['eps_annealing_steps']
            self.last_annealing_step = self.eps_annealing_steps + self.ch.cfg['DQN']['num_pure_random_steps_train']
            if self.eps_annealing_steps > 0:
                self.slope = -(self.eps_init - self.eps_final)/self.eps_annealing_steps
                self.intercept = self.eps_init - self.slope*self.ch.cfg['DQN']['num_pure_random_steps_train']

    def load_checkpoint(self, filepath, config_handler=''):
        # load previously saved state file
        fh = open(filepath, 'rb')
        fdict = pickle.load(fh)
        fh.close()
        if config_handler != '':
            # use given config handler
            del fdict['ch']
            self.ch = config_handler

        self.__dict__.update(fdict)

        self.heads = np.arange(self.ch.cfg['DQN']['n_ensemble'])
        self.random_state = np.random.RandomState()
        self.random_state.set_state(fdict['state_random_state'])
        # TODO NOTE this does not restart at same env state
        self.seed = self.ch.cfg['RUN']['%s_seed'%self.phase]
        self.env = self.ch.create_environment(self.seed)
        buffer_path = filepath.replace('.pkl', '.npz')
        self.memory_buffer = ReplayMemory(load_file=buffer_path)
        # TODO should you load the count from the memory buffer - ?
        # TODO what about episode number - it will be off now
        self.step_number = self.memory_buffer.count
        self.setup_eps()

    def save_checkpoint(self, checkpoint_basepath):
        # pass in step number because we always want to use training step number as reference
        self.save_time = time.time()
        self.plot_progress(checkpoint_basepath)
        # TODO save this class - except for random state i assume
        self.memory_buffer.save_buffer(checkpoint_basepath+'.npz')
        # TOO big - prob need to save specifics
        ## preserve random state -
        self.state_random_state = self.random_state.get_state()
        save_dict = {
                    'episodic_reward':self.episodic_reward,
                    'episodic_reward_avg':self.episodic_reward_avg,
                    'episodic_step_count':self.episodic_step_count,
                    'episodic_step_ends':self.episodic_step_ends,
                    'episodic_loss':self.episodic_loss,
                    'episodic_times':self.episodic_times,
                    'state_random_state':self.state_random_state,
                    'episode_number':self.episode_number,
                    'step_number':self.step_number,
                    'phase':self.phase,
                    'save_time':self.save_time,
                    'ch':self.ch,
                    'episodic_eps':self.episodic_eps,
                    }
        fh = open(checkpoint_basepath+'.pkl', 'wb')
        pickle.dump(save_dict, fh)
        fh.close()
        print('finished pickle in', time.time()-self.save_time)

    def end_episode(self):
        # catalog
        self.end_time = time.time()
        self.end_step_number = deepcopy(self.step_number)
        # add to lists
        self.episodic_reward.append(np.sum(self.episode_rewards))
        self.episodic_step_count.append(self.end_step_number-self.start_step_number)
        self.episodic_step_ends.append(self.end_step_number)
        self.episodic_loss.append(np.mean(self.episode_losses))
        self.episodic_times.append(self.end_time-self.start_time)
        try:
            self.episodic_eps.append(self.eps)
        except:
            self.episodic_eps = [1.0 for x in range(len(self.episodic_times))]
        # smoothed reward over last 100 episodes
        self.episodic_reward_avg.append(np.mean(self.episodic_reward[-self.ch.cfg['PLOT']['num_prev_steps_avg']:]))
        num_steps = self.episodic_step_count[-1]
        print("*** %s E%05d S%010d AH%s-R%s num random/total steps:%s/%s***"%(
            self.phase, self.episode_number, self.step_number, self.active_head,
            self.episodic_reward[-1], self.num_random_steps, num_steps ))
        self.episode_active = False
        self.episode_number += 1

    def start_episode(self):
        self.start_time = time.time()
        self.random_state.shuffle(self.heads)
        self.active_head = self.heads[0]
        self.end_step_number = -1

        self.episode_losses = []
        self.episode_actions = []
        self.episode_rewards = []
        self.start_step_number = deepcopy(self.step_number)
        self.num_random_steps = 0

        # restart counters
        self.terminal = False
        self.life_lost = True
        self.episode_reward = 0

        state = self.env.reset()
        self.prev_action = 0
        self.prev_reward = 0
        for i in range(state.shape[0]):
            # add enough memories to use the memory buffer
            # not sure if this is correct
            self.memory_buffer.add_experience(action=0,
                                          frame=state[-1], # use last frame in state bc it is only nonzero one
                                          reward=0,
                                          terminal=0,
                                          end=0,
                                          )

        # get correctly formatted last state
        self.state = self.memory_buffer.get_last_state()
        if self.state.shape != (self.memory_buffer.agent_history_length,self.memory_buffer.frame_height,self.memory_buffer.frame_width):
            print("start shape wrong")
            embed()
        self.episode_active = True
        return self.state

    def plot_current_episode(self, plot_basepath=''):
        if plot_basepath == '':
            plot_basepath = self.get_plot_basepath()
        plot_dict = {
                     'mean loss':self.episode_losses,
                     'actions':self.episode_actions,
                     'rewards':self.episode_rewards,}
        suptitle = 'E%s S%s-%s R%s'%(self.episode_number, self.start_step_number,
                                            self.end_step_number, self.episodic_reward[-1])
        plot_path = plot_basepath+'_ep%06d.png'%self.episode_number
        #step_range = np.arange(self.start_step_number, self.end_step_number)
        #self.plot_data(plot_path, plot_dict, suptitle, xname='episode steps', xdata=step_range)
        self.plot_data(plot_path, plot_dict, suptitle, xname='episode steps')#, xdata=step_range)
        ep_steps = self.end_step_number-self.start_step_number
        self.plot_histogram(plot_basepath+'_ep_histrewards_%06d.png'%self.episode_number, data=self.episode_rewards, bins=self.reward_space,  title='rewards TR%s'%self.episode_reward)
        self.plot_histogram(plot_basepath+'_ep_histactions_%06d.png'%self.episode_number, data=self.episode_actions, bins=self.env.action_space,  title='actions acthead:%s nrand:%s/%s'%(self.active_head, self.num_random_steps, ep_steps))

    def plot_last_episode(self):
        ep_steps = self.end_step_number-self.start_step_number
        ep_states, ep_actions, ep_rewards, ep_next_states, ep_terminals, ep_masks, indexes = self.memory_buffer.get_last_n_states(ep_steps)
        plot_basepath = self.get_plot_basepath()+'_episode_states_frames'
        self.plot_episode_movie(plot_basepath, ep_states, ep_actions, ep_rewards, ep_next_states, ep_terminals, ep_masks, indexes)

    def plot_episode_movie(self, plot_basepath, states, actions, rewards, next_states, terminals, masks, indexes):
        if not os.path.exists(plot_basepath):
            os.makedirs(plot_basepath)
        n_steps = states.shape[0]
        print('plotting episode of length %s'%n_steps)
        if self.latent_representation_function == None:
            n_cols = 2
        else:
            pred_next_states, zs, latents = self.latent_representation_function(states, actions, rewards, self.ch)
            n_cols = 4
        latent_image_path = os.path.join(plot_basepath, 'latent_step_%05d.png')
        ep_reward = sum(rewards)
        movie_path = plot_basepath+'_movie_R%04d.mp4'%ep_reward

        print('starting to make movie', movie_path)
        # write frame by frame then use ffmpeg to generate movie
        #image_path = os.path.join(plot_basepath, 'step_%05d.png')
        #w_path = plot_basepath+'_write_movie_R%04d.sh'%ep_reward
        #a = open(w_path, 'w')
        #cmd = "ffmpeg -n -r 30 -i %s -c:v libx264 -pix_fmt yuv420p %s"%(os.path.abspath(image_path),os.path.abspath(movie_path))
        #a.write(cmd)
        #a.close()
        #w,h = states[0,3].shape
        #treward = 0
        #for step in range(min(n_steps, 100)):
        #    f, ax = plt.subplots(1, n_cols)
        #    if not step%20:
        #        print('plotting step', step)
        #    ax[0].imshow(states[step, 3], cmap=plt.cm.gray)
        #    #ax[0].set_title('OS-A%s' %(actions[step]))
        #    ax[1].imshow(next_states[step, 3], cmap=plt.cm.gray)
        #    treward+=rewards[step]
        #    if self.latent_representation_function != None:
        #        ax[2].imshow(pred_next_states[step], cmap=plt.cm.gray)
        #        z = np.hstack((zs[step,0], zs[step,1], zs[step,2]))
        #        ax[3].imshow(z)
        #    for aa in range(n_cols):
        #        ax[aa].set_xticks([])
        #        ax[aa].set_yticks([])
        #    f.suptitle('%sA%sR%sT%sD%s'%(step, actions[step], rewards[step], treward, int(terminals[step])))
        #    plt.savefig(image_path%step)
        #    plt.close()

        # generate movie directly
        max_frames = 5000
        n = min(n_steps, max_frames)
        for step in range(n):
            if self.latent_representation_function != None:
                z = np.hstack((zs[step,0], zs[step,1], zs[step,2]))
                zo = resize(z, (84,84), cval=0, order=0)
                # TODO - is imwrite clipping zo since it is not a uint8?
                img = np.hstack((states[step,3], next_states[step, 3], pred_next_states[step], zo))
            else:
                img = np.hstack((states[step,3], next_states[step, 3]))

            if not step:
                movie = np.zeros((n, img.shape[0], img.shape[1]))
                latent_movie = np.zeros((n, z.shape[0], z.shape[1]))
            movie[step] = img
            latent_movie[step] = z
        vwrite(movie_path,movie)

    def plot_histogram(self, plot_path, data, bins, title=''):
        n, bins, _ = plt.hist(data, bins=bins)
        plt.xticks(bins, bins)
        plt.yticks(n, n)
        plt.xlim(min(bins), max(bins)+1)
        plt.title(title)
        plt.savefig(plot_path)
        plt.close()

    def plot_progress(self, plot_basepath=''):
        if plot_basepath == '':
            plot_basepath = self.get_plot_basepath()
        det_plot_dict = {
            'episodic step count':self.episodic_step_count,
            'episodic time':self.episodic_times,
            'mean episodic loss':self.episodic_loss,
            'eps':self.episodic_eps,
             }

        suptitle = 'Details E%s S%s'%(self.episode_number, self.end_step_number)
        edet_plot_path = plot_basepath+'_details_episodes.png'
        sdet_plot_path = plot_basepath+'_details_episodes.png'
        if self.end_step_number > 1:
            #exdata = np.arange(self.episode_number)
            #self.plot_data(edet_plot_path, det_plot_dict, suptitle, xname='episode', xdata=exdata)
            #self.plot_data(sdet_plot_path, det_plot_dict, suptitle, xname='steps', xdata=self.episodic_step_ends)
            self.plot_data(edet_plot_path, det_plot_dict, suptitle, xname='episode')#, xdata=exdata)
            self.plot_data(sdet_plot_path, det_plot_dict, suptitle, xname='steps', xdata=self.episodic_step_ends)

            rew_plot_dict = {
                'episodic reward':self.episodic_reward,
                'smooth episodic reward':self.episodic_reward_avg,
            }

            suptitle = 'Reward E%s S%s R%s'%(self.episode_number, self.end_step_number, self.episodic_reward[-1])
            erew_plot_path = plot_basepath+'_reward_episodes.png'
            srew_plot_path = plot_basepath+'_reward_steps.png'
            #self.plot_data(erew_plot_path, rew_plot_dict, suptitle, xname='episode', xdata=np.arange(self.episode_number))
            #self.plot_data(srew_plot_path, rew_plot_dict, suptitle, xname='steps', xdata=self.episodic_step_ends)
            self.plot_data(erew_plot_path, rew_plot_dict, suptitle, xname='episode')#, xdata=np.arange(self.episode_number))
            self.plot_data(srew_plot_path, rew_plot_dict, suptitle, xname='steps', xdata=self.episodic_step_ends)

    def plot_data(self, savepath, plot_dict, suptitle, xname, xdata=None):
        st = time.time()
        print('starting plot data')
        n = len(plot_dict.keys())
        f,ax = plt.subplots(n,1,figsize=(6,3*n))
        #f,ax = plt.subplots(n,1)
        try:
            for xx, name in enumerate(sorted(plot_dict.keys())):
                if xdata is not None:
                    ax[xx].plot(xdata, plot_dict[name])
                else:
                    ax[xx].plot(plot_dict[name])
                ax[xx].set_title('%s'%(name))
                ax[xx].set_ylabel(name)
                print(name, xname, st-time.time())
            ax[xx].set_xlabel(xname)
            f.suptitle('%s %s'%(self.phase, suptitle))
            print('end sup', st-time.time())
            f.savefig(savepath)
            print("saved: %s" %savepath)
            plt.close()
            print('finished')
        except Exception:
            print("plot")
            embed()

    def get_plot_basepath(self):
        return self.ch.get_checkpoint_basepath(self.step_number)+'_%s'%self.phase

    def handle_plotting(self, plot_basepath='', force_plot=False):
        # will plot at beginning of episode
        #if not self.episode_number % self.ch.cfg['PLOT']['plot_episode_every_%s_episodes'%self.phase]:
        # dont plot first episode
        plot_basepath = self.get_plot_basepath()
        if self.episode_number:
            if force_plot:
                self.plot_current_episode(plot_basepath)
                self.plot_progress(plot_basepath)
            if self.episode_number==1 or not self.episode_number % self.ch.cfg['PLOT']['plot_episode_every_%s_episodes'%self.phase]:
                self.plot_current_episode(plot_basepath)
            if self.episode_number==1 or not self.episode_number % self.ch.cfg['PLOT']['plot_every_%s_episodes'%self.phase]:
                self.plot_progress(plot_basepath)

    def step(self, action):
        next_state, reward, self.life_lost, self.terminal = self.env.step(action)
        self.prev_action = action
        self.prev_reward = np.sign(reward)
        # the replay buffer will convert the observed state as needed
        self.memory_buffer.add_experience(action=action,
                                          frame=next_state[-1],
                                          reward=self.prev_reward,
                                          terminal=self.life_lost,
                                          end=self.terminal,
                                            )
        self.episode_actions.append(self.prev_action)
        self.episode_rewards.append(self.prev_reward)
        self.step_number+=1
        self.state = self.memory_buffer.get_last_state()
        if self.state.shape[0] == 0:
            print('handler state chan 0')
            embed()


    def set_eps(self):
        # TODO function to find eps - for now use constant
        if self.step_number <= self.ch.cfg['DQN']['num_pure_random_steps_%s'%self.phase]:
            self.eps = 1.0
        if self.phase == 'train':
            self.eps = self.eps_final
            if self.step_number < self.last_annealing_step:
                self.eps = self.slope*self.step_number+self.intercept
        else:
            self.eps = self.ch.cfg['EVAL']['eps_eval']

    def random_action(self):
        self.num_random_steps +=1
        # pass action_idx to env.action_space
        return self.random_state.choice(range(self.env.num_actions))

    def is_random_action(self):
        self.set_eps()
        r = self.random_state.rand()
        if r < self.eps:
            return True
        else:
            return False


