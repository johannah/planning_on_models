import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import configparser
from glob import glob
from copy import deepcopy
import _pickle as cPickle

from env import Environment
from replay import ReplayMemory

from IPython import embed
"""
TODO - create error checking for all of the variables in config
Copy .ini file over to working directory
copy code to working directory
"""

def collect_random_experience(seed, env, memory_buffer, num_random_steps, num_actions):
    # note that since we are making the env different here
    # we should always use a different env for the random portion vs the
    # learning agent
    # create new replay memory
    print("starting random experience collection for %s steps"%num_random_steps)
    step_number = 0
    epoch_num = 0
    random_state = np.random.RandomState(seed)
    heads = np.arange(memory_buffer.num_heads)
    while step_number < num_random_steps:
        epoch_frame = 0
        terminal = False
        life_lost = True
        # use real state
        episode_reward_sum = 0
        random_state.shuffle(heads)
        active_head = heads[0]
        print("Gathering data with head=%s"%active_head)
        # at every new episode - recalculate action/reward weight
        state = env.reset()
        while not terminal:
            action = random_state.randint(0, num_actions-1)
            next_state, reward, life_lost, terminal = env.step(action)
            # TODO - dead as end? should be from ini file or is it handled
            # in env?
            # Store transition in the replay memory
            memory_buffer.add_experience(action=action,
                                         frame=next_state[-1],
                                         reward=1+np.sign(reward), # add one so that rewards are <=0
                                         terminal=life_lost,
                                         )
            step_number += 1
            epoch_frame += 1
            episode_reward_sum += reward
        print("finished epoch %s: reward %s" %(epoch_num, episode_reward_sum))
        print("%s/%s steps completed" %(step_number, num_random_steps))
        epoch_num += 1
    return memory_buffer

class ConfigHandler():
    """
    class to wrap configs and setup the housekeeping materials needed to run an experiment
    """
    def __init__(self, config_file, device, restart_last_run=False, restart_run=''):

        self.device = device
        self.output_base = '../../'
        self.model_savedir = 'model_savedir'
        self.random_buffer_dir = os.path.join(self.output_base, self.model_savedir, 'random_buffers')
        self.start_time = time.time()

        self._load_config(config_file)
        self._get_output_name(restart_last_run, restart_run)
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
                    # handle float
                    if '.' in val:
                        self.cfg[section][key] = float(val)
                    # hadnle int
                    else:
                        self.cfg[section][key] = int(val)
                except Exception:
                    # handle list
                    if ',' in val:
                        list_val = [int(x) for x in self.cfg['ENV']['action_space'][1:-1].split(',')]
                        self.cfg[section][key] = list_val

    def _find_dependent_constants(self):
        self.cfg['ENV']['max_steps'] = self.cfg['ENV']['max_frames']/self.cfg['ENV']['frame_skip']
        self.cfg['PLOT']['fake_acts']= [self.cfg['PLOT']['random_head']for x in range(int(self.cfg['DQN']['n_ensemble']))]
        self.cfg['ENV']['num_rewards'] = len(self.cfg['ENV']['reward_space'])
        self.norm_by = float(self.cfg['ENV']['norm_by'])
        self.num_rewards = len(self.cfg['ENV']['reward_space'])
        self.num_actions = len(self.cfg['ENV']['action_space'])

    def get_random_buffer_path(self, phase, seed):
        assert phase in ['train', 'eval']
        game = os.path.split(self.cfg['ENV']['game'])[1].split('.')[0]
        n = self.cfg['RUN']['num_pure_random_steps_%s'%phase]
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
                   dead_as_end=self.cfg['ENV']['dead_as_end'],
                   max_episode_steps=self.cfg['ENV']['max_episode_steps'])

    def search_for_latest_replay_buffer(self, phase):
        latest = sorted(glob(os.path.join(self.output_dir, '*.npz')))
        if len(latest):
            return latest[-1]
        else:
            return ""

    def create_empty_memory_buffer(self, seed, buffer_size):
        return  ReplayMemory(action_space=self.cfg['ENV']['action_space'],
                               size=buffer_size,
                               frame_height=self.cfg['ENV']['obs_height'],
                               frame_width=self.cfg['ENV']['obs_width'],
                               agent_history_length=self.cfg['ENV']['history_size'],
                               batch_size=self.cfg['DQN']['batch_size'],
                               num_heads=self.cfg['DQN']['n_ensemble'],
                               bernoulli_probability=self.cfg['DQN']['bernoulli_probability'],
                               seed=seed,
                                       )

    def load_memory_buffer(self, phase, load_previously_saved=True):
        """
         phase: string should be "train" or "eval" to indicate which memory buffer to load

         function will load latest experience in the model_savedir/name or create a random replay buffer of specified size to start from
        """
        assert phase in ['train', 'eval']
        buffer_size = self.cfg['RUN']['%s_buffer_size'%phase]
        seed = self.cfg['RUN']['%s_seed'%phase]
        init_empty_with_random=self.cfg['RUN']['load_random_%s_buffer'%phase]
        num_random_steps = self.cfg['RUN']['num_pure_random_steps_%s'%phase]
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
        random_buffer_path = self.get_random_buffer_path(phase, seed)
        if os.path.exists(random_buffer_path):
            print("loading random replay buffer:%s"%random_buffer_path)
            return ReplayMemory(load_file=random_buffer_path)
        else:
            # no buffer file was found, and we want an empty buffer
            print('did not find saved replay buffers')
            print('cannot find a suitable random replay buffers... creating one - this will take some time')
            # did not find any checkpoints - load random buffer
            random_memory_buffer = self.create_empty_memory_buffer(seed, buffer_size)

            env = self.create_environment(seed)
            random_memory_buffer = collect_random_experience(seed, env, random_memory_buffer, num_random_steps, self.num_actions)
            # save the random buffer
            random_memory_buffer.save_buffer(random_buffer_path)
            return random_memory_buffer

class StateManager():
    def __init__(self, config_handler):
        self.ch = config_handler

    def create_new_state_instance(self, phase):
        self.save_time = time.time()-100000
        self.phase = phase
        self.step_number = 0
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

        self.env = self.ch.create_environment(self.seed)
        self.memory_buffer = self.ch.load_memory_buffer(self.phase)


    def load_checkpoint(self, filepath):
        # load previously saved state file
        fh = open(filepath, 'rb')
        fdict = cPickle.load(fh)
        fh.close()
        self.__dict__.update(fdict)

        # handle random state
        self.random_state = np.random.RandomState()
        self.random_state.set_state(self.state_random_state)
        del self.state_random_state

        buffer_path = filepath.replace('.pkl', '.npz')
        self.memory_buffer = ReplayMemory(load_file=buffer_path)
        # TODO NOTE this does not restart at same env state
        self.env = self.ch.create_environment(self.seed)

    def save_checkpoint(self, checkpoint_basepath):
        # pass in step number because we always want to use training step number as reference
        self.save_time = time.time()
        self.plot_progress(checkpoint_basepath)
        # TODO save this class - except for random state i assume
        self.memory_buffer.save_buffer(checkpoint_basepath+'.npz')
        # TOO big - prob need to save specifics
        ## preserve random state -
        #self.state_random_state = self.random_state.get_state()
        #fh = open(checkpoint_basepath+'.pkl', 'wb')
        #myvars = deepcopy(self.__dict__)
        ## delete the big stuff
        #del myvars['episode_actions']
        #del myvars['episode_rewards']
        #del myvars['random_state']
        #del myvars['memory_buffer']
        #del myvars['env']
        #cPickle.dump(myvars, fh, 2)
        #fh.close()
        #del myvars

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
        # smoothed reward over last 100 episodes
        self.episodic_reward_avg.append(np.mean(self.episodic_reward[-100:]))
        print("*** %s E%05d S%010d R%s ***"%(self.phase, self.episode_number, self.step_number, self.episodic_reward[-1]))
        self.episode_number += 1

    def start_episode(self):
        self.start_time = time.time()
        self.random_state.shuffle(self.heads)
        self.active_head = self.heads[0]

        self.episode_losses = []
        self.episode_actions = []
        self.episode_rewards = []
        self.start_step_number = deepcopy(self.step_number)

        # restart counters
        self.terminal = False
        self.life_lost = True
        self.episode_reward = 0

        self.state = self.env.reset()
        return self.state

    def plot_current_episode(self, plot_basepath):
        plot_dict = {
                     'mean loss':self.episode_losses,
                     'actions':self.episode_actions,
                     'rewards':self.episode_rewards, }
        suptitle = 'E%s S%s-%s R%s'%(self.episode_number, self.start_step_number,
                                            self.end_step_number, self.episodic_reward[-1])
        plot_path = plot_basepath+'_ep%06d.png'%self.episode_number
        #step_range = np.arange(self.start_step_number, self.end_step_number)
        #self.plot_data(plot_path, plot_dict, suptitle, xname='episode steps', xdata=step_range)
        self.plot_data(plot_path, plot_dict, suptitle, xname='episode steps')#, xdata=step_range)

    def plot_progress(self, plot_basepath):
        det_plot_dict = {
            'episodic step count':self.episodic_step_count,
            'episodic time':self.episodic_times,
            'mean episodic loss':self.episodic_loss,
             }

        suptitle = 'Details E%s S%s'%(self.episode_number, self.end_step_number)
        edet_plot_path = plot_basepath+'_details_episodes.png'
        sdet_plot_path = plot_basepath+'_details_episodes.png'
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

    def handle_plotting(self, plot_basepath=''):
        # will plot at beginning of episode
        if plot_basepath == '':
            plot_basepath = self.ch.get_checkpoint_basepath(self.step_number)+'_%s'%self.phase
        #if not self.episode_number % self.ch.cfg['PLOT']['plot_episode_every_%s_episodes'%self.phase]:
        # dont plot first episode
        if self.episode_number:
            if not self.episode_number % self.ch.cfg['PLOT']['plot_episode_every_%s_episodes'%self.phase]:
                self.plot_current_episode(plot_basepath)
            if not self.episode_number % self.ch.cfg['PLOT']['plot_every_%s_episodes'%self.phase]:
                self.plot_progress(plot_basepath)

    def step(self, action):
        next_state, reward, self.life_lost, self.terminal = self.env.step(action)
        self.memory_buffer.add_experience(action=action,
                                            frame=next_state[-1],
                                            reward=1+np.sign(reward),
                                            terminal=self.life_lost,
                                            )
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.step_number+=1
        self.state = next_state
        self.last_state = self.state

    def set_eps(self):
        # TODO function to find eps - for now use constant
        if self.phase == 'train':
            self.eps = self.ch.cfg['DQN']['eps_init']
        else:
            self.eps = self.ch.cfg['EVAL']['eps_eval']

    def is_random_action(self):
        self.set_eps()
        r = self.random_state.rand()
        if r < self.eps:
            return True, self.random_state.randint(0, self.ch.num_actions-1)
        else:
            return False, -1


