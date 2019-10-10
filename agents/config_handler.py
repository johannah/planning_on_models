import os
import time
import numpy as np
import configparser
from glob import glob

from env import Environment
from replay import ReplayMemory

from IPython import embed
"""
TODO - create error checking for all of the variables in config
"""

def collect_random_experience(seed, env, replay_memory, num_random_steps):
    # note that since we are making the env different here
    # we should always use a different env for the random portion vs the
    # learning agent
    # create new replay memory
    print("starting random experience collection for %s steps"%num_random_steps)
    step_number = 0
    epoch_num = 0
    random_state = np.random.RandomState(seed)
    heads = np.arange(replay_memory.num_heads)
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
            action = random_state.randint(0, env.num_actions)
            next_state, reward, life_lost, terminal = env.step(action)
            # TODO - dead as end? should be from ini file or is it handled
            # in env?
            # Store transition in the replay memory
            replay_memory.add_experience(action=action,
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
    return replay_memory

class ConfigHandler():
    """
    class to wrap configs and setup the housekeeping materials needed to run an experiment
    """
    def __init__(self, config_file, restart_last_run=False, restart_run=''):

        self.output_base = '../../'
        self.model_savedir = 'model_savedir'
        self.random_buffer_dir = os.path.join(self.model_savedir, 'random_buffers')
        self.start_time = time.time()

        self._load_config(config_file)
        self._get_output_name(restart_last_run, restart_run)
        self._find_dependent_constants()


    def _get_output_name(self, restart_last_run, restart_run):
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

    def get_buffer_path(self, phase, num_train_steps):
        # always use training steps for reference
        assert phase in ['train', 'eval']
        game = os.path.split(self.cfg['ENV']['game'])[1].split('.')[0]
        seed = self.cfg['RUN']['%s_seed'%phase]
        buffer_size = self.cfg['RUN']['%s_buffer_size'%phase]
        filename = '%s_B%06dS%06d_N%010d_%s.npz' %(game, buffer_size,  seed, num_train_steps, phase)
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

    def create_empty_replay_memory(self, seed, buffer_size):
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

    def load_replay_memory(self, phase, ):
        """
         phase: string should be "train" or "eval" to indicate which memory buffer to load

         function will load latest experience in the model_savedir/name or create a random replay buffer of specified size to start from
        """
        assert phase in ['train', 'eval']
        buffer_size = self.cfg['RUN']['%s_buffer_size'%phase]
        seed = self.cfg['RUN']['%s_seed'%phase]
        init_empty_with_random=self.cfg['RUN']['load_random_%s_buffer'%phase]
        num_random_steps = self.cfg['RUN']['num_pure_random_steps_%s'%phase]

        buffer_path = self.search_for_latest_replay_buffer(phase)
        if buffer_path != "":
            print("loading buffer from past experience:%s"%buffer_path)
            return ReplayMemory(load_file=buffer_path)
        if not init_empty_with_random:
            # no buffer file was found, and we want an empty buffer
            print("creating empty replay buffer")
            return self.create_empty_replay_memory(seed, buffer_size)

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
            random_replay_memory = self.create_empty_replay_memory(seed, buffer_size)

            env = self.create_environment(seed)
            random_replay_memory = collect_random_experience(seed, env, random_replay_memory, num_random_steps=num_random_steps)
            # save the random buffer
            random_replay_memory.save_buffer(random_buffer_path)
            return random_replay_memory

class StateManager():
    def __init__(self, phase, step_number, episode_number):
        self.phase = phase
        self.step_number = step_number
        self.episode_number = episode_number

    def end_episode(self):

    def start_episode(self):
        self.start_time = time.time()
        self.episode_losses = []
        self.episode_actions = []
        self.episode_rewards = []


   #info['args'] = args
    #info['load_time'] = datetime.date.today().ctime()
    #info['num_rewards'] = len(info['REWARD_SPACE'])
    #    replay_memory = ReplayMemory(load_file=info['REPLAY_MEMORY_LOADPATH'])
    #    start_step_number = replay_memory.count
    #if info['REPLAY_MEMORY_LOADPATH'] == "":
    #    valid_replay_memory = ReplayMemory(action_space=env.action_space,
    #                             size=int(info['BUFFER_SIZE']*.1),
    #                             frame_height=info['OBS_SIZE'][0],
    #                             frame_width=info['OBS_SIZE'][1],
    #                             agent_history_length=info['HISTORY_SIZE'],
    #                             batch_size=info['BATCH_SIZE'],
    #                             num_heads=info['N_ENSEMBLE'],
    #                             bernoulli_probability=info['BERNOULLI_PROBABILITY'],
    #                             #latent_frame_height=info['LATENT_SIZE'],
    #                             #latent_frame_width=info['LATENT_SIZE'])
    #                              )

    #else:
    #    valid_replay_memory = ReplayMemory(load_file=info['REPLAY_MEMORY_VALID_LOADPATH'])
    #random_state = np.random.RandomState(info["SEED"])

    #if args.model_loadpath != '':
    #    # load data from loadpath - save model load for later. we need some of
    #    # these parameters to setup other things
    #    print('loading model from: %s' %args.model_loadpath)
    #    model_dict = torch.load(args.model_loadpath)
    #    info = model_dict['info']
    #    info['DEVICE'] = device
    #    # set a new random seed
    #    info["SEED"] = model_dict['cnt']
    #    model_base_filedir = os.path.split(args.model_loadpath)[0]
    #    start_step_number = start_last_save = model_dict['cnt']
    #    info['loaded_from'] = args.model_loadpath
    #    perf = model_dict['perf']
    #    start_step_number = perf['steps'][-1]
    #else:
    #    # create new project
    #    perf = {'steps':[],
    #            'avg_rewards':[],
    #            'episode_step':[],
    #            'episode_head':[],
    #            'eps_list':[],
    #            'episode_loss':[],
    #            'episode_reward':[],
    #            'episode_times':[],
    #            'episode_relative_times':[],
    #            'eval_rewards':[],
    #            'eval_steps':[],
    #            'head_rewards':[[] for x in range(info['N_ENSEMBLE'])],
    #            }

    #    start_last_save = 0
    #    # make new directory for this run in the case that there is already a
    #    # project with this name
    #    run_num = 0
    #    model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
    #    while os.path.exists(model_base_filedir):
    #        run_num +=1
    #        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
    #    os.makedirs(model_base_filedir)
    #    print("----------------------------------------------")
    #    print("starting NEW project: %s"%model_base_filedir)

    #model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    #write_info_file(info, model_base_filepath, start_step_number)
    #heads = list(range(info['N_ENSEMBLE']))
    #seed_everything(info["SEED"])


    #info['model_base_filepath'] = model_base_filepath
    #info['model_base_filedir'] = model_base_filedir
    #info['num_actions'] = env.num_actions
    #info['action_space'] = range(info['num_actions'])

    ##vqenv = VQEnv(info, vq_model_loadpath=info['VQ_MODEL_LOADPATH'], device='cpu')
    ##vq_model_dict = torch.load(info['VQ_MODEL_LOADPATH'], map_location=lambda storage, loc: storage)
    #############################################VQ##################################
    #info, vqvae_model = init_vq_model(info)

    #############################################VQ##################################

    #policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
    #                                  n_actions=env.num_actions,
    #                                  reshape_size=info['RESHAPE_SIZE'],
    #                                  #num_channels=info['HISTORY_SIZE'],
    #                                  num_channels=info['NUM_Z'],
    #                                  dueling=info['DUELING'],
    #                                  num_clusters=info['NUM_K'],
    #                                  use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])
    #target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
    #                                  n_actions=env.num_actions,
    #                                  reshape_size=info['RESHAPE_SIZE'],
    #                                  #num_channels=info['HISTORY_SIZE'],
    #                                  num_channels=info['NUM_Z'],
    #                                  dueling=info['DUELING'],
    #                                  num_clusters=info['NUM_K'],
    #                                  use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])
    #if info['PRIOR']:
    #    prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
    #                            n_actions=env.num_actions,
    #                            reshape_size=info['RESHAPE_SIZE'],
    #                            #num_channels=info['HISTORY_SIZE'],
    #                            num_channels=info['NUM_Z'],
    #                            dueling=info['DUELING'],
    #                            num_clusters=info['NUM_K'],
    #                            use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])

    #    print("using randomized prior")
    #    policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
    #    target_net = NetWithPrior(target_net, prior_net, info['PRIOR_SCALE'])

    #target_net.load_state_dict(policy_net.state_dict())
    ## create optimizer
    ##opt = optim.RMSprop(policy_net.parameters(),
    ##                    lr=info["RMS_LEARNING_RATE"],
    ##                    momentum=info["RMS_MOMENTUM"],
    ##                    eps=info["RMS_EPSILON"],
    ##                    centered=info["RMS_CENTERED"],
    ##                    alpha=info["RMS_DECAY"])

    #parameters = list(policy_net.parameters())+list(vqvae_model.parameters())
    #opt = optim.Adam(parameters, lr=info['ADAM_LEARNING_RATE'])
    #if args.model_loadpath is not '':
    #    # what about random states - they will be wrong now???
    #    # TODO - what about target net update cnt
    #    target_net.load_state_dict(model_dict['target_net_state_dict'])
    #    policy_net.load_state_dict(model_dict['policy_net_state_dict'])
    #    opt.load_state_dict(model_dict['optimizer'])
    #    print("loaded model state_dicts")
    #    if args.buffer_loadpath == '':
    #        args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.npz')
    #        print("auto loading buffer from:%s" %args.buffer_loadpath)
    #        try:
    #            replay_memory.load_buffer(args.buffer_loadpath)
    #        except Exception as e:
    #            print(e)
    #            print('not able to load from buffer: %s. exit() to continue with empty buffer' %args.buffer_loadpath)


