import os
import time
import configparser
from IPython import embed

class ConfigHandler():
    def __init__(self, config_file):
        self.config_file = config_file
        self.start_time = time.time()
        self.random_buffer_dir = 'random_buffers'
        if not os.path.exists(self.random_buffer_dir):
            os.makedirs(self.random_buffer_dir)
        cfg = configparser.ConfigParser()
        cfg.read(self.config_file)
        self.cfg = dict(cfg)
        for k in cfg.keys():
            self.cfg[k] = dict(cfg[k])
        self.make_correct_types()
        self._find_dependent_constants()

    def make_correct_types(self):
        # all of the values are read as strings
        for section in self.cfg.keys():
            for key in self.cfg[section].keys():
                try:
                    val = self.cfg[section][key]
                    if '.' in val:
                        self.cfg[section][key] = float(val)
                    else:
                        self.cfg[section][key] = int(val)
                except Exception:
                    if ',' in val:
                        list_val = [int(x) for x in self.cfg['ENV']['action_space'][1:-1].split(',')]
                        self.cfg[section][key] = list_val

        # todo handle list

    def _find_dependent_constants(self):
        self.cfg['ENV']['max_steps'] = self.cfg['ENV']['max_frames']/self.cfg['ENV']['frame_skip']
        self.cfg['PLOT']['fake_acts']= [self.cfg['PLOT']['random_head']for x in range(int(self.cfg['DQN']['n_ensemble']))]
        self.cfg['ENV']['num_rewards'] = len(self.cfg['ENV']['reward_space'])


    def get_default_random_buffer_name(self, phase='train'):
        assert phase in ['train', 'eval']
        game = os.path.split(self.cfg['ENV']['game'])[1].split('.')[0]
        seed = self.cfg['RUN']['%s_seed'%phase]
        n = self.cfg['RUN']['min_steps_to_learn']
        filename = '%s_S%06d_N%s_%s.npz' %(game, seed, n, phase)
        filepath = os.path.join(self.random_buffer_dir, filename)
        base_config = os.path.split(self.config_file)[1].replace('.npz', '_')
        out_config = os.path.join(self.random_buffer_dir, (filename + base_config))
        cmd = 'cp %s %s' %(self.config_file, out_config)
        os.system(cmd)
        # TODO load the comparison
        return filepath





        #self.start_step_number = 0

    #info['args'] = args
    #info['load_time'] = datetime.date.today().ctime()
    #info['num_rewards'] = len(info['REWARD_SPACE'])

    ## create environment
    ## create replay buffer
    #if info['REPLAY_MEMORY_LOADPATH'] == "":
    #    replay_memory = ReplayMemory(action_space=env.action_space,
    #                             size=info['BUFFER_SIZE'],
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


