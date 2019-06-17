
#class VQRolloutStateManager(object):
#    def __init__(self, forward_model_loadpath, agent_filepath, n_playout=50,
#                 DEVICE='cpu', num_samples=40, seed=393):
#        # env will be deepcopied version of true state
#        self.n_playout=n_playout
#        self.DEVICE = DEVICE
#        self.num_samples = num_samples
#        self.seed = seed
#        self.random_state = np.random.RandomState(self.seed)
#        self.agent_filepath = agent_filepath
#        self.load_models(forward_model_loadpath)
#        self.rollout_number = 0
#        #self.gamma = gamma
#        #self.gammas = [self.gamma**i for i in range(self.rollout_limit)]
#
#    def load_models(self, forward_model_loadpath):
#        self.forward_model_loadpath = forward_model_loadpath
#        self.forward_model_dict = torch.load(self.forward_model_loadpath,
#                                             map_location=lambda storage,
#                                             loc: storage)
#        self.forward_info = self.forward_model_dict['info']
#        self.forward_largs = self.forward_info['args'][-1]
#        self.vq_model_loadpath = self.forward_largs.train_data_file.replace('_train_forward.npz', '.pt')
#        self.vq_model_dict = torch.load(self.vq_model_loadpath,
#                                           map_location=lambda storage, loc: storage)
#
#        self.vq_info = self.vq_model_dict['info']
#        self.vq_largs = self.vq_info['args'][-1]
#        self.vqvae_model = VQVAE(num_clusters=self.vq_largs.num_k,
#                                 encoder_output_size=self.vq_largs.num_z,
#                                 num_output_mixtures=self.vq_info['num_output_mixtures'],
#                                 in_channels_size=self.vq_largs.number_condition,
#                                 n_actions=self.vq_info['num_actions'],
#                                 int_reward=self.vq_info['num_rewards'])
#        # load vq mod
#        print("loading vq model:%s"%self.vq_model_loadpath)
#        self.vqvae_model.load_state_dict(self.vq_model_dict['vqvae_state_dict'])
#        self.conv_forward_model = ForwardResNet(BasicBlock, data_width=self.forward_info['hsize'],
#                                           num_channels=self.forward_info['num_channels'],
#                                           num_output_channels=self.vq_largs.num_k,
#                                           dropout_prob=0.0)
#        self.conv_forward_model.load_state_dict(self.forward_model_dict['conv_forward_model'])
#        # base_channel_actions used when we take one action at a time
#        self.base_channel_actions = torch.zeros((self.n_playout, self.forward_info['num_actions'], self.forward_info['hsize'], self.forward_info['hsize']))
#        self.action_space = range(self.vq_info['num_actions'])
#
#    def decode_vq_from_latents(self, latents):
#        latents = latents.long()
#        N,H,W = latents.shape
#        C = self.vq_largs.num_z
#        with torch.no_grad():
#            x_d, z_q_x, actions, rewards = self.vqvae_model.decode_clusters(latents,N,H,W,C)
#        # vqvae_model predicts the action that took this particular latent from t-1
#        # to t-0
#        # vqvae_model predcts the reward that was seen at t=0
#        pred_actions = torch.argmax(actions, dim=1).cpu().numpy()
#        pred_rewards = torch.argmax(rewards, dim=1).cpu().numpy()
#        return x_d, pred_actions, pred_rewards
#
#    def sample_from_latents(self, x_d):
#        # TODO
#        nmix = 30
#        rec_mest = torch.Tensor(x_d[:,:nmix])
#        if self.num_samples:
#            rec_sams = np.zeros((x_d.shape[0], self.num_samples, 1, 80, 80))
#            for n in range(self.num_samples):
#                sam = sample_from_discretized_mix_logistic(rec_mest, self.vq_largs.nr_logistic_mix, only_mean=False)
#                rec_sams[:,n] = (((sam+1)/2.0)).cpu().numpy()
#            rec_est = np.mean(rec_sams, axis=1)
#        rec_mean = sample_from_discretized_mix_logistic(rec_mest, self.vq_largs.nr_logistic_mix, only_mean=True)
#        rec_mean = (((rec_mean+1)/2.0)).cpu().numpy()
#        return rec_est, rec_mean
#
#    def get_state_representation(self, state):
#        # todo - transform from np to the right kind of torch array - need to
#        x_d,_,_,latents,_,_ = self.get_vq_state(state/self.vq_info['NORM_BY'])
#        #latent_state = torch.stack((latents[0][None,None], latents[1][None,None]), dim=0)
#        return latents.float(), x_d
#
#    def get_vq_state(self, states):
#        # normalize and make 80x80
#        s = (2*reshape_input(torch.FloatTensor(states).to(self.DEVICE))-1)
#        # make sure s has None on 0th
#        with torch.no_grad():
#            x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals = self.vqvae_model(s)
#        return x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals
#
#    def get_next_latent(self, latent_states, actions):
#        # states should be last two states as np array
#        # should state be normalized already when it comes in?
#        # reset base channel actions
#        #self.base_channel_actions *= 0
#        for a in self.action_space:
#            self.base_channel_actions[actions==a,a]=1
#        tf_state_input = torch.cat((self.base_channel_actions,latent_states),dim=1)
#        with torch.no_grad():
#            pred_next_latent = self.conv_forward_model(tf_state_input)
#        pred_next_latent = torch.argmax(pred_next_latent, dim=1)
#        x_d, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(pred_next_latent)
#        # latent state consists of [latent_t-1, latent_t]
#        next_latent_states = torch.cat((latent_states[:,1][:,None], pred_next_latent[:,None].float()), dim=1)
#        return next_latent_states, x_d, pred_vq_actions, pred_vq_rewards
#
#    def get_next_state(self, latent_state, action):
#        next_latent_state, x_d, pred_action, pred_reward = self.get_next_latent(latent_state, action)
#        return next_latent_state, pred_reward
#
#    def plot_rollout(self, rollout_number, x_ds, latents, actions, rewards):
#        rdir = os.path.join(self.agent_filepath, "R%06d"%rollout_number)
#        os.makedirs(rdir)
#        s_est,s_mean = self.sample_from_latents(x_ds)
#        print('rollout', rollout_number)
#        for i in range(actions.shape[0]-1):
#            f,ax=plt.subplots(2,2)
#            ax[0,0].imshow(s_est[i,0])
#            ax[0,0].set_title('S%s-S%s'%(i,i+1))
#            ax[0,1].set_title('A%s R%s'%(actions[i], rewards[i]))
#            ax[0,1].imshow(s_est[i+1,0])
#            ax[1,0].imshow(latents[i,1])
#            ax[1,1].imshow(latents[i+1,1])
#            plt.savefig(os.path.join(rdir, 'n%04d.png'%i))
#            plt.close()
#
#        cmd = 'convert %s %s' %(os.path.join(rdir, 'n*.png'), os.path.join(rdir, '_R%06d.gif'%rollout_number))
#        os.system(cmd)
#
#    def get_rollout_action_from_state(self, latent_state):
#        action = self.random_state.choice(self.action_space)
#        return action
#
#    #def rollout_from_state(self, latent_state, forward_step, keep_traces=False):
#    #    # TODO - if we predicted end of life - this should be changed
#    #    total_rollout_reward = 0
#    #    st = time.time()
#    #    if keep_traces:
#    #        actions = np.zeros(self.rollout_limit, dtype=np.int)
#    #        latents = np.zeros((self.rollout_limit+1,2,10,10))
#    #        x_ds = np.zeros((self.rollout_limit+1, 60, 80, 80))
#    #        rewards = np.zeros(self.rollout_limit, dtype=np.int)
#    #        x_d, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(latent_state[:,1])
#    #        x_ds[0] = x_d
#    #        latents[0] = latent_state.cpu().numpy()
#    #    for i in range(self.rollout_limit):
#    #        action = self.random_state.choice(self.action_space)
#    #        next_latent_state, x_d, _, pred_reward = self.get_next_latent(latent_state, action)
#    #        reward = self.gammas[i]*pred_reward
#    #        total_rollout_reward += reward
#    #        if keep_traces:
#    #            latents[i] = next_latent_state.cpu().numpy()
#    #            actions[i] = action
#    #            rewards[i] = reward
#    #            x_ds[i+1] = x_d
#    #            latents[i+1] = next_latent_state.cpu().numpy()
#    #        latent_state = next_latent_state
#    #        forward_step+=1
#    #    if keep_traces:
#    #        self.plot_rollout(self.rollout_number, x_ds, latents, actions, rewards)
#    #    self.rollout_number+=1
#    #    et = time.time()
#    #    print("rollout took", et-st, total_rollout_reward)
#    #    return total_rollout_reward
#
#    def get_valid_actions(self, state):
#        return self.action_space
#
#    def is_finished(self):
#        return False


