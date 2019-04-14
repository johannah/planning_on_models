import numpy as np
from copy import deepcopy
import os
import sys
sys.path.append('../models')
from IPython import embed
import config
import torch

from forward_conv import BasicBlock, ForwardResNet
from ae_utils import sample_from_discretized_mix_logistic
from train_atari_action_vqvae import reshape_input
from vqvae import VQVAE

class AtariStateManager(object):
    def __init__(self, env, seed=393):
        # env will be deepcopied version of true state
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.env = deepcopy(env)

    def get_state(self):
        return self.env.ale.getRAM()

    def get_next_state(self, state, action):
        next_state = np.zeros_like(state)
        reward = -9
        return next_state, reward

    def get_valid_actions(self, state):
        return self.env.action_space

    def is_finished(self):
        return self.env.finished

class VQRolloutStateManager(object):
    def __init__(self, forward_model_loadpath, rollout_limit=10, DEVICE='cpu', num_samples=40, seed=393):
        # env will be deepcopied version of true state
        self.DEVICE = DEVICE
        self.rollout_limit = rollout_limit
        self.num_samples = num_samples
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.load_models(forward_model_loadpath)

    def load_models(self, forward_model_loadpath):
        self.forward_model_loadpath = forward_model_loadpath
        self.forward_model_dict = torch.load(self.forward_model_loadpath,
                                             map_location=lambda storage,
                                             loc: storage)
        self.forward_info = self.forward_model_dict['info']
        self.forward_largs = self.forward_info['args'][-1]
        self.vq_model_loadpath = self.forward_largs.train_data_file.replace('_train_forward.npz', '.pt')
        self.vq_model_dict = torch.load(self.vq_model_loadpath,
                                           map_location=lambda storage, loc: storage)

        self.vq_info = self.vq_model_dict['info']
        self.vq_largs = self.vq_info['args'][-1]
        self.vqvae_model = VQVAE(num_clusters=self.vq_largs.num_k,
                                 encoder_output_size=self.vq_largs.num_z,
                                 num_output_mixtures=self.vq_info['num_output_mixtures'],
                                 in_channels_size=self.vq_largs.number_condition,
                                 n_actions=self.vq_info['num_actions'],
                                 int_reward=self.vq_info['num_rewards'])
        # load vq mod
        print("loading vq model:%s"%self.vq_model_loadpath)
        self.vqvae_model.load_state_dict(self.vq_model_dict['vqvae_state_dict'])
        self.conv_forward_model = ForwardResNet(BasicBlock, data_width=self.forward_info['hsize'],
                                           num_channels=self.forward_info['num_channels'],
                                           num_output_channels=self.vq_largs.num_k,
                                           dropout_prob=0.0)
        self.conv_forward_model.load_state_dict(self.forward_model_dict['conv_forward_model'])
        # base_channel_actions used when we take one action at a time
        self.base_channel_actions = torch.zeros((1, self.forward_info['num_actions'], self.forward_info['hsize'], self.forward_info['hsize']))
        self.action_space = range(self.vq_info['num_actions'])

    def decode_vq_from_latents(self, latents):
        latents = latents.long()
        N,H,W = latents.shape
        C = self.vq_largs.num_z
        with torch.no_grad():
            x_d, z_q_x, actions, rewards = self.vqvae_model.decode_clusters(latents,N,H,W,C)
        # vqvae_model predicts the action that took this particular latent from t-1
        # to t-0
        # vqvae_model predcts the reward that was seen at t=0
        pred_actions = torch.argmax(actions, dim=1).cpu().numpy()
        pred_rewards = torch.argmax(rewards, dim=1).cpu().numpy()
        return x_d, pred_actions, pred_rewards

    def sample_from_latents(self, x_d):
        # TODO
        nmix = 30
        rec_mest = x_d[:,:nmix].detach()
        if self.num_samples:
            rec_sams = np.zeros((x_d.shape[0], self.num_samples, 1, 80, 80))
            for n in range(self.num_samples):
                sam = sample_from_discretized_mix_logistic(rec_mest, self.vq_largs.nr_logistic_mix, only_mean=False)
                rec_sams[:,n] = (((sam+1)/2.0)).cpu().numpy()
            rec_est = np.mean(rec_sams, axis=1)
        rec_mean = sample_from_discretized_mix_logistic(rec_mest, self.vq_largs.nr_logistic_mix, only_mean=True)
        rec_mean = (((rec_mean+1)/2.0)).cpu().numpy()
        return rec_est, rec_mean

    def get_state_representation(self, state):
        # todo - transform from np to the right kind of torch array - need to
        _,_,_,latents,_,r = self.get_vq_state(state/255.0)
        #latent_state = torch.stack((latents[0][None,None], latents[1][None,None]), dim=0)
        return latents.float()

    def get_vq_state(self, states):
        # normalize and make 80x80
        s = (2*reshape_input(torch.FloatTensor(states))-1)
        # make sure s has None on 0th
        with torch.no_grad():
            x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals = self.vqvae_model(s)
        return x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals

    def get_next_latent(self, latent_state, action):
        # states should be last two states as np array
        # should state be normalized already when it comes in?
        # reset base channel actions
        self.base_channel_actions *= 0
        self.base_channel_actions[0,action] = 1.0
        tf_state_input = torch.cat((self.base_channel_actions,
                                    latent_state),dim=1)
        with torch.no_grad():
            pred_next_latent = self.conv_forward_model(tf_state_input)
        pred_next_latent = torch.argmax(pred_next_latent, dim=1)
        x_d, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(pred_next_latent)
        # latent state consists of [latent_t-1, latent_t]
        next_latent_state = torch.cat((latent_state[0,1][None,None], pred_next_latent[None].float()), dim=1)
        return next_latent_state, x_d, pred_vq_actions[0], pred_vq_rewards[0]

    def get_next_state(self, latent_state, action):
        next_latent_state, x_d, pred_action, pred_reward = self.get_next_latent(latent_state, action)
        return next_latent_state

    def rollout_from_state(self, latent_state, keep_traces=False):
        # TODO - if we predicted end of life - this should be changed
        total_rollout_reward = 0
        actions = np.zeros(self.rollout_limit)
        latents = np.zeros((self.rollout_limit+1,2,10,10))
        latents[0] = latent_state.cpu().numpy()
        rewards = np.zeros(self.rollout_limit)
        for i in range(self.rollout_limit):
            #action = self.random_state.choice(self.action_space)
            action = 1# self.random_state.choice(self.action_space)
            next_latent_state, _, _, pred_reward = self.get_next_latent(latent_state, action)
            total_rollout_reward += pred_reward
            if keep_traces:
                latents[i+1] = next_latent_state.cpu().numpy()
                actions[i] = action
                rewards[i] = pred_reward
        return total_rollout_reward

    def get_valid_actions(self, state):
        return self.action_space

    def is_finished(self):
        return False




