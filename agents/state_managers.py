import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
import sys
sys.path.append('../models')
from IPython import embed
import config
import torch
import torch.optim as optim
import time
from datasets import AtariDataset
from forward_conv import BasicBlock, ForwardResNet
from ae_utils import sample_from_discretized_mix_logistic, reshape_input
from train_atari_vqvae_diff_action_reward import run as vqvae_run
from vqvae import VQVAErl
from create_reward_dataset import find_episodic_rewards, make_dataset
from train_atari_vqvae_diff_action import train_vqvae
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_
from forward_conv import ForwardResNet, BasicBlock
from torchvision.utils import save_image
from ae_utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from IPython import embed
from lstm_utils import plot_dict_losses
from ae_utils import save_checkpoint
torch.set_num_threads(4)

class VQEnv(object):
    def __init__(self, vq_info,  seed=393, vq_model_loadpath='', forward_model_loadpath='', device='cpu'):
        # env will be deepcopied version of true state
        self.DEVICE = device
        self.num_samples = vq_info['NUM_SAMPLES']
        self.vq_info = vq_info
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        if vq_model_loadpath == '':
            self.init_models()
        else:
            self.load_vq_model(vq_model_loadpath)

    def load_vq_model(self, vq_model_loadpath):
        self.vq_model_loadpath = vq_model_loadpath
        self.vq_model_dict = torch.load(self.vq_model_loadpath, map_location=lambda storage, loc: storage)
        try:
            self.vq_info = self.vq_model_dict['vq_info']
        except:
            self.vq_info = self.vq_model_dict['info']
            old_keys = list(self.vq_info.keys())
            for a in old_keys:
                if 'list' not in a:
                    self.vq_info[a.upper()] = self.vq_info[a]
            args = self.vq_info['args'][-1].__dict__
            for a in args.keys():
                self.vq_info[a.upper()] = args[a]

        self.vqvae_opt = self.vq_model_dict['vq_optimizer']
        self.vqvae_model = VQVAErl(num_clusters=self.vq_info['NUM_K'],
                            encoder_output_size=self.vq_info['NUM_Z'],
                            num_output_mixtures=self.vq_info['num_output_mixtures'],
                            in_channels_size=self.vq_info['NUMBER_CONDITION'],
                            n_actions=self.vq_info['num_actions'],
                            int_reward=self.vq_info['num_rewards'],
                            ).to(self.DEVICE)
        self.vqvae_model.load_state_dict(self.vq_model_dict['vqvae_state_dict'])
        print("loaded vqvae model")


    def create_forward_model(self, forward_model_loadpath):
        self.forward_model_loadpath = forward_model_loadpath
        self.forward_model = ForwardResNet(BasicBlock,
                                           data_width=self.vq_info['LATENT_SIZE'],
                                           num_channels=self.vq_info['num_channels'],
                                           num_output_channels=self.vq_info['NUM_K'],
                                           dropout_prob=self.vq_info['FORWARD_DROPOUT'],
                                           num_rewards=self.vq_info['num_rewards'],
                                           num_actions=self.vq_info['num_actions'])
        self.forward_opt = optim.Adam(self.forward_model.parameters(), lr=self.vq_info['FORWARD_LEARNING_RATE'])
        self.forward_train_cnt = 0

    def train_vq_model(self, train_buffer, valid_buffer):
        #self.vqvae_model, self.vq_opt = vqvae_run(self.vq_info, self.vqvae_model, self.vqvae_opt, train_buffer, valid_buffer, num_samples_to_train=1000000, save_every_samples=50000*5)
        #train_batch = train_buffer.get_minibatch(self.vq_info['VQ_BATCH_SIZE'])
        self.vqvae_model.train()
        self.vqvae_opt.zero_grad()
        #state_input, actions, rewards = make_state(batch, info['DEVICE'], info['NORM_BY'])
        #x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(state_input)
        #z_q_x.retain_grad()
        #rec_losses, rec_ests = find_rec_losses(alpha=info['ALPHA_REC'],
        #                             nr=info['NR_LOGISTIC_MIX'],
        #                             nmix=info['nmix'],
        #                             x_d=x_d, true=state_input,
        #                             DEVICE=info['DEVICE'])

        #loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
        #loss_reward = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
        #loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        #loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())
        #vqvae_model.embedding.zero_grad()

        #[rec_losses[x].backward(retain_graph=True) for x in range(info['num_channels'])]
        #loss_act.backward(retain_graph=True)
        #loss_reward.backward(retain_graph=True)
        #z_e_x.backward(z_q_x.grad, retain_graph=True)
        #loss_2.backward(retain_graph=True)
        #loss_3.backward()

        #parameters = list(vqvae_model.parameters())
        #clip_grad_value_(parameters, 5)
        #opt.step()
        #bs = float(x_d.shape[0])
        #avg_train_losses = [loss_reward.item()/bs, loss_act.item()/bs,
        #                    rec_losses[0].item()/bs, rec_losses[1].item()/bs,
        #                    rec_losses[2].item()/bs, rec_losses[3].item()/bs,
        #                    loss_2.item()/bs, loss_3.item()/bs]
        #opt.zero_grad()

    def init_vq_model(self):
        #data_dir = os.path.split(train_data_file)[0]
        data_dir = self.vq_info['model_base_filepath']
        run_num = 0
        model_base_filedir = os.path.join(data_dir, self.vq_info['VQ_SAVENAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(data_dir, self.vq_info['VQ_SAVENAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        model_base_filepath = os.path.join(model_base_filedir, self.vq_info['VQ_SAVENAME'])
        print("VQ MODEL BASE FILEPATH", model_base_filepath)
        self.vq_info['train_cnt'] = 0
        self.vq_info['vq_train_cnts'] = []
        self.vq_info['vq_train_losses_list'] = []
        self.vq_info['vq_valid_cnts'] = []
        self.vq_info['vq_valid_losses_list'] = []
        self.vq_info['vq_save_times'] = []
        self.vq_info['vq_last_save'] = 0
        self.vq_info['vq_last_plot'] = 0
        self.vq_info['vq_model_base_filedir'] = model_base_filedir
        self.vq_info['vq_model_base_filepath'] = model_base_filepath

        self.vq_info['num_channels'] = 2
        self.vq_info['num_output_mixtures']= (2*self.vq_info['NR_LOGISTIC_MIX']+self.vq_info['NR_LOGISTIC_MIX'])*self.vq_info['num_channels']
        nmix = int(self.vq_info['num_output_mixtures']/2)
        self.vq_info['nmix'] = nmix
        self.vqvae_model = VQVAErl(num_clusters=self.vq_info['NUM_K'],
                            encoder_output_size=self.vq_info['NUM_Z'],
                            num_output_mixtures=self.vq_info['num_output_mixtures'],
                            in_channels_size=self.vq_info['NUMBER_CONDITION'],
                            n_actions=self.vq_info['num_actions'],
                            int_rewards=self.vq_info['num_rewards'],
                            ).to(self.DEVICE)

        parameters = list(self.vqvae_model.parameters())
        self.opt = optim.Adam(parameters, lr=self.vq_info['VQ_LEARNING_RATE'])
        self.action_space = range(self.vq_info['num_actions'])

    def init_forward_model(self):
        #data_dir = os.path.split(train_data_file)[0]
        data_dir = self.forwrad_info['model_base_filepath']
        run_num = 0
        model_base_filedir = os.path.join(data_dir, self.vq_info['VQ_SAVENAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(data_dir, self.vq_info['VQ_SAVENAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        model_base_filepath = os.path.join(model_base_filedir, self.vq_info['VQ_SAVENAME'])
        print("VQ MODEL BASE FILEPATH", model_base_filepath)
        self.vq_info['train_cnt'] = 0
        self.vq_info['vq_train_cnts'] = []
        self.vq_info['vq_train_losses_list'] = []
        self.vq_info['vq_valid_cnts'] = []
        self.vq_info['vq_valid_losses_list'] = []
        self.vq_info['vq_save_times'] = []
        self.vq_info['vq_last_save'] = 0
        self.vq_info['vq_last_plot'] = 0
        self.vq_info['vq_model_base_filedir'] = model_base_filedir
        self.vq_info['vq_model_base_filepath'] = model_base_filepath

        self.vq_info['num_channels'] = 2
        self.vq_info['num_output_mixtures']= (2*self.vq_info['NR_LOGISTIC_MIX']+self.vq_info['NR_LOGISTIC_MIX'])*self.vq_info['num_channels']
        nmix = int(self.vq_info['num_output_mixtures']/2)
        self.vq_info['nmix'] = nmix
        self.vqvae_model = VQVAErl(num_clusters=self.vq_info['NUM_K'],
                            encoder_output_size=self.vq_info['NUM_Z'],
                            num_output_mixtures=self.vq_info['num_output_mixtures'],
                            in_channels_size=self.vq_info['NUMBER_CONDITION'],
                            n_actions=self.vq_info['num_actions'],
                            int_rewards=self.vq_info['num_rewards'],
                            ).to(self.DEVICE)

        parameters = list(self.vqvae_model.parameters())
        self.opt = optim.Adam(parameters, lr=self.vq_info['VQ_LEARNING_RATE'])
        self.action_space = range(self.vq_info['num_actions'])

    def decode_vq_from_latents(self, latents):
        latents = latents.long()
        N,H,W = latents.shape
        C = self.vq_info['NUM_Z']
        with torch.no_grad():
            x_d, z_q_x, actions, rewards = self.vqvae_model.decode_clusters(latents,N,H,W,C)
        # vqvae_model predicts the action that took this particular latent from t-1
        # to t-0
        # vqvae_model predcts the reward that was seen at t=0
        pred_actions = torch.argmax(actions, dim=1).cpu().numpy()
        pred_rewards = torch.argmax(rewards, dim=1).cpu().numpy()
        return x_d, pred_actions, pred_rewards

    def sample_mean_from_latents(self, x_d):
        # TODO
        nmix = 30
        rec_mest = torch.Tensor(x_d[:,:nmix])
        rec_mean = sample_from_discretized_mix_logistic(rec_mest, self.vq_info['NR_LOGISTIC_MIX'], only_mean=True)
        rec_mean = (((rec_mean+1)/2.0)).cpu().numpy()
        return rec_mean

    def sample_from_latents(self, x_d):
        # TODO
        nmix = 30
        num_samples = 40
        rec_mest = torch.Tensor(x_d[:,:nmix])
        if num_samples:
            rec_sams = np.zeros((x_d.shape[0], num_samples, 1, 80, 80))
            for n in range(num_samples):
                sam = sample_from_discretized_mix_logistic(rec_mest, self.vq_info['NR_LOGISTIC_MIX'], only_mean=False)
                rec_sams[:,n] = (((sam+1)/2.0)).cpu().numpy()
            rec_est = np.mean(rec_sams, axis=1)
        rec_mean = sample_from_discretized_mix_logistic(rec_mest, self.vq_info['NR_LOGISTIC_MIX'], only_mean=True)
        rec_mean = (((rec_mean+1)/2.0)).cpu().numpy()
        return rec_est, rec_mean

    def get_state_representation(self, state):
        # todo - transform from np to the right kind of torch array - need to
        x_d,_,z_q,latents,_,_ = self.get_vq_state(state/self.vq_info['NORM_BY'])
        #latent_state = torch.stack((latents[0][None,None], latents[1][None,None]), dim=0)
        return latents.float(), x_d

    def get_vq_state(self, states):
        # normalize and make 80x80
        s = (2*reshape_input(torch.FloatTensor(states).to(self.DEVICE))-1)
        # make sure s has None on 0th
        with torch.no_grad():
            x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = self.vqvae_model(s)
        return x_d.detach(), z_e_x.detach(), z_q_x.detach(), latents.detach(), pred_actions.detach(), pred_rewards.detach()

    def get_next_latent(self, latent_states, actions):
        # states should be last two states as np array
        # should state be normalized already when it comes in?
        # reset base channel actions
        #self.base_channel_actions *= 0
        for a in self.action_space:
            self.base_channel_actions[actions==a,a]=1
        tf_state_input = torch.cat((self.base_channel_actions,latent_states),dim=1)
        with torch.no_grad():
            pred_next_latent = self.conv_forward_model(tf_state_input)
        pred_next_latent = torch.argmax(pred_next_latent, dim=1)
        #x_d, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(pred_next_latent)
        x_d, z_q_x, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(pred_next_latent)
        # latent state consists of [latent_t-1, latent_t]
        next_latent_states = torch.cat((latent_states[:,1][:,None], pred_next_latent[:,None].float()), dim=1)
        return next_latent_states, x_d, pred_vq_actions, pred_vq_rewards

    def get_next_state(self, latent_state, action):
        next_latent_state, x_d, pred_action, pred_reward = self.get_next_latent(latent_state, action)
        return next_latent_state, pred_reward

    def plot_rollout(self, rollout_number, x_ds, latents, actions, rewards):
        rdir = os.path.join(self.agent_filepath, "R%06d"%rollout_number)
        os.makedirs(rdir)
        s_est,s_mean = self.sample_from_latents(x_ds)
        print('rollout', rollout_number)
        for i in range(actions.shape[0]-1):
            f,ax=plt.subplots(2,2)
            ax[0,0].imshow(s_est[i,0])
            ax[0,0].set_title('S%s-S%s'%(i,i+1))
            ax[0,1].set_title('A%s R%s'%(actions[i], rewards[i]))
            ax[0,1].imshow(s_est[i+1,0])
            ax[1,0].imshow(latents[i,1])
            ax[1,1].imshow(latents[i+1,1])
            plt.savefig(os.path.join(rdir, 'n%04d.png'%i))
            plt.close()

        cmd = 'convert %s %s' %(os.path.join(rdir, 'n*.png'), os.path.join(rdir, '_R%06d.gif'%rollout_number))
        os.system(cmd)

    def get_rollout_action_from_state(self, latent_state):
        action = self.random_state.choice(self.action_space)
        return action

    #def rollout_from_state(self, latent_state, forward_step, keep_traces=False):
    #    # TODO - if we predicted end of life - this should be changed
    #    total_rollout_reward = 0
    #    st = time.time()
    #    if keep_traces:
    #        actions = np.zeros(self.rollout_limit, dtype=np.int)
    #        latents = np.zeros((self.rollout_limit+1,2,10,10))
    #        x_ds = np.zeros((self.rollout_limit+1, 60, 80, 80))
    #        rewards = np.zeros(self.rollout_limit, dtype=np.int)
    #        x_d, pred_vq_actions, pred_vq_rewards = self.decode_vq_from_latents(latent_state[:,1])
    #        x_ds[0] = x_d
    #        latents[0] = latent_state.cpu().numpy()
    #    for i in range(self.rollout_limit):
    #        action = self.random_state.choice(self.action_space)
    #        next_latent_state, x_d, _, pred_reward = self.get_next_latent(latent_state, action)
    #        reward = self.gammas[i]*pred_reward
    #        total_rollout_reward += reward
    #        if keep_traces:
    #            latents[i] = next_latent_state.cpu().numpy()
    #            actions[i] = action
    #            rewards[i] = reward
    #            x_ds[i+1] = x_d
    #            latents[i+1] = next_latent_state.cpu().numpy()
    #        latent_state = next_latent_state
    #        forward_step+=1
    #    if keep_traces:
    #        self.plot_rollout(self.rollout_number, x_ds, latents, actions, rewards)
    #    self.rollout_number+=1
    #    et = time.time()
    #    print("rollout took", et-st, total_rollout_reward)
    #    return total_rollout_reward

    def get_valid_actions(self, state):
        return self.action_space

    def is_finished(self):
        return False






