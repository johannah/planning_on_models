
import math
import time
import numpy as np
from IPython import embed
from copy import deepcopy
import logging
import os
import subprocess
import sys
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from imageio import imwrite

import gym
from models.vqvae import AutoEncoder
from models.pixel_cnn import GatedPixelCNN
from skimage.morphology import  disk
from skimage.filters.rank import  median
from models.datasets import prepare_img, undo_img_scaling, chicken_color, input_ysize, input_xsize, base_chicken
min_pixel, max_pixel = 0., 255.
from models.utils import discretized_mix_logistic_loss, get_cuts, to_scalar
from models.utils import sample_from_discretized_mix_logistic
from planning.mcts import PTreeNode

class VQPCNN_model():
    def __init__(self, DEVICE, model_savedir, load_vq_name, load_pcnn_name='NA',
                   dsize=80, nr_logistic_mix=10,
                   num_z=64, num_clusters=512,
                   N_LAYERS=10, DIM=256, history_size=4,
                   vq_num_samples=1,pcnn_num_samples=1,
                   ):
        self.DEVICE = DEVICE

        self.vq_num_samples = vq_num_samples
        self.num_z = num_z
        self.nr_logistic_mix = nr_logistic_mix
        self.num_clusters = num_clusters
        self.DIM = DIM
        self.vq_num_examples = 0
        self.vq_num_samples = 0

        self.pcnn_num_samples = pcnn_num_samples
        self.history_size = history_size
        self.cond_size = self.history_size*self.DIM
        self.N_LAYERS = N_LAYERS
        self.probs_size = (2*self.nr_logistic_mix)+self.nr_logistic_mix
        self.pcnn_num_examples = 0
        self.pcnn_num_samples = 0

        # attempt to load model which exists
        if load_vq_name != 'NA':
            vq_model_loadpath = os.path.join(model_savedir, load_vq_name)
        if load_pcnn_name != 'NA':
            pcnn_model_loadpath = os.path.join(model_savedir, load_pcnn_name)
        self.load_vq_model(vq_model_loadpath)
        self.load_pcnn_model(pcnn_model_loadpath)


    def load_vq_model(self, vq_model_loadpath):
        if os.path.exists(vq_model_loadpath):
            self.vq_model = AutoEncoder(nr_logistic_mix=self.nr_logistic_mix, num_clusters=self.num_clusters, encoder_output_size=self.num_z).to(self.DEVICE)
            vq_model_dict = torch.load(vq_model_loadpath, map_location=lambda storage, loc: storage)
            self.vq_model.load_state_dict(vq_model_dict['state_dict'])
            # TODO - this isnt quite right - this number is epoch
            self.vq_num_examples = vq_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(self.vq_num_examples,vq_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(vq_model_loadpath))
            # TODO create vq_model dict and epochs and all
            sys.exit()

    def load_pcnn_model(self, pcnn_model_loadpath):
        if os.path.exists(pcnn_model_loadpath):
            self.pcnn_model = GatedPixelCNN(self.num_clusters, self.DIM, self.N_LAYERS,
                                       self.history_size, spatial_cond_size=self.cond_size).to(self.DEVICE)
            pcnn_model_dict = torch.load(pcnn_model_loadpath, map_location=lambda storage, loc: storage)
            self.pcnn_model.load_state_dict(pcnn_model_dict['state_dict'])
            self.pcnn_num_examples = pcnn_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(self.pcnn_num_examples,pcnn_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(pcnn_model_loadpath))
            sys.exit()

    def encode_obs(self, obs_states):
        # normalize data before putting into vqvae
        # transofrms from range 0,255 to range 0.0 to 1.0
        prep_obs_states = obs_states
        norm_obs_states = ((prep_obs_states-min_pixel)/float(max_pixel-min_pixel) ).astype(np.float32)
        pt_norm_obs_states = Variable(torch.FloatTensor(norm_obs_states)).to(self.DEVICE)
        x_d, z_e_x, z_q_x, obs_latents = self.vq_model(pt_norm_obs_states)
        return x_d, z_e_x, z_q_x, obs_latents

    def find_future_latents(self, cond_latents, rollout_steps):
        print("starting future predict for %d steps"%rollout_steps)
        # predict next
        # assumes input is pytorch
        pst = time.time()
        spat_cond = Variable(torch.LongTensor(cond_latents[None])).to(self.DEVICE)
        # TODO find latent_shape
        latent_shape = spat_cond.shape[-2:]
        for future_step in range(1,rollout_steps+1):
            pred_latents = self.pcnn_model.generate(spatial_cond=spat_cond, shape=latent_shape, batch_size=1)
            # add this predicted one to the tail
            spat_cond = torch.cat((spat_cond[0,1:],pred_latents))[None]
            if future_step == 1:
                all_pred_latents = pred_latents
            else:
                all_pred_latents = torch.cat((all_pred_latents, pred_latents))
        ped = time.time()
        print("latent pred time", round(ped-pst, 2))
        return all_pred_latents

    def decode_latents(self, latents):
        num_pred,l_shape_h,l_shape_w = latents.shape
        ptlatents = Variable(torch.LongTensor(latents)).view(num_pred, -1).to(self.DEVICE)
        z_q_x = self.vq_model.embedding(ptlatents)
        z_q_x = z_q_x.view(num_pred,l_shape_h,l_shape_w,-1).permute(0,3,1,2)
        x_d = self.vq_model.decoder(z_q_x)

        x_tilde = sample_from_discretized_mix_logistic(x_d,self.nr_logistic_mix,only_mean=True)
        pred_obs = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
        return pred_obs[:,0]



    #_, ys, xs = obs_states.shape
    #proad_states = np.zeros((rollout_steps,ys,xs))
    #print("starting image")
    #ist = time.time()
    ## generate road
    #z_q_x = vq_model.embedding(all_pred_latents.view(all_pred_latents.size(0),-1))
    #z_q_x = z_q_x.view(all_pred_latents.shape[0],latent_shape[0], latent_shape[1], -1).permute(0,3,1,2)
    #x_d = vq_model.decoder(z_q_x)
    #x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=True)
    #proad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel

    #for cc in range(num_samples):
    #    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=False)
    #    sroad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
    #    # for each predicted state
    #    proad_states = np.maximum(proad_states, sroad_states)

    #iet = time.time()
    #print("image pred time", round(iet-ist, 2))
    #return proad_states.astype(np.int)[:,0]


#    def forward_step(self,)
#
#def get_vqvae_pcnn_model(state_index, est_inds, cond_states, num_samples=0):
#    rollout_steps = len(est_inds)
#    print("starting vqvaepcnn - %s predictions for state index %s " %(len(est_inds), state_index))
#    # (6,6) or (10,10)
#    est = time.time()
#    print("condition prep time", round(est-st,2))
#    for ind, frame_num  in enumerate(est_inds):
#        pst = time.time()
#        print("predicting latent: %s" %frame_num)
#        # predict next
#        spat_cond = cond_latents[None].to(DEVICE)
#        pred_latents = pcnn_model.generate(spatial_cond=spat_cond, shape=latent_shape, batch_size=1)
#        # add this predicted one to the tail
#        cond_latents = torch.cat((cond_latents[1:],pred_latents))
#        if not ind:
#            all_pred_latents = pred_latents
#        else:
#            all_pred_latents = torch.cat((all_pred_latents, pred_latents))
#
#        ped = time.time()
#        print("latent pred time", round(ped-pst, 2))
#    proad_states = np.zeros((rollout_steps,ys,xs))
#    print("starting image")
#    ist = time.time()
#    # generate road
#    z_q_x = vq_model.embedding(all_pred_latents.view(all_pred_latents.size(0),-1))
#    z_q_x = z_q_x.view(all_pred_latents.shape[0],latent_shape[0], latent_shape[1], -1).permute(0,3,1,2)
#    x_d = vq_model.decoder(z_q_x)
#    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=True)
#    proad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
#
#    for cc in range(num_samples):
#        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=False)
#        sroad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
#        # for each predicted state
#        proad_states = np.maximum(proad_states, sroad_states)
#
#    iet = time.time()
#    print("image pred time", round(iet-ist, 2))
#    return proad_states.astype(np.int)[:,0]

if __name__ == '__main__':
    vqpcnn_model = VQPCNN_model('cpu',
                                '../../model_savedir',
                                load_vq_name='nfreeway_vqvae4layer_nl_k512_z64e00250_good.gpkl',
                                load_pcnn_name='erpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00040_good.gpkl',
                                dsize=80, nr_logistic_mix=10,
                                num_z=64, num_clusters=512,
                                N_LAYERS = 10, # layers in pixelcnn
                                DIM = 256,
                                history_size=4,
                                )

