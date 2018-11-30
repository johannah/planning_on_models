# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import datetime
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

from models import config
from models.datasets import prepare_img, undo_img_scaling, chicken_color, input_ysize, input_xsize, base_chicken
from models.utils import discretized_mix_logistic_loss, get_cuts, to_scalar
from models.utils import sample_from_discretized_mix_logistic

from pmcts import PMCTS
from vqvae_pcnn_future_model import VQPCNN_model
min_pixel, max_pixel = 0., 255.
latent_xsize = 10
latent_ysize = 10

def frame_skip_step(env, last_frame, action, reward, done, num_steps_to_take, total_steps_taken):
    observ = last_frame
    for i in range(num_steps_to_take):
        if not done:
            o, r, done, _ = env.step(action)
            reward += r
            observ = np.maximum(o, last_frame)
            total_steps_taken +=1
            last_frame = o
    return env, observ, last_frame, reward, done, total_steps_taken

def add_real_experience(records, observations, pobservations, lobservations, obs_count, ref_state_index, obs, action, reward):
    agent_loc, pobs = prepare_img(deepcopy(obs))
    # going into encode should be
    x_d, z_e_x, z_q_x, obs_latents = future_model.encode_obs(pobs[None,None,:])
    observations[obs_count] = obs
    pobservations[obs_count] = pobs
    lobservations[obs_count] = obs_latents[0] # first dimension is instance - we only use one
    records['rewards'].append(reward)
    records['actions'].append(action)
    records['ref_state_index'].append(ref_state_index)
    records['obs_count'].append(obs_count)
    records['decision_times'].append(time.time())
    records['agent_position_y_max'].append(np.max(agent_loc[0]))
    records['agent_position_y_min'].append(np.min(agent_loc[0]))
    records['agent_position_x_max'].append(np.max(agent_loc[1]))
    records['agent_position_x_min'].append(np.min(agent_loc[1]))
    return records, observations, pobservations, lobservations

def predict_future_latents(records, lobservations, lpredictions, obs_count, rollout_steps):
    conditioning = lobservations[obs_count-4:obs_count]
    future_latents = future_model.find_future_latents(conditioning, rollout_steps)
    lpredictions[obs_count] = future_latents
    records['future_prediction_indexes'].append(obs_count)
    print('adding obs_count', obs_count, lpredictions[obs_count].sum())
    return records, future_latents, lpredictions

def run_trace(seed,
        n_playouts, rollout_steps,
        prob_fn='random',
        history_size=4, start_index=0,
        step_limit=18000, frame_skip=4,
        do_render=False):

    # log params
    states = []
    true_env = gym.make('FreewayNoFrameskip-v4')
    action_space = range(true_env.action_space.n)
    mcts_rdn = np.random.RandomState(seed+1)
    rdn = np.random.RandomState(seed)

    records = {'decision_times':[],
              'actions':[],
              'rewards':[],
              'agent_position_y_max':[],
              'agent_position_y_min':[],
              'agent_position_x_max':[],
              'agent_position_x_min':[],
              'obs_count':[],
              'ref_state_index':[],
              'future_prediction_indexes':[],
              }

    # prepare initial
    done = False
    reward = 0
    obs_count = 0
    action = 0
    ref_state_index = 0
    last_frame  = true_env.reset()
    true_env, obs, last_frame, reward, done, ref_state_index = frame_skip_step(true_env, last_frame, action, reward, done, start_index, ref_state_index)

    agent_loc, pobs = prepare_img(deepcopy(obs))
    obs_xsize, obs_ysize, obs_chans = obs.shape
    pobs_xsize, pobs_ysize = pobs.shape
    # real observations from simulator
    observations = np.zeros((step_limit+5, obs_xsize, obs_ysize, obs_chans), dtype=obs.dtype)
    # my shaped observations - 1 channel, shrunk....
    pobservations = np.zeros((step_limit+5, pobs_xsize, pobs_ysize), dtype=pobs.dtype)
    # TODO - am putting this in a np but it is actually pt
    lobservations = np.zeros((step_limit+5, latent_xsize, latent_ysize))
    # TODO - am putting this in a np but it is actually pt
    # latent predictions
    lpredictions = np.zeros((step_limit+5, rollout_steps, latent_xsize, latent_ysize))

    # fast forward history steps so agent observes 4
    # start - must wait history_size frames before making decision for so we can
    # use our model action is 0
    for i in range(history_size):
        # do nothing and collect 4 frames
        action = 0
        true_env, obs, last_frame, reward, done, ref_state_index = frame_skip_step(true_env, last_frame, action, reward, done, frame_skip, ref_state_index)
        records, observations, pobservations, lobservations = add_real_experience(records, observations, pobservations, lobservations, obs_count, ref_state_index, obs, action, reward)
        obs_count +=1

    #pmcts = PMCTS(random_state=mcts_rdn,
    #              node_probs_name=prob_fn,
    #              n_playouts=n_playouts,
    #              rollout_steps=rollout_steps,
    #              future_model=future_model,history_size=history_size)



    while not done:
        if obs_count >= step_limit:
            done = True
        if rdn.rand()<.01:
            # search
            records, this_state_future_latents, lpredictions = predict_future_latents(records, lobservations, lpredictions, obs_count, rollout_steps)
            print('adding obs_count', obs_count, lpredictions[obs_count].sum())
            action = rdn.choice(action_space)
        else:
            action = 1 #TODO

        true_env, obs, last_frame, reward, done, ref_state_index = frame_skip_step(true_env, last_frame, action, reward, done, frame_skip, ref_state_index)
        records, observations, pobservations, lobservations = add_real_experience(records, observations, pobservations, lobservations, obs_count, ref_state_index, obs, action, reward)
        obs_count +=1
        if not obs_count % 1000:
            print(obs_count, reward, action, done)

    # only save latents which had imaginations
    return records, observations[:obs_count+1], pobservations[:obs_count+1], lobservations[:obs_count+1], lpredictions[records['future_prediction_indexes']]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-bs', '--buffer_size', type=int, default=5, help='buffer size around q value')
    parser.add_argument('--history_size', type=int, default=4, help='number of frames to use in conditioning - should be 4 unless retrained')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=5, help='limit number of steps taken be random rollout')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')
    parser.add_argument('-t', '--future_model', type=str, default='vqvae_pcnn')
    parser.add_argument('-msf', '--env_model_type', type=str, default='equiv_model_step')
    parser.add_argument('-sams', '--num_samples', type=int , default=5)
    parser.add_argument('-gs', '--goal_speed', type=float , default=0.5)
    parser.add_argument('-neo', '--neo_goal_prior', type=float , default=0.01)
    parser.add_argument('-sm', '--smoothing', type=float , default=0.5)
    parser.add_argument('-sl', '--step_limit', type=float , default=1000)
    parser.add_argument('-fs', '--frame_skip', type=float , default=4)
    parser.add_argument('-ne', '--num_episodes', type=float , default=30)

    parser.add_argument('--save_pkl', action='store_false', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('-gap', '--plot_playout_gap', type=int, default=5, help='gap between plot playouts for each step')
    parser.add_argument('-f', '--prior_fn', type=str, default='equal_node_probs_fn', help='options are goal_node_probs_fn or equal_node_probs_fn')
    parser.add_argument('--vq_model_name', default='nfreeway_vqvae4layer_nl_k512_z64e00250_good.gpkl')
    parser.add_argument('--pcnn_model_name', default='erpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00040_good.gpkl')

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    use_cuda = args.cuda
    seed = args.seed
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    if args.future_model == 'vqvae_pcnn':
        future_model = VQPCNN_model(DEVICE,
                                    config.model_savedir,
                                    load_vq_name=args.vq_model_name,
                                    load_pcnn_name=args.pcnn_model_name,
                                    dsize=80, nr_logistic_mix=10,
                                    num_z=64, num_clusters=512,
                                    N_LAYERS = 10, # layers in pixelcnn
                                    DIM = 256,
                                    history_size=args.history_size,
                                    )


    for start_index in range(30):
        # load relevent file

        dt = datetime.datetime.now().strftime("%Y-%m-%d-T%H-%M-%S")
        sum_fpath =  os.path.join(config.results_savedir, 'd%s_SI%03d_summary.pkl' %(dt,start_index))
        data_fpath = os.path.join(config.results_savedir, 'd%s_SI%03d_data.npz' %(dt,start_index))

        if not os.path.exists(config.results_savedir):
            os.makedirs(config.results_savedir)

        if not os.path.exists(data_fpath):
            print("STARTING EPISODE start_index %s" %(start_index))
            st = time.time()
            e_records, e_obs, e_pobs, e_lobs, e_lpreds  = run_trace(seed=seed,
                          n_playouts=args.num_playouts,
                          rollout_steps=args.rollout_steps,
                          prob_fn=args.prior_fn,
                          history_size=args.history_size, start_index=start_index,
                          step_limit=args.step_limit, frame_skip=args.frame_skip, do_render=args.render)

            et = time.time()
            e_records['DEVICE'] = DEVICE
            e_records['full_end_time'] = et
            e_records['full_start_time'] = st
            e_records['seed'] = seed
            e_records['args'] = args
            pickle.dump(e_records,open(sum_fpath, 'w'))
            np.savez(data_fpath, observations=e_obs,
                                pobservations=e_pobs,
                                lobservations=e_lobs,
                                lpredictions=e_lpreds,
                     )
            print("saved start_index %s"%start_index)
            embed()
    print("FINISHED")


