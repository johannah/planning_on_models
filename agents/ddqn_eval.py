# extending on code from
# https://github.com/58402140/Fruit

import matplotlib
matplotlib.use('Agg')
import sys
# TODO - fix install
sys.path.append('../models')
import config
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prepare_atari import DMAtariEnv
from experience_handler import experience_replay
from IPython import embed
from imageio import imwrite
from glob import glob
from ddqn_atari import DDQNCoreNet
from dqn_utils import handle_step, seed_everything

def run_eval_episode(epoch_num, total_steps):
    start_steps = total_steps
    checkpoint = ''
    episode_steps = 0
    start = time.time()
    episodic_reward = 0.0
    S, action, reward, finished = env.reset()
    # init current state buffer with initial frame
    S_hist = [S for _ in range(info['HISTORY_SIZE'])]
    policy_net.eval()
    episode_actions = [action]
    total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S, action, reward, finished, info['RANDOM_HEAD'], info['FAKE_ACTS'], 0, exp_replay)
    # end fake start
    while not finished:
        epsilon = random_state.rand()
        if epsilon < .01:
            action = random_state.choice(action_space)
            print("random action", action)
        else:
            with torch.no_grad():
                # always do this calculation - as it is used for debugging
                S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(info['DEVICE'])
                vals = policy_net(S_hist_pt).cpu().data.numpy()
                action = np.argmax(vals, axis=-1)[0]

        # randomly choose which head to say was used in case there are
        S_prime, reward, finished = env.step4(action)
        total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S_prime, action, reward, finished, info['RANDOM_HEAD'], info['FAKE_ACTS'], episodic_reward, exp_replay, checkpoint)
        episode_actions.append(action)

    stop = time.time()
    ep_time =  stop - start
    print("EPISODE:%s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, episodic_reward, total_steps-start_steps, total_steps))
    print('actions', episode_actions)
    return episodic_reward, total_steps, ep_time

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-n', '--num_eval_episodes', default=100, help='')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)
    accumulation_rewards = []
    overall_time = 0.

    print('loading model from %s' %args.model_loadpath)
    model_dict = torch.load(args.model_loadpath)
    info = model_dict['info']
    info["SEED"] = model_dict['cnt']
    model_base_filedir = os.path.split(args.model_loadpath)[0]
    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    last_save = model_dict['cnt']

    env = DMAtariEnv(info['GAME'],random_seed=info['SEED']+100)
    action_space = np.arange(env.env.action_space.n)
    seed_everything(info["SEED"]+100)
    policy_net = DDQNCoreNet(info['HISTORY_SIZE'], env.env.action_space.n).to(info['DEVICE'])
    target_net = DDQNCoreNet(info['HISTORY_SIZE'], env.env.action_space.n).to(info['DEVICE'])

    # what about random states - they will be wrong now???
    # TODO - what about target net update cnt
    target_net.load_state_dict(model_dict['target_net_state_dict'])
    policy_net.load_state_dict(model_dict['policy_net_state_dict'])
    #opt.load_state_dict(model_dict['optimizer'])
    total_steps = model_dict['cnt']
    epoch_start = model_dict['epoch']

    exp_replay = experience_replay(batch_size=info['BATCH_SIZE'],
                                   max_size=10000,
                                   history_size=info['HISTORY_SIZE'],
                                   name='eval_buffer', random_seed=info['SEED'],
                                   buffer_file='', is_eval=True)

    random_state = np.random.RandomState(info["SEED"])
    next(exp_replay) # Start experience-replay coroutines

evaluation_rewards = []
eval_steps = 0
for epoch_num in range(args.num_eval_episodes):
    eval_reward, eval_steps, etime = run_eval_episode(epoch_num, eval_steps)
    overall_time += etime
    evaluation_rewards.append(eval_reward)

filename = model_base_filepath + "_%010dq.pkl"%total_steps
exp_replay.send((filename, None))

print(evaluation_rewards)
print("MEAN", np.mean(evaluation_rewards))
print("MED", np.median(evaluation_rewards))
print("MAX", np.max(evaluation_rewards))
print("MIN", np.min(evaluation_rewards))

