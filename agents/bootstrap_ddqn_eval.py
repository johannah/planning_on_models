# extending on code from
# https://github.com/58402140/Fruit

import matplotlib
matplotlib.use('Agg')
import sys
# TODO - fix install
sys.path.append('../models')
import config
from ae_utils import save_checkpoint
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
from dqn_model import EnsembleNet, NetWithPrior
from experience_handler import experience_replay
from IPython import embed
from imageio import imwrite
from glob import glob
from dqn_utils import handle_step, seed_everything


def run_eval_episode(epoch_num, total_steps):
    checkpoint = ''
    start_steps = total_steps
    episode_steps = 0
    start = time.time()
    random_state.shuffle(heads)
    active_head = heads[0]
    episodic_reward = 0.0
    S, action, reward, finished = env.reset()
    # init current state buffer with initial frame
    S_hist = [S for _ in range(info['HISTORY_SIZE'])]
    policy_net_ensemble.eval()
    episode_actions = [action]
    total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S, action, reward, finished, info['RANDOM_HEAD'], info['FAKE_ACTS'], 0, exp_replay)
    # end fake start
    while not finished:
        with torch.no_grad():
            # always do this calculation - as it is used for debugging
            S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(info['DEVICE'])
            vals = [q.cpu().data.numpy() for q in policy_net_ensemble(S_hist_pt, None)]
            acts = [np.argmax(v, axis=-1)[0] for v in vals]

        act_counts = Counter(acts)
        max_count = max(act_counts.values())
        top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count]
        # break action ties with random choice
        random_state.shuffle(top_actions)
        action = top_actions[0]
        top_heads = [k for (k,a) in enumerate(acts) if a in top_actions]
        # randomly choose which head to say was used in case there are
        # multiple heads that chose same action
        k_used = top_heads[random_state.randint(len(top_heads))]
        S_prime, reward, finished = env.step4(action)
        total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, exp_replay, checkpoint)
        episode_actions.append(action)
    stop = time.time()
    ep_time =  stop - start
    print("EPISODE:%s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, episodic_reward, total_steps-start_steps, total_steps))
    print('actions', episode_actions)
    return episodic_reward, total_steps, ep_time

def write_info_file(model_loaded=''):
    info_f = open(os.path.join(model_base_filedir, 'info%s.txt'%model_loaded), 'w')
    info_f.write(datetime.date.today().ctime()+'\n')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-n', '--num_eval_episodes', default=100, help='')
    #parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz model file full path')
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
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"]+100)
    policy_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])
    target_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])


    target_net_ensemble.load_state_dict(model_dict['target_net_ensemble_state_dict'])
    policy_net_ensemble.load_state_dict(model_dict['policy_net_ensemble_state_dict'])
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

