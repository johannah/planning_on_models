# Author: Johanna Hansen & Kyle Kastner
# License: BSD 3-Clause
# Kyle's mcts original mcts implementation is here:
# https://gist.github.com/kastnerkyle/1563d1867a33eac6203879ac41df2407
# See similar MCTS implementation here
# https://github.com/junxiaosong/AlphaZero_Gomoku
# changes from high level mcts pseudo-code in survey
# http://mcts.ai/pubs/mcts-survey-master.pdf
# expand all children, but only rollout one
# section biases to unexplored nodes, so the children with no rollout
# will be explored quickly

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time
from copy import deepcopy
from IPython import embed
from collections import Counter
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint, linearly_decaying_epsilon
from env import Environment
from state_managers import VQRolloutStateManager
from replay import ReplayMemory
import config
from mcts import MCTS

def plot_episode(mcts,true_obs, rec_obs, actions, rewards, latent_states_list):
    epath = os.path.join(model_base_filepath, 'steps')
    os.makedirs(epath)
    print("plotting episode")
    print(epath)
    for i in range(true_obs.shape[0]):
        f,ax=plt.subplots(2,3)
        next_latent_state, next_x_d, _, pred_reward =  mcts.state_manager.get_next_latent(latent_states_list[i], actions[i])
        next_s_est,next_s_mean = mcts.state_manager.sample_from_latents(next_x_d)
        ax[0,0].imshow(true_obs[i])
        ax[0,0].set_title('S%d'%i)
        ax[0,1].imshow(rec_obs[i,0])
        ax[0,1].set_title('A%sR%s'%(actions[i], rewards[i]))
        ax[1,0].imshow(latent_states_list[i][0,0])
        ax[1,0].set_title('S%d'%(i-1))
        ax[1,1].imshow(latent_states_list[i][0,1])
        ax[1,1].set_title('S%d'%i)
        ax[0,2].imshow(next_s_est[0,0])
        iname = os.path.join(epath, 'S%06d'%i)
        plt.savefig(iname)
        plt.close()
    cmd = 'convert %s %s'%(os.path.join(epath, 'S*.png'),
                           os.path.join(epath, '_steps.gif'))
    os.system(cmd)



def run(step_number, last_save):
    mcts_random = np.random.RandomState(1110)
    vqfr_sm = VQRolloutStateManager(info['FORWARD_MODEL_LOADPATH'],
                                    model_base_filedir,
                                    n_playout=info['NUM_PLAYOUTS'],
                                    )
    mcts = MCTS(vqfr_sm, n_playout=info['NUM_PLAYOUTS'],
                rollout_limit=info['ROLLOUT_LIMIT'],
                gamma=info['GAMMA'], random_state=mcts_random)
    terminal = False
    for e in range(info['N_EPISODES']):
        #state = mcts.state_manager.get_init_state()
        _ = env.reset()
        noop_action = 0
        last_state, reward, life_lost, terminal = env.step(noop_action)
        state, reward, life_lost, terminal = env.step(noop_action)
        last_latent_obs,last_x_d = mcts.state_manager.get_state_representation(last_state[None])
        latent_obs,x_d = mcts.state_manager.get_state_representation(state[None])
        print(x_d.max(), x_d.min())
        latent_state = torch.stack((last_latent_obs,latent_obs), dim=1)

        # can i pass env to something
        #winner, end = mcts.state_manager.is_finished(state)
        #states = [state]
        start_steps = step_number
        st = time.time()
        episode_reward_sum = 0
        latent_state_list = []
        obs_state_list = []
        actions = []
        rewards = []
        x_ds = []
        step_times = []
        while not terminal:
            st = time.time()
            # mcts will take this state and roll it forward
            action, ap = mcts.sample_action(latent_state, temp=1E-3, add_noise=False)
            latent_state_list.append(latent_state)
            action = random_state.choice(np.array([0,1,2]),
                                         p=np.array([0.05,.9,0.05]))
            next_state, reward, life_lost, terminal = env.step(action)
            obs_state_list.append(next_state[-1])
            if reward>0:
                print("REWARD")
            print(step_number, 'A', action, 'R', reward)
            actions.append(action)
            rewards.append(reward)
            mcts.update_tree_root(action)
            step_number += 1
            episode_reward_sum += reward
            next_latent,x_d = mcts.state_manager.get_state_representation(next_state[None])
            x_ds.append(x_d)
            # latent state needs previous latent and obs latent
            latent_state = torch.stack((latent_state[0,1][None], next_latent), dim=1)
            state = next_state
            et = time.time()
            print(et-st)
            print(actions)
            step_times.append(et-st)
            #if life_lost:
            #    mcts.reset_tree()
            #else:
            #    mcts.reconstruct_tree()


        rec_est, rec_mean = mcts.state_manager.sample_from_latents(torch.cat(x_ds))
        plot_episode(mcts,np.array(obs_state_list), rec_est, actions, rewards, latent_state_list)
        et = time.time()
        ep_time = et-st
        perf['step_times'].append(step_times)
        perf['episode_num']+=1
        perf['steps'].append(step_number)
        perf['episode_step'].append(step_number-start_steps)
        perf['episode_reward'].append(episode_reward_sum)
        perf['episode_times'].append(ep_time)
        perf['episode_relative_times'].append(time.time()-info['START_TIME'])
        perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz replay buffer file full path')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        "GAME":'roms/freeway.bin', # ale binary
        "FORWARD_MODEL_LOADPATH":"../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/convVQoutmdiffRAB01/convVQoutmdiffRAB_0074514304ex.pt",
        "MIN_SCORE_GIF":0, # min score to plot gif in eval
        "DEVICE":device, #cpu vs gpu set by argument
        "NUM_PLAYOUTS":50,
        "ROLLOUT_LIMIT":10,
        "NAME":'MCTS', # start files with name
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        "CHECKPOINT_EVERY_STEPS":500000, # how often to write pkl of model and npz of data buffer
        "EVAL_FREQUENCY":500000, # how often to run evaluation episodes
        "ADAM_LEARNING_RATE":6.25e-5,
        "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPISODES":1,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "GAMMA":.99, # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 50,
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":101,
        "OBS_SIZE":(84,84),
        "RESHAPE_SIZE":64*7*7,
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        #"MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP":4, # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
        "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life
    }

    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    random_state = np.random.RandomState(info["SEED"])
    if 1:
        # create new project
        perf = {'steps':[],
                'avg_rewards':[],
                'episode_step':[],
                'eps_list':[],
                'episode_loss':[],
                'episode_num':0,
                'step_times':[],
                'episode_reward':[],
                'episode_times':[],
                'episode_relative_times':[],
                'eval_rewards':[],
                'eval_steps':[]}

        start_step_number = 0
        start_last_save = 0
        # make new directory for this run in the case that there is already a
        # project with this name
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(info, model_base_filepath, start_step_number)
    seed_everything(info["SEED"])
    run(start_step_number, start_last_save)

