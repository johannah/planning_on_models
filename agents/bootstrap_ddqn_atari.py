import os
import sys
import numpy as np
from IPython import embed

import math
#from logger import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from replay_buffer import ReplayBuffer
#from experience_handler import experience_replay
#from prepare_atari import DMAtariEnv
from dqn_model import EnsembleNet
from dqn_utils import handle_step, seed_everything, write_info_file
from env import Environment
from glob import glob
sys.path.append('../models')
from lstm_utils import plot_dict_losses
import config
from ae_utils import save_checkpoint
from dqn_utils import seed_everything, write_info_file

def train_batch(cnt):
    st = time.time()
    # min history to learn is 200,000 frames in dqn
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    if rbuffer.ready(info['BATCH_SIZE']):
        samples = rbuffer.sample_random(info['BATCH_SIZE'], pytorchify=True)
        states, actions, rewards, next_states, ongoing_flags, masks, _ = samples

        opt.zero_grad()
        q_values = policy_net(states, None)
        next_q_values = policy_net(next_states, None)
        next_q_state_values = target_net(next_states, None)
        cnt_losses = []
        for k in range(info['N_ENSEMBLE']):
            #TODO finish masking
            total_used = 1.0
            #total_used = torch.sum(mask_pt[:, k])
            if total_used > 0.0:
                q_value = q_values[k].gather(1, actions[:,None]).squeeze(1)
                next_q_value = next_q_state_values[k].gather(1, next_q_values[k].max(1)[1].unsqueeze(1)).squeeze(1)
                expected_q_value = rewards + (info["GAMMA"] * next_q_value * ongoing_flags)
                #loss = (q_value-expected_q_value.detach()).pow(2).mean()
                # TODO do mask
                loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
                cnt_losses.append(loss)
                losses[k] = loss.cpu().detach().item()

        all_loss = torch.stack(cnt_losses).sum()
        all_loss.backward()
        for param in policy_net.core_net.parameters():
            if param.grad is not None:
                param.grad.data *=1.0/float(info['N_ENSEMBLE'])
        opt.step()
        if not cnt%info['TARGET_UPDATE']:
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print('updating target network at %s'%cnt)
            target_net.load_state_dict(policy_net.state_dict())
    #board_logger.scalar_summary('batch train time per cnt', cnt, time.time()-st)
    #board_logger.scalar_summary('loss per cnt', cnt, np.mean(losses))
    return np.mean(losses)

def handle_checkpoint(last_save, cnt, epoch, last_mean):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        print("checkpoint")
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'epoch':epoch,
                 'policy_net_state_dict':policy_net.state_dict(),
                 'target_net_state_dict':target_net.state_dict(),
                 'last_mean':last_mean,
                 'steps':steps,
                 'episode_step':episode_step,
                 'episode_head':episode_head,
                 'episode_loss':episode_loss,
                 'episode_reward':episode_reward,
                 'episode_times':episode_times,
                 'avg_rewards':avg_rewards,
                 }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
        save_checkpoint(state, filename)
        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer.pkl"%cnt)
        rbuffer.save(buff_filename)
        return last_save
    else: return last_save


def run_training_episode(epoch_num, total_steps):
    finished = False
    start = time.time()
    episodic_losses = []
    start_steps = total_steps
    episodic_reward = 0.0
    _S = env.reset()
    rbuffer.add_init_state(_S)
    episode_actions = []
    policy_net.train()
    random_state.shuffle(heads)
    active_head = heads[0]
    losses = []
    while not finished:
        est = time.time()
        with torch.no_grad():
            _Spt = torch.Tensor(_S[None]).to(info['DEVICE'])
            vals = policy_net(_Spt, active_head)
            action = torch.argmax(vals, dim=1).item()
            ## always do this calculation - as it is used for debugging
            #vals = policy_net(_Spt, None)
            #acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
            #action = acts[active_head]
            #vals = policy_net(_Spt)
            #action = np.argmax(vals.cpu().data.numpy(),-1)[0]
        #board_logger.scalar_summary('time get action per step', total_steps, time.time()-est)
        bfa = time.time()
        _S_prime, reward, finished = env.step(action)
        rbuffer.add_experience(next_state=_S_prime[-1], action=action, reward=reward, finished=finished)
        #board_logger.scalar_summary('time take_step_and_add per step', total_steps, time.time()-bfa)
        losses.append(train_batch(total_steps))
        _S = _S_prime
        episodic_reward += reward
        total_steps+=1
        eet = time.time()
    stop = time.time()
    ep_time =  stop - start

    steps.append(total_steps)
    episode_step.append(total_steps-start_steps)
    episode_head.append(active_head)
    episode_loss.append(np.mean(losses))
    episode_reward.append(episodic_reward)
    episode_times.append(ep_time)
    avg_rewards.append(np.mean(episode_reward[-100:]))

    #board_logger.scalar_summary('%s head reward per episode'%active_head, epoch_num, episodic_reward)
    #board_logger.scalar_summary('head per episode', epoch_num, active_head)
    #board_logger.scalar_summary('reward per episode', epoch_num, episodic_reward)
    #board_logger.scalar_summary('reward per step', total_steps, episodic_reward)
    #board_logger.scalar_summary('time per episode', epoch_num, ep_time)
    #board_logger.scalar_summary('steps per episode', epoch_num, total_steps-start_steps)
    print("EPISODE:%s HEAD %s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, active_head, episodic_reward, total_steps-start_steps, total_steps))
    print("time for episode", ep_time)
    if not epoch_num%10:
        # TODO plot title
        plot_dict_losses({'episode steps':{'index':np.arange(epoch_num+1), 'val':episode_step}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
        plot_dict_losses({'episode head':{'index':np.arange(epoch_num+1), 'val':episode_head}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
        plot_dict_losses({'steps loss':{'index':steps, 'val':episode_loss}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
        plot_dict_losses({'steps reward':{'index':steps, 'val':episode_reward}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
        plot_dict_losses({'episode reward':{'index':np.arange(epoch_num+1), 'val':episode_reward}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
        plot_dict_losses({'episode times':{'index':np.arange(epoch_num+1), 'val':episode_times}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
        plot_dict_losses({'steps avg reward':{'index':steps, 'val':avg_rewards}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
        print('avg reward', avg_rewards[-1])
    return episodic_reward, total_steps, ep_time

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz model file full path')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        "GAME":'roms/breakout.bin', # gym prefix
        "DEVICE":device,
        "NAME":'_ROMSBreakout_BT9_LR', # start files with name
        "N_ENSEMBLE":9,
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":500000,
        "ADAM_LEARNING_RATE":0.00001,
        "ADAM_EPSILON":1.5e-4,
        "RMS_LEARNING_RATE": 0.00001,
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "CLIP_REWARD_MAX":1,
        "CLIP_REWARD_MAX":-1,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "PRINT_EVERY":1, # How often to print statistics
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "EPSILON_MAX":1.0, # Epsilon greedy exploration ~prob of random action, 0. disables
        "EPSILON_MIN":.1,
        "EPSILON_DECAY":1000000,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1,
        "FAKE_ACTION":-3,
        "FAKE_REWARD":-5,
        "NETWORK_INPUT_SIZE":(84,84),
        }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    accumulation_rewards = []
    overall_time = 0.
    info['load_time'] = datetime.date.today().ctime()

    if args.model_loadpath != '':
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        total_steps = model_dict['cnt']
        info['DEVICE'] = device
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        last_save = model_dict['cnt']
        info['loaded_from'] = args.model_loadpath
        epoch_start = model_dict['epoch']+1
        steps = model_dict['steps']
        episode_step = model_dict['episode_step']
        episode_head = model_dict['episode_head']
        episode_loss = model_dict['episode_loss']
        episode_reward = model_dict['episode_reward']
        episode_times = model_dict['episode_times']
        avg_rewards = model_dict['avg_rewards']
    else:
        total_steps = 0
        last_save = 0
        epoch_start = 0
        run_num = 0
        steps = []
        episode_step = []
        episode_head = []
        episode_loss = []
        episode_reward = []
        episode_times = []
        avg_rewards = []


        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(info, model_base_filepath, total_steps)
    env = Environment(info['GAME'])
    action_space = np.arange(env.num_actions)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])
    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])


    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'], eps=info['ADAM_EPSILON'])

    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])

    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            print("NOT LOADING BUFFER")
            args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.pkl')
            print("auto loading buffer from:%s" %args.buffer_loadpath)
            rbuffer.load(args.buffer_loadpath)

    rbuffer = ReplayBuffer(max_buffer_size=info['BUFFER_SIZE'],
                           history_size=info['HISTORY_SIZE'],
                           min_sampling_size=info['MIN_HISTORY_TO_LEARN'],
                           num_masks=info['N_ENSEMBLE'],
                           bernoulli_probability=info['BERNOULLI_PROBABILITY'],
                           device=info['DEVICE'])


    random_state = np.random.RandomState(info["SEED"])
    #board_logger = TensorBoardLogger(model_base_filedir)
    last_target_update = 0
    all_rewards = []

    print("Starting training")
    for epoch_num in range(epoch_start, info['N_EPOCHS']):
        ep_reward, total_steps, etime = run_training_episode(epoch_num, total_steps)
        all_rewards.append(ep_reward)
        overall_time += etime
        last_mean = np.mean(all_rewards[-100:])
        #board_logger.scalar_summary("avg reward last 100 episodes", epoch_num, last_mean)
        last_save = handle_checkpoint(last_save, total_steps, epoch_num, last_mean)


