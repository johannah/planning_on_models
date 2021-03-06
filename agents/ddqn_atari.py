import os
import sys
import numpy as np
from IPython import embed
import pickle

import math
from logger import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from replay_buffer import ReplayBuffer
#from experience_handler import experience_replay
#from prepare_atari import DMAtariEnv
from env import Environment
sys.path.append('../models')
from glob import glob
import config
from ae_utils import save_checkpoint
from dqn_utils import linearly_decaying_epsilon, handle_step, seed_everything, write_info_file

class DDQNCoreNet(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(DDQNCoreNet, self).__init__()
        self.num_channels =  num_channels
        self.num_actions = num_actions
        conv1 = nn.Conv2d(self.num_channels, 32, kernel_size=8, stride=4)
        torch.nn.init.kaiming_uniform_(conv1.weight, nonlinearity='relu')
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        torch.nn.init.kaiming_uniform_(conv2.weight, nonlinearity='relu')
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        torch.nn.init.kaiming_uniform_(conv3.weight, nonlinearity='relu')
        self.conv_layers = nn.Sequential(conv1, nn.ReLU(),
                                         conv2, nn.ReLU(),
                                         conv3, nn.ReLU())
        reshape = 64*7*7
        lin1 = nn.Linear(reshape, 512)
        lin2 = nn.Linear(512, self.num_actions)
        self.lin_layers = nn.Sequential(lin1, nn.ReLU(), lin2)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.lin_layers(x.view(x.shape[0], -1))

def train_batch(cnt):
    st = time.time()
    # min history to learn is 200,000 frames in dqn
    loss = 0.0
    if rbuffer.ready(info['BATCH_SIZE']):
        samples = rbuffer.sample_random(info['BATCH_SIZE'], pytorchify=True)
        states, actions, rewards, next_states, ongoing_flags, _ = samples
        opt.zero_grad()
        q_values = policy_net(states)
        next_q_values = policy_net(next_states)
        next_q_state_values = target_net(next_states)
        q_value = q_values.gather(1, actions[:,None]).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + (info["GAMMA"] * next_q_value * ongoing_flags)
        #loss = (q_value-expected_q_value.detach()).pow(2).mean()
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
        loss.backward()
        opt.step()
        loss = loss.item()
        if not cnt%info['TARGET_UPDATE']:
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print('updating target network')
            target_net.load_state_dict(policy_net.state_dict())
    board_logger.scalar_summary('batch train time per cnt', cnt, time.time()-st)
    board_logger.scalar_summary('loss per cnt', cnt, loss)
    return loss

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
                 }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
        save_checkpoint(state, filename)
        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer.pkl"%cnt)
        rbuffer.save(buff_filename)
        return last_save
    else: return last_save


def run_training_episode(epoch_num, total_steps):
    finished = False
    episode_steps = 0
    start = time.time()
    episodic_losses = []
    start_steps = total_steps
    episodic_reward = 0.0
    _S = env.reset()
    rbuffer.add_init_state(_S)
    episode_actions = []
    policy_net.train()

    while not finished:
        est = time.time()
        epsilon = linearly_decaying_epsilon(info["EPSILON_DECAY"], total_steps, info["MIN_HISTORY_TO_LEARN"], info["EPSILON_MIN"])
        board_logger.scalar_summary('epsilon by step', total_steps, epsilon)

        if (random_state.rand() < epsilon):
            action = random_state.choice(action_space)
            vals = [-1.0 for aa in action_space]
        else:
            with torch.no_grad():
                # always do this calculation - as it is used for debugging
                _Spt= torch.Tensor(_S[None]).to(info['DEVICE'])
                vals = policy_net(_Spt)
                action = np.argmax(vals.cpu().data.numpy(),-1)[0]
        board_logger.scalar_summary('time get action per step', total_steps, time.time()-est)
        bfa = time.time()
        _S_prime, reward, finished = env.step(action)
        rbuffer.add_experience(next_state=_S_prime[-1], action=action, reward=reward, finished=finished)
        board_logger.scalar_summary('time take_step_and_add per step', total_steps, time.time()-bfa)
        train_batch(total_steps)
        _S = _S_prime
        episodic_reward += reward
        total_steps+=1
        eet = time.time()
    stop = time.time()
    ep_time =  stop - start
    board_logger.scalar_summary('reward per episode', epoch_num, episodic_reward)
    board_logger.scalar_summary('reward per step', total_steps, episodic_reward)
    board_logger.scalar_summary('time per episode', epoch_num, ep_time)
    board_logger.scalar_summary('steps per episode', epoch_num, total_steps-start_steps)
    if not epoch_num%5:
        print("EPISODE:%s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, episodic_reward, total_steps-start_steps, total_steps))
        #print('actions',episode_actions)
        print("time for episode", ep_time)
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
        "NAME":'_ROMSBreakout_ddqnMyER', # start files with name
        "N_EVALUATIONS":10, # Number of evaluation episodes to run
        "BERNOULLI_P": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        #"CHECKPOINT_EVERY_STEPS":100,
        "CHECKPOINT_EVERY_STEPS":200000,
        "ADAM_LEARNING_RATE":0.00025,
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
        "BUFFER_SIZE":1e5, # Buffer size for experience replay
        "EPSILON_MAX":1.0, # Epsilon greedy exploration ~prob of random action, 0. disables
        "EPSILON_MIN":.1,
        "EPSILON_DECAY":1000000,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":18, # Learning rate for Adam
        "RANDOM_HEAD":-1,
        "FAKE_ACTION":-3,
        "FAKE_REWARD":-5,
        "NETWORK_INPUT_SIZE":(84,84),
        }

    info['FAKE_ACTS'] = [info['FAKE_ACTION']]
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
        epoch_start = model_dict['epoch']
    else:
        total_steps = 0
        last_save = 0
        epoch_start = 0
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    env = Environment(info['GAME'])
    action_space = np.arange(env.num_actions)
    seed_everything(info["SEED"])
    policy_net = DDQNCoreNet(info['HISTORY_SIZE'], env.num_actions).to(info['DEVICE'])
    target_net = DDQNCoreNet(info['HISTORY_SIZE'], env.num_actions).to(info['DEVICE'])
    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'], eps=info['ADAM_EPSILON'])
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])
    rbuffer = ReplayBuffer(max_buffer_size=info['BUFFER_SIZE'],
                                 history_size=info['HISTORY_SIZE'],
                                 min_sampling_size=info['MIN_HISTORY_TO_LEARN'],
                                 num_masks=0, device=info['DEVICE'])


    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        # TODO cant load buffer yet
        if args.buffer_loadpath == '':
            args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.pkl')
            print("auto loading buffer from:%s" %args.buffer_loadpath)
            rbuffer.load(args.buffer_loadpath)
    info['args'] = args
    write_info_file(info, model_base_filepath, total_steps)
    random_state = np.random.RandomState(info["SEED"])

    board_logger = TensorBoardLogger(model_base_filedir)
    last_target_update = 0
    print("Starting training")
    all_rewards = []

    epsilon_by_frame = lambda frame_idx: info['EPSILON_MIN'] + (info['EPSILON_MAX'] - info['EPSILON_MIN']) * math.exp( -1. * frame_idx / info['EPSILON_DECAY'])
    for epoch_num in range(epoch_start, info['N_EPOCHS']):
        ep_reward, total_steps, etime = run_training_episode(epoch_num, total_steps)
        all_rewards.append(ep_reward)
        overall_time += etime
        last_mean = np.mean(all_rewards[-100:])
        board_logger.scalar_summary("avg reward last 100 episodes", epoch_num, last_mean)
        last_save = handle_checkpoint(last_save, total_steps, epoch_num, last_mean)


