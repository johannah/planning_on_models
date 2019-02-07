import os
import sys
import numpy as np
from IPython import embed

import math
from logger import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from replay_buffer import ReplayBuffer, samples_to_tensors
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
    if cnt > info['MIN_HISTORY_TO_LEARN']:
        samples = replay_buffer.sample(info['BATCH_SIZE'])
        states, actions, rewards, next_states, ongoing_flags = samples_to_tensors(samples, info['DEVICE'])
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
    board_logger.scalar_summary('batch train time per cnt', cnt, st-time.time())
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
        npy_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer.npz"%cnt)
        save_checkpoint(state, filename)
        replay_buffer.save_buffer(npy_filename)
        return last_save
    else: return last_save


#def handle_new_step(random_state, cnt, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer, checkpoint='', n_ensemble=1, bernoulli_p=1.0):
#    # mask to determine which head can use this experience
#    exp_mask = random_state.binomial(1, bernoulli_p, n_ensemble).astype(np.uint8)
#    # at this observed state
#    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
#    batch = replay_buffer.send((checkpoint, experience))
#    # update so "state" representation is past history_size frames
#    episodic_reward += reward
#    cnt+=1
#    return cnt, batch, episodic_reward


def run_training_episode(epoch_num, total_steps):
    finished = False
    episode_steps = 0
    start = time.time()
    episodic_losses = []
    start_steps = total_steps
    episodic_reward = 0.0
    _S = env.reset()
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
        replay_buffer.append(_S, action, reward, _S_prime, finished)

        #board_logger.scalar_summary('time take action per step', total_steps, time.time()-bfa)
        #cst = time.time()
        #asst = time.time()
        #total_steps, batch, episodic_reward = handle_new_step(random_state, total_steps, S_hist[-1], action, reward, finished, info['RANDOM_HEAD'], vals, episodic_reward, exp_replay, checkpoint)
        #board_logger.scalar_summary('time handle_step per step', total_steps, time.time()-asst)
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
        "NAME":'_ROMSBreakout_ddqn', # start files with name
        "N_EVALUATIONS":10, # Number of evaluation episodes to run
        "BERNOULLI_P": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "CHECKPOINT_EVERY_STEPS":100000,
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
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
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
    write_info_file(info, model_base_filepath, total_steps)
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

    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            args.buffer_loadpath = glob(args.model_loadpath.replace('.pkl', '*.npz'))[0]
            print("auto loading buffer from:%s" %args.buffer_loadpath)
    replay_buffer = ReplayBuffer(info['BUFFER_SIZE'])
    #exp_replay = experience_replay(batch_size=info['BATCH_SIZE'],
    #                               max_size=info['BUFFER_SIZE'],
    #                               history_size=info['HISTORY_SIZE'],
    #                               name='train_buffer', random_seed=info['SEED'],
    #                               buffer_file=args.buffer_loadpath)

    random_state = np.random.RandomState(info["SEED"])
    #next(exp_replay) # Start experience-replay coroutines

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


