import os
import sys
import numpy as np
from IPython import embed

import math
#from logger import TensorBoardLogger
import torch
torch.set_num_threads(3)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from replay_buffer import ReplayBuffer
from dqn_model import EnsembleNet, weights_init
from dqn_utils import seed_everything, write_info_file
from env import Environment
from glob import glob
sys.path.append('../models')
from lstm_utils import plot_dict_losses
import config
from ae_utils import save_checkpoint

def train_batch(cnt):
    st = time.time()
    # min history to learn is 200,000 frames in dqn
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    if rbuffer.ready(info['BATCH_SIZE']):
        if not cnt%1000:
            print('training', cnt)
        samples = rbuffer.sample_random(info['BATCH_SIZE'], pytorchify=True)
        states, actions, rewards, next_states, ongoing_flags, masks, _ = samples

        opt.zero_grad()
        q_policy_vals = policy_net(states, None)
        #next_q_state_values = target_net(next_states, None)
        next_q_target_vals = target_net(next_states, None)
        next_q_policy_vals = policy_net(next_states, None)
        cnt_losses = []
        for k in range(info['N_ENSEMBLE']):
            #TODO finish masking
            total_used = 1.0
            #total_used = torch.sum(mask_pt[:, k])
            if total_used > 0.0:
                next_q_vals = next_q_target_vals[k].data
                if info['DOUBLE_DQN']:
                    next_actions = next_q_policy_vals[k].data.max(1, True)[1]
                    next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
                else:
                    next_qs = next_q_vals.max(1)[0] # max returns a pair

                targets = rewards + info['GAMMA'] * next_qs * ongoing_flags
                preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
                loss = F.smooth_l1_loss(preds, targets)
                cnt_losses.append(loss)
                losses[k] = loss.cpu().detach().item()

        all_loss = torch.stack(cnt_losses).sum()
        all_loss.backward()
        for param in policy_net.core_net.parameters():
            if param.grad is not None:
                param.grad.data *=1.0/float(info['N_ENSEMBLE'])
        opt.step()
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
        print("SKIPPING SAVE OF BUFFER")
        #rbuffer.save(buff_filename)
        return last_save
    else: return last_save


def run_training_episode(epoch_num, total_steps):
    finished = False
    start = time.time()
    start_steps = total_steps
    episodic_reward = 0.0
    _S = env.reset()
    rbuffer.add_init_state(_S)
    policy_net.train()
    random_state.shuffle(heads)
    active_head = heads[0]
    losses = []
    while not finished:
        #with torch.no_grad():
            #_Spt = torch.Tensor(_S[None]).to(info['DEVICE'])
            #vals = policy_net(_Spt, active_head)
            #action = torch.argmax(vals, dim=1).item()
            ## always do this calculation - as it is used for debugging
            #vals = policy_net(_Spt, None)
            #acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
            #action = acts[active_head]
            #vals = policy_net(_Spt)
            #action = np.argmax(vals.cpu().data.numpy(),-1)[0]
        #board_logger.scalar_summary('time get action per step', total_steps, time.time()-est)
        action = action_getter.pt_get_action(total_steps, state=_S[None], active_head=active_head)
        _S_prime, reward, finished = env.step(action)
        rbuffer.add_experience(next_state=_S_prime[-1], action=action, reward=reward, finished=finished)
        if not total_steps%info['LEARN_EVERY_STEPS']:
            losses.append(train_batch(total_steps))
        else:
            losses.append(0)
        if (not (total_steps%info['TARGET_UPDATE']) and (total_steps > info['MIN_HISTORY_TO_LEARN'])):
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print('updating target network at %s'%total_steps)
            target_net.load_state_dict(policy_net.state_dict())
        _S = _S_prime
        episodic_reward += reward
        total_steps+=1
    stop = time.time()
    ep_time =  stop - start

    steps.append(total_steps)
    episode_step.append(total_steps-start_steps)
    episode_head.append(active_head)
    episode_loss.append(np.mean(losses))
    episode_reward.append(episodic_reward)
    episode_times.append(ep_time)
    episode_relative_times.append(time.time()-info['START_TIME'])
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
        plot_dict_losses({'episode steps':{'index':np.arange(epoch_num+1), 'val':episode_relative_times}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
        plot_dict_losses({'episode head':{'index':np.arange(epoch_num+1), 'val':episode_head}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
        plot_dict_losses({'steps loss':{'index':steps, 'val':episode_loss}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
        plot_dict_losses({'steps reward':{'index':steps, 'val':episode_reward}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
        plot_dict_losses({'episode reward':{'index':np.arange(epoch_num+1), 'val':episode_reward}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
        plot_dict_losses({'episode times':{'index':np.arange(epoch_num+1), 'val':episode_times}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
        plot_dict_losses({'steps avg reward':{'index':steps, 'val':avg_rewards}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
        print('avg reward', avg_rewards[-1])
    return episodic_reward, total_steps, ep_time

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


class ActionGetter():
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=100000,
                 replay_memory_start_size=50000, max_frames=25000000, random_seed=122):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames
        self.random_state = np.random.RandomState(random_seed)

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

    def pt_get_action(self, frame_number, state, active_head=None, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            main_dqn: A DQN object
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2*frame_number + self.intercept_2

        if self.random_state.rand(1) < eps:
            return self.random_state.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                state = torch.Tensor(state).to(info['DEVICE'])
                vals = policy_net(state, active_head)
                if active_head is not None:
                    action = torch.argmax(vals, dim=1).item()
                    return action
                else:
                    # vote
                    acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                    action = most_common(acts)
                    return action


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.pkl replay buffer file full path')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        "GAME":'roms/breakout.bin', # gym prefix
        "DEVICE":device,
        "NAME":'_d3Breakout_BT1_eps', # start files with name
        "DUELING":True,
        "DOUBLE_DQN":True,
        "N_ENSEMBLE":1,
        "LEARN_EVERY_STEPS":4, # should be 1, but is 4 in fg91
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":500000,
        "ADAM_LEARNING_RATE":0.00001,
        "ADAM_EPSILON":1.5e-4,
        "RMS_LEARNING_RATE": 0.0001,
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
        "MAX_FRAMES":30000000,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1,
        "FAKE_ACTION":-3,
        "FAKE_REWARD":-5,
        "NETWORK_INPUT_SIZE":(84,84),
        "START_TIME":time.time()
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
        episode_relative_times = []
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
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

    policy_net.apply(weights_init)
    target_net.load_state_dict(policy_net.state_dict())

    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'], eps=info['ADAM_EPSILON'])


    action_getter = ActionGetter(env.num_actions,
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_frames=info['MAX_FRAMES'], eps_annealing_frames=info['EPSILON_DECAY'])
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])

    rbuffer = ReplayBuffer(max_buffer_size=info['BUFFER_SIZE'],
                           history_size=info['HISTORY_SIZE'],
                           min_sampling_size=info['MIN_HISTORY_TO_LEARN'],
                           num_masks=info['N_ENSEMBLE'],
                           bernoulli_probability=info['BERNOULLI_PROBABILITY'],
                           device=info['DEVICE'])


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
            try:
                rbuffer.load(args.buffer_loadpath)
            except Exception as e:
                print(e)
                print('not able to load from buffer: %s. exit() to continue with empty buffer' %args.buffer_loadpath)
                embed()

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


