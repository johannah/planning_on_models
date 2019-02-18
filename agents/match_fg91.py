import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from __future__ import print_function
"""
Implementation of DeepMind's Deep Q-Learning by Fabio M. Graetz, 2018
If you have questions or suggestions, write me a mail fabiograetzatgooglemaildotcom
"""
import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize

import sys
import numpy as np
from IPython import embed

from collections import Counter
import math
#from logger import TensorBoardLogger
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
#from replay_buffer import ReplayBuffer
from dqn_model import EnsembleNet, weights_init
from dqn_utils import seed_everything, write_info_file
from env import Environment
from glob import glob
sys.path.append('../models')
import config
from ae_utils import save_checkpoint
import cv2

"""
TODO -
does our network perform well when there is only one head? if this is the case,
what is happening to the gradients in multple head situation. how should we
distinguish the multiple head effect from the effect of episilon greedy ?
Check the rainbow way of calculating loss
"""

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, info['NETWORK_INPUT_SIZE'], cv2.INTER_NEAREST)
    return output

class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=100000,
                 replay_memory_start_size=50000, max_steps=25000000, random_seed=122):
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
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_steps - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_steps

    def pt_get_action(self, step_number, state, active_head=None, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            step_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            main_dqn: A DQN object
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.eps_annealing_frames > 0:
            if step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*step_number + self.intercept
            if step_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*step_number + self.intercept_2
        else:
            eps = self.eps_final

        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)

        else:
            #state = torch.Tensor(state.astype(np.float32)/255.0)[None,:].to(info['DEVICE'])
            state = torch.Tensor(state.astype(np.float32))[None,:].to(info['DEVICE'])
            #print('ga',state.max(), state.min())
            vals = policy_net(state, active_head)
            if active_head is not None:
                action = torch.argmax(vals, dim=1).item()
                return eps,action
            else:
                # vote
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                action = most_common(acts)
                return eps,action
class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, random_seed=300):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.count = 0
        self.current = 0

        self.random_state = np.random.RandomState(random_seed)
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        self.batch_size = 32
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        #print('add experience', action, frame.dtype, frame.sum(), frame.max(), frame.min())
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self, batch_size):
        if self.indices.shape[0] != batch_size:
            self.indices = np.empty(self.batch_size, dtype=np.int32)

        for i in range(batch_size):
            while True:
                index = self.random_state.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self, batch_size):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < batch_size + self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        if self.states.shape[0] != batch_size:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)
            self.indices = np.empty(batch_size, dtype=np.int32)

        self._get_valid_indices(batch_size)
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices]

def ptlearn(states, actions, rewards, next_states, terminal_flags):
    #states = torch.Tensor(states.astype(np.float)/255.0).to(info['DEVICE'])
    #next_states = torch.Tensor(next_states.astype(np.float)/255.0).to(info['DEVICE'])
    states = torch.Tensor(states.astype(np.float)).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)).to(info['DEVICE'])
    #print('Ls',states.max(), states.min())
    #print('Lns',next_states.max(), next_states.min())
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    #samples = rbuffer.sample_random(info['BATCH_SIZE'], pytorchify=True)
    #states, actions, rewards, next_states, ongoing_flags, masks, _ = samples

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

            preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
            targets = rewards + info['GAMMA'] * next_qs * (1-terminal_flags)
            loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    # with one head - at beginning, loss is 0.114
    core_sum = 0
    core_k_sum = 0
    heads_sum = 0

    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            core_sum += param.grad.data.sum()
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
            core_k_sum += param.grad.data.sum()
    for head_net in policy_net.net_list:
        for param in head_net.parameters():
            if param.grad is not None:
                #param.grad.data *=1.0/float(info['N_ENSEMBLE'])
                heads_sum += param.grad.data.sum()
    opt.step()
    return loss.item()

def generate_gif(step_number, frames_for_gif, reward, name=''):
    """
        Args:
            step_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    if len(frames_for_gif[0].shape) == 3:
        for idx, frame_idx in enumerate(frames_for_gif):
            frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

        gif_fname = os.path.join(model_base_filedir, "ATARI_frame_%010d_reward_%04d_color%s.gif"%(step_number, int(reward), name))
    else:
        for idx, frame_idx in enumerate(frames_for_gif):
            frames_for_gif[idx] = resize(frame_idx, (420, 320), preserve_range=True, order=0).astype(np.uint8)
        gif_fname = os.path.join(model_base_filedir, "ATARI_frame_%010d_reward_%04d_gray%s.gif"%(step_number, int(reward), name))

    print("WRITING GIF", gif_fname)
    imageio.mimsave(gif_fname, frames_for_gif, duration=1/30)

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def train_and_eval():
    """Contains the training and evaluation loops"""
    step_number = 0
    rewards = []
    eval_rewards = []
    eval_steps = []
    ptloss = 0
    eps_list = []
    epoch_num = 0
    life_lost = False
    while step_number < info['TOTAL_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_step = 0
        while epoch_step < info['EVAL_FREQUENCY']:
            state = env.reset()
            start_steps = step_number
            ep_eps_list = []
            st = time.time()
            episode_reward_sum = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            epoch_num += 1
            #plot_states = []
            #plot_color_states = []
            for totf in range(info['MAX_EPISODE_LENGTH_STEPS']):
                if life_lost:
                    action = 1
                    eps = 0
                else:
                    eps, action = action_getter.pt_get_action(step_number, state=state, active_head=active_head)
                #plot_states.append(state[-1])
                #plot_color_states.append(env.ale.getScreenRGB())
                ep_eps_list.append(eps)
                #next_state, processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(action)
                next_state, reward, life_lost, terminal = env.step(action)
                step_number += 1
                epoch_step += 1
                episode_reward_sum += reward

                # Store transition in the replay memory
                my_replay_memory.add_experience(action=action,
                                                frame=state[-1],
                                                reward=reward,
                                                terminal=life_lost)

                if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags = my_replay_memory.get_minibatch(info['BATCH_SIZE'])
                    ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags)
                if step_number % info['TARGET_UPDATE_FREQUENCY'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s'%step_number)
                    target_net.load_state_dict(policy_net.state_dict())

                if terminal:
                    terminal = False
                    break
                state = next_state


            #generate_gif(step_number, plot_states, episode_reward_sum, name='train')
            #generate_gif(step_number, plot_color_states, episode_reward_sum, name='train')
            et = time.time()
            ep_time = et-st
            rewards.append(episode_reward_sum)
            steps.append(step_number)
            episode_step.append(step_number-start_steps)

            eps_list.append(np.mean(ep_eps_list))
            episode_head.append(active_head)
            episode_loss.append(ptloss)
            episode_reward.append(episode_reward_sum)
            episode_times.append(ep_time)
            episode_relative_times.append(time.time()-info['START_TIME'])
            avg_rewards.append(np.mean(rewards[-100:]))
            if not epoch_num%50:
                # TODO plot title
                plot_dict_losses({'episode steps':{'index':np.arange(epoch_num), 'val':episode_step}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
                plot_dict_losses({'episode steps':{'index':np.arange(epoch_num), 'val':episode_relative_times}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
                plot_dict_losses({'episode head':{'index':np.arange(epoch_num), 'val':episode_head}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
                plot_dict_losses({'steps loss':{'index':steps, 'val':episode_loss}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
                plot_dict_losses({'steps reward':{'index':steps, 'val':episode_reward}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
                plot_dict_losses({'episode reward':{'index':np.arange(epoch_num), 'val':episode_reward}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
                plot_dict_losses({'episode times':{'index':np.arange(epoch_num), 'val':episode_times}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
                plot_dict_losses({'steps avg reward':{'index':steps, 'val':avg_rewards}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
                plot_dict_losses({'steps eps':{'index':steps, 'val':eps_list}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)

                print('avg reward', avg_rewards[-1])
                #generate_gif(step_number, env.plot_frames, episode_reward_sum, name='trainENV')
                #generate_gif(step_number, env.gray_plot_frames, episode_reward_sum, name='trainENV')

                # Scalar summaries for tensorboard
                if step_number > info['MIN_HISTORY_TO_LEARN']:
                    summ = sess.run(PERFORMANCE_SUMMARIES,
                                    feed_dict={
                                               PTLOSS_PH:np.mean(ptloss),
                                               REWARD_PH:np.mean(rewards[-100:])})

                    SUMM_WRITER.add_summary(summ, step_number)
                with open('rewards.dat', 'a') as reward_file:
                    print(len(rewards), step_number, np.mean(rewards[-100:]), file=reward_file)
        avg_eval_reward = evaluate(step_number)
        eval_rewards.append(avg_eval_reward)
        eval_steps.append(step_number)
        plot_dict_losses({'eval rewards':{'index':eval_steps, 'val':eval_rewards}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)


def evaluate(step_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    life_lost = terminal = True
    gif = True
    frames_for_gif = []
    gray_frames_for_gif = []
    eval_rewards = []
    evaluate_step_number = 0
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state  = env.reset()
        episode_reward_sum = 0
        terminal = False
        episode_steps = 0
        while not terminal or episode_steps < info['MAX_STEPS']:
            # Fire (action 1), when a life was lost or the game just started,
            # so that the agent does not stand around doing nothing. When playing
            # with other environments, you might want to change this...
            if life_lost:
                action = 1
                eps = 0
            else:
                eps, action = action_getter.pt_get_action(step_number, state, active_head=None, evaluation=True)

            episode_steps +=1
            next_state,  reward, life_lost, terminal = env.step(action)
            evaluate_step_number += 1
            episode_reward_sum += reward
            state = next_state
            #if gif:
            #    gray_frames_for_gif.append(state[-1])
            #    frames_for_gif.append(env.ale.getScreenRGB())
        eval_rewards.append(episode_reward_sum)
        gif = False # Save only the first game of the evaluation as a gif

    print("Evaluation score:\n", np.mean(eval_rewards))
    try:

        generate_gif(step_number, env.plot_frames, eval_rewards[-1], name='eval')
        generate_gif(step_number, env.gray_plot_frames, eval_rewards[-1], name='eval')
        #generate_gif(step_number, frames_for_gif, eval_rewards[0])
        #generate_gif(step_number, gray_frames_for_gif, eval_rewards[0])
    except IndexError:
        print("No evaluation game finished")

    # Show the evaluation score in tensorboard
    summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
    SUMM_WRITER.add_summary(summ, step_number)
    efile = os.path.join(model_base_filedir, 'rewards_eval.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), file=eval_reward_file)
    return np.mean(eval_rewards)

if __name__ == '__main__':
    # original taks ~ 3 hours to get to 2mill frames and avg value is 20

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-g', '--gif', action='store_true', default=True)
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
        #"GAME":'Breakout', # gym prefix
        "DEVICE":device,
        #"NAME":'Dbug_multi_FRANKBreakout_9PTA_init', # start files with name
        #"NAME":'matchROMSFRANK9', # start files with name
        "NAME":'DEBUGmatchRMSROMSFRANK9', # start files with name
        #"NAME":'DEBUGmatch9', # start files with name
        "DUELING":True,
        "DOUBLE_DQN":True,
        "N_ENSEMBLE":9,
        "LEARN_EVERY_STEPS":4, # should be 1, but is 4 in fg91
        "EVAL_FREQUENCY":500000,
        "NUM_EVAL_EPISODES":1,
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE_FREQUENCY":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "BUFFER_SIZE":1000000, # Buffer size for experience replay
        "EPS_INITIAL":1.0,
        "EPS_FINAL":0.01,
        #"EPS_FINAL":0.1,
        "EPS_EVAL":0.01,
        "EPS_ANNEALING_FRAMES":1000000,
        #"EPS_ANNEALING_FRAMES":0,
        "EPS_FINAL_FRAME":0.01,
        "CHECKPOINT_EVERY_STEPS":200000,
        "ADAM_LEARNING_RATE":0.00001 ,
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
        "MAX_EPISODE_LENGTH_STEPS":18000,
        "TOTAL_STEPS":50000000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "EPSILON_MAX":1.0, # Epsilon greedy exploration ~prob of random action, 0. disables
        "EPSILON_MIN":.1,
        "EPSILON_DECAY":1000000,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1,
        "MAX_STEPS":18000,
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
    #ENV_NAME = '%sDeterministic-v4'%info['GAME']

    ###########################################
    tf.reset_default_graph()

    # Control parameters
    EVAL_STEPS = 10000               # Number of frames for one evaluation
    NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                     # evaluation episode
    SUMM_WRITER = tf.summary.FileWriter(model_base_filedir)
    #atari = Atari(ENV_NAME, NO_OP_STEPS)
    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

    target_net.load_state_dict(policy_net.state_dict())
    #opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])
    opt = optim.RMSprop(policy_net.parameters(),
                        lr=info["RMS_LEARNING_RATE"],
                        momentum=info["RMS_MOMENTUM"],
                        eps=info["RMS_EPSILON"],
                        centered=info["RMS_CENTERED"],
                        alpha=info["RMS_DECAY"])
#    rbuffer = ReplayBuffer(max_buffer_size=info['BUFFER_SIZE'],
#                           history_size=info['HISTORY_SIZE'],
#                           min_sampling_size=info['MIN_HISTORY_TO_LEARN'],
#                           num_masks=info['N_ENSEMBLE'],
#                           bernoulli_probability=info['BERNOULLI_PROBABILITY'],
#                           device=info['DEVICE'])
#
#
    random_state = np.random.RandomState(info["SEED"])
    last_target_update = 0
    all_rewards = []

    # Scalar summaries for tensorboard: loss, average reward and evaluation score
    with tf.name_scope('Performance'):
        PTLOSS_PH = tf.placeholder(tf.float32, shape=None, name='ptloss_summary')
        PTLOSS_SUMMARY = tf.summary.scalar('ptloss', PTLOSS_PH)
        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
        EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
        EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

    PERFORMANCE_SUMMARIES = tf.summary.merge([PTLOSS_SUMMARY, REWARD_SUMMARY])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    my_replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                    frame_height=info['NETWORK_INPUT_SIZE'][0],
                                    frame_width=info['NETWORK_INPUT_SIZE'][1],
                                    agent_history_length=info['HISTORY_SIZE'],
                                    random_seed=info['SEED']+10)

    action_getter = ActionGetter(n_actions=env.num_actions,
                                 eps_initial=info['EPS_INITIAL'],
                                 eps_final=info['EPS_FINAL'],
                                 eps_final_frame=info['EPS_FINAL_FRAME'],
                                 eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                                 eps_evaluation=info['EPS_EVAL'],
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_and_eval()
