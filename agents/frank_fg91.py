from __future__ import print_function
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
from replay_buffer import ReplayBuffer
from dqn_model import EnsembleNet, weights_init
from dqn_utils import seed_everything, write_info_file
#from env import Environment
from glob import glob
sys.path.append('../models')
from lstm_utils import plot_dict_losses
import config
from ae_utils import save_checkpoint


"""
TODO -
does our network perform well when there is only one head? if this is the case,
what is happening to the gradients in multple head situation. how should we
distinguish the multiple head effect from the effect of episilon greedy ?
Check the rainbow way of calculating loss
"""
class ProcessFrame:
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame:frame})

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
        if self.eps_annealing_frames > 0:
            self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
            self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
            self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_steps - self.eps_annealing_frames - self.replay_memory_start_size)
            self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_steps

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
        elif self.eps_annealing_frames > 0:
            # TODO check this
            if frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*frame_number + self.intercept
            elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*frame_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            state = torch.transpose(torch.Tensor(state.astype(np.float)),2,0)[None,:].to(info['DEVICE'])
            vals = policy_net(state, active_head)
            if active_head is not None:
                action = torch.argmax(vals, dim=1).item()
                return eps, action
            else:
                # vote
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                action = most_common(acts)
                return eps, action

    def get_action(self, session, frame_number, state, main_dqn, evaluation=False):
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
        return session.run(main_dqn.best_action, feed_dict={main_dqn.input:[state]})[0]

class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        self.random_state = np.random.RandomState(393)

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

    def _get_valid_indices(self):
        for i in range(self.batch_size):
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

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

###############################################
# from rainbow code

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def one_hot(x, n):
    assert x.dim() == 2
    one_hot_x = torch.zeros(x.size(0), n).cuda()
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x

def target_q_values(states):
    q_vals = target_net(torch.Variable(states, volatile=True), None).data
    return q_vals

def online_q_values(states):
    q_vals = policy_net(torch.Variable(states, volatile=True), None).data
    return q_vals

def compute_targets(rewards, next_states, non_ends, gamma):
    """Compute batch of targets for dqn
    params:
        rewards: Tensor [batch]
        next_states: Tensor [batch, channel, w, h]
        non_ends: Tensor [batch]
        gamma: float
    """
    next_q_vals = target_q_values(next_states)
    if info['DOUBLE_DQN']:
        next_actions = online_q_values(next_states).max(1, True)[1]
        next_actions = one_hot(next_actions, atari.env.action_space.n)
        next_qs = (next_q_vals * next_actions).sum(1)
    else:
        next_qs = next_q_vals.max(1)[0] # max returns a pair

    targets = rewards + gamma * next_qs * non_ends
    return targets

def rainbow_loss(states, actions, targets):
    """
    params:
        states: Variable [batch, channel, w, h]
        actions: Variable [batch, num_actions] one hot encoding
        targets: Variable [batch]
    """
    assert_eq(actions.shape[1], atari.env.action_space.n)
    qs = policy_net(states)
    preds = (qs * actions).sum(1)
    err = nn.functional.smooth_l1_loss(preds, targets)
    return err

#################################

def rainbow_learn(session, states, actions, rewards, next_states, terminal_flags):
    opt.zero_grad()
    states = torch.Tensor(states).transpose(1,3).to(info['DEVICE'])
    next_states = torch.Tensor(next_states).transpose(1,3).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    ongoing_flags = [not x for x in terminal_flags]
    ongoing_flags = torch.Tensor(ongoing_flags.astype(np.int)).to(info['DEVICE'])
    targets = compute_targets(rewards, next_states, ongoing_flags, info['GAMMA'])
    loss = rainbow_loss(states, actions, targets)
    loss.backward()
    opt.step()


def learn(session, main_dqn, target_dqn, states, actions, rewards, new_states, terminal_flags):
    # Draw a minibatch from the replay memory
    #states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(info['BATCH_SIZE']), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (info['GAMMA']*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    return loss

def ptlearn(states, actions, rewards, next_states, terminal_flags):
    """
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
    Returns:
        loss: The loss of the minibatch, for tensorboard
    Draws a minibatch from the replay memory, calculates the
    target Q-value that the prediction Q-value is regressed to.
    Then a parameter update is performed on the main DQN.
    """
    #print(states.max(), states.min())
    states = torch.Tensor(states.astype(np.float)).transpose(1,3).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)).transpose(1,3).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
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
            # clip loss from original lua code - unstable, maybe i am doing
            # something try
            # https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
            #loss = torch.clamp(targets-preds, -1, 1)
            #loss = torch.mean(loss**2)
            #loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            cnt_losses.append(loss)
            #loss.backward(retain_graph=True)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    # with one head - at beginning, loss is 0.114
    core_sum = 0
    core_k_sum = 0
    heads_sum = 0

    for param in policy_net.core_net.parameters():
        if param.grad is not None:
    #        core_sum += param.grad.data.sum()
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    #        core_k_sum += param.grad.data.sum()
    #for head_net in policy_net.net_list:
    #    for param in head_net.parameters():
    #        if param.grad is not None:
    #            #param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    #            heads_sum += param.grad.data.sum()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 5)
    opt.step()
    return np.mean(losses)

def generate_gif(frame_number, frames_for_gif, reward):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)


    if reward > 5:
        gif_fname = os.path.join(model_base_filedir, "ATARI_frame_%010d_reward_%04d.gif"%(frame_number, int(reward)))
        print("WRITING GIF", gif_fname)
        imageio.mimsave(gif_fname, frames_for_gif, duration=1/30)

class Atari:
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, no_op_steps=10, agent_history_length=4, random_seed=293):
        self.env = gym.make(envName)
        self.frame_processor = ProcessFrame()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        self.random_seed = np.random.RandomState(random_seed)

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True # Set to true so that the agent starts
                                  # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(self.random_seed.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, infor = self.env.step(action)

        if infor['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = infor['ale.lives']

        processed_new_frame = self.frame_processor.process(sess, new_frame)   #
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) #
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def train():
    """Contains the training and evaluation loops"""
    frame_number = 0
    rewards = []
    ptloss_list = []
    eval_rewards = []
    eval_steps = []

    eps_list = []
    epoch_num = 0
    while frame_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal_life_lost = atari.reset(sess)
            start_steps = frame_number
            st = time.time()
            episode_reward_sum = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            epoch_num += 1
            ep_eps_list = []
            for totf in range(info['MAX_EPISODE_LENGTH']):
                eps,action = action_getter.pt_get_action(frame_number, state=atari.state, active_head=active_head)
                ep_eps_list.append(eps)
                processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward

                # Store transition in the replay memory
                my_replay_memory.add_experience(action=action,
                                                frame=processed_new_frame[:, :, 0],
                                                reward=reward,
                                                terminal=terminal_life_lost)

                if frame_number % info['LEARN_EVERY_STEPS'] == 0 and frame_number > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags = my_replay_memory.get_minibatch()
                    #tfloss = learn(sess, MAIN_DQN, TARGET_DQN, _states, _actions, _rewards, _next_states, _terminal_flags)
                    ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags)
                    #tfloss_list.append(tfloss)
                    ptloss_list.append(ptloss)
                if frame_number % info['TARGET_UPDATE'] == 0 and frame_number >  info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s'%frame_number)
                    target_net.load_state_dict(policy_net.state_dict())
                    #network_updater.update_networks(sess)

                if terminal:
                    # reached real end of episode (0 lives left) -
                    terminal = False
                    break

            et = time.time()
            ep_time = et-st
            rewards.append(episode_reward_sum)
            #print(epoch_num, "FRAME NUM", frame_number, episode_reward_sum)
            steps.append(frame_number)
            episode_step.append(frame_number-start_steps)
            episode_head.append(active_head)
            eps_list.append(np.mean(ep_eps_list))
            episode_loss.append(np.mean(ptloss_list))
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
                plot_dict_losses({'steps eps':{'index':steps, 'val':eps_list}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
                plot_dict_losses({'steps reward':{'index':steps, 'val':episode_reward}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
                plot_dict_losses({'episode reward':{'index':np.arange(epoch_num), 'val':episode_reward}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
                plot_dict_losses({'episode times':{'index':np.arange(epoch_num), 'val':episode_times}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
                plot_dict_losses({'steps avg reward':{'index':steps, 'val':avg_rewards}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
                print('avg reward', avg_rewards[-1])

            # Output the progress:
            if not epoch_num % 10:
                # Scalar summaries for tensorboard
                if frame_number > info['MIN_HISTORY_TO_LEARN']:
                    summ = sess.run(PERFORMANCE_SUMMARIES,
                                    feed_dict={
                                               PTLOSS_PH:np.mean(ptloss_list),
                                               REWARD_PH:np.mean(rewards[-100:])})

                    SUMM_WRITER.add_summary(summ, frame_number)
                    ptloss_list = []

                    print("Adding tensorboard", len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, np.mean(rewards[-100:]), file=reward_file)
        avg_eval_reward = evaluate(frame_number)
        eval_rewards.append(avg_eval_reward)
        eval_steps.append(frame_number)
        plot_dict_losses({'eval rewards':{'index':eval_steps, 'val':eval_rewards}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)

def evaluate(frame_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    terminal = True
    gif = True
    frames_for_gif = []
    eval_rewards = []
    evaluate_frame_number = 0
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        terminal_life_lost = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
        terminal = False
        episode_steps = 0
        while (not terminal) and (episode_steps < info['MAX_EPISODE_LENGTH']):
            eps,action = action_getter.pt_get_action(frame_number, atari.state, active_head=None, evaluation=True)

            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
            evaluate_frame_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            if gif:
                frames_for_gif.append(new_frame)
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)

        eval_rewards.append(episode_reward_sum)
        gif = False # Save only the first game of the evaluation as a gif

    print("Evaluation score:\n", np.mean(eval_rewards))
    try:
        generate_gif(frame_number, frames_for_gif, eval_rewards[0])
    except IndexError:
        print("No evaluation game finished")

    # Show the evaluation score in tensorboard
    summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
    SUMM_WRITER.add_summary(summ, frame_number)
    efile = os.path.join(model_base_filedir, 'rewardsEval.dat')
    with open(efile, 'a') as eval_reward_file:
        print(frame_number, np.mean(eval_rewards), file=eval_reward_file)
    return np.mean(eval_rewards)

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
        "GAME":'Breakout', # gym prefix
        "DEVICE":device,
        #"NAME":'Dbug_multi_FRANKBreakout_9PTA_init_EPS0BIAS', # start files with name
        "NAME":'DEBUGAtari', # start files with name
        "DUELING":True,
        "DOUBLE_DQN":True,
        "N_ENSEMBLE":9,
        "LEARN_EVERY_STEPS":4, # should be 1, but is 4 in fg91
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "EPS_INITIAL":1.0,
        #"EPS_FINAL":0.01,
        "EPS_FINAL":0.1,
        "EPS_EVAL":0.0,
        "EPS_ANNEALING_FRAMES":1000000,
        #"EPS_ANNEALING_FRAMES":0,
        "EPS_FINAL_FRAME":0.01,
        "NUM_EVAL_EPISODES":1,
        "BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":200000,
        "EVAL_FREQUENCY":200000,
        "ADAM_LEARNING_RATE":0.00001 ,
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
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1,
        "FAKE_ACTION":-3,
        "FAKE_REWARD":-5,
        "NETWORK_INPUT_SIZE":(84,84),
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6),
        "MAX_EPISODE_LENGTH":18000, # Equivalent of 5 minutes of gameplay at 60 frames per second
        }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    accumulation_rewards = []
    overall_time = 0.
    info['load_time'] = datetime.date.today().ctime()
    my_replay_memory = ReplayMemory(size=info['BUFFER_SIZE'], batch_size=info['BATCH_SIZE'])
    #network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

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
    #env = Environment(info['GAME'])
    #action_space = np.arange(env.num_actions)


    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])
    ENV_NAME = '%sDeterministic-v4'%info['GAME']
    # You can increase the learning rate to 0.00025 in Pong for quicker results


    ###########################################
    tf.reset_default_graph()
    NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
    SUMM_WRITER = tf.summary.FileWriter(model_base_filedir)

    atari = Atari(ENV_NAME, NO_OP_STEPS)
    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=atari.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=atari.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

    target_net.load_state_dict(policy_net.state_dict())

    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])
    random_state = np.random.RandomState(info["SEED"])
    #board_logger = TensorBoardLogger(model_base_filedir)
    last_target_update = 0
    all_rewards = []
    print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                    atari.env.unwrapped.get_action_meanings()))
    action_getter = ActionGetter(n_actions=atari.env.action_space.n,
                                 eps_initial=info['EPS_INITIAL'],
                                 eps_final=info['EPS_FINAL'],
                                 eps_final_frame=info['EPS_FINAL_FRAME'],
                                 eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                                 eps_evaluation=info['EPS_EVAL'],
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])


    # Scalar summaries for tensorboard: loss, average reward and evaluation score
    with tf.name_scope('Performance'):
        #TFLOSS_PH = tf.placeholder(tf.float32, shape=None, name='tfloss_summary')
        #TFLOSS_SUMMARY = tf.summary.scalar('tfloss', TFLOSS_PH)
        PTLOSS_PH = tf.placeholder(tf.float32, shape=None, name='ptloss_summary')
        PTLOSS_SUMMARY = tf.summary.scalar('ptloss', PTLOSS_PH)
        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
        EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
        EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

    PERFORMANCE_SUMMARIES = tf.summary.merge([PTLOSS_SUMMARY, REWARD_SUMMARY])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train()

