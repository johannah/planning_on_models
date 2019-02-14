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
            if active_head is not None:
                state = torch.transpose(torch.Tensor(state),2,0)[None,:].to(info['DEVICE'])
                vals = policy_net(state, active_head)
                action = torch.argmax(vals, dim=1).item()
                return action
            else:
                # vote
                state = torch.transpose(torch.Tensor(state),2,0)[None,:].to(info['DEVICE'])
                vals = policy_net(state, None)
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                action = most_common(acts)
                return action

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
                index = random.randint(self.agent_history_length, self.count - 1)
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
    #print('q_vals', q_vals)
    #print('doubleg', double_q)
    #print('tf targetq', target_q)
    #print('tf doubleq', double_q)
    #print('loss', loss)
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
    states = torch.Tensor(states).transpose(1,3).to(info['DEVICE'])
    next_states = torch.Tensor(next_states).transpose(1,3).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    st = time.time()
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
            # clip loss from original lua code - unstable, maybe i am doing
            # something try
            # https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
            #loss = torch.clamp(targets-preds, -1, 1)
            #loss = torch.mean(loss**2)
            loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            cnt_losses.append(loss)
            #loss.backward(retain_graph=True)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    # with one head - at beginning, loss is 0.114
    """"
    # WITH one head in a model that seems to work -
    print(np.sum(losses), core_sum, heads_sum)
    0.15649263560771942 tensor(-53.9158) tensor(-135.9045)
    0.07511568069458008 tensor(49.5142) tensor(-44.2417)
    0.063167504966259 tensor(-59.0345) tensor(-99.5942)
    0.03253338485956192 tensor(89.4640) tensor(-28.0189)
    0.04259048029780388 tensor(68.2152) tensor(-2.7756)
    0.04358198121190071 tensor(397.2363) tensor(153.4211)
    0.04792924225330353 tensor(327.2467) tensor(132.6360)
    """
    """
     with 9 heads and no k scaling
    print(np.sum(losses), core_sum, heads_sum)
    2.2094377912580967 tensor(2607.5942) tensor(-273.4381)
    1.598130401223898 tensor(1483.8365) tensor(-466.0103)
    1.2488596090115607 tensor(731.1756) tensor(-450.2681)
    0.77371034771204 tensor(195.8453) tensor(-720.5662)
    0.7554639400914311 tensor(1115.9290) tensor(-190.6120)
    0.7469962313771248 tensor(1518.5344) tensor(288.0397)
    0.7440020181238651 tensor(1687.2618) tensor(826.8621)
    1.1307623535394669 tensor(3049.7002) tensor(2555.6411)
    1.331430234014988 tensor(3869.8418) tensor(3364.0515)
    1.6879942370578647 tensor(4534.9482) tensor(4111.5322)
    """
    """

     with 9 heads and 1/k scaling of grads
    print(np.sum(losses), core_sum, heads_sum, core_k_sum)
    2.6111491434276104 tensor(2617.0588) tensor(-419.1773) tensor(290.7843)
    1.6717215571552515 tensor(1495.3574) tensor(-692.1512) tensor(166.1508)
    1.2669921685010195 tensor(143.8809) tensor(-1116.1006) tensor(15.9868)
    0.6811386086046696 tensor(63.4201) tensor(-776.9857) tensor(7.0467)
    0.7174982316792011 tensor(247.4868) tensor(-515.8655) tensor(27.4985)
    0.46023492119275033 tensor(1212.9198) tensor(87.3714) tensor(134.7689)
    0.63468979857862 tensor(1709.7010) tensor(830.2244) tensor(189.9668)
    1.2908978443592787 tensor(3511.1318) tensor(2659.7249) tensor(390.1258)
    1.4473802708089352 tensor(4192.8340) tensor(3365.2498) tensor(465.8704)
    """
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
    #nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    #print(np.sum(losses), core_sum, heads_sum, core_k_sum)
    opt.step()
    #board_logger.scalar_summary('batch train time per cnt', cnt, time.time()-st)
    #board_logger.scalar_summary('loss per cnt', cnt, np.mean(losses))
    return np.mean(losses)

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)


    if reward > 5:
        gif_fname = os.path.join(path, "ATARI_frame_%010d_reward_%04d.gif"%(frame_number, int(reward)))
        print("WRITING GIF", gif_fname)
        imageio.mimsave(gif_fname, frames_for_gif, duration=1/30)
    #imageio.mimsave('{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
    #                frames_for_gif, duration=1/30)

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
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=info['BATCH_SIZE'])
    #network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES, eps_annealing_frames=info['EPSILON_DECAY'])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #sess.run(init)

        frame_number = 0
        rewards = []
        tfloss_list = []
        ptloss_list = []

        epoch_num = 0
        while frame_number < MAX_FRAMES:
            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                start_steps = frame_number
                st = time.time()
                episode_reward_sum = 0
                random_state.shuffle(heads)
                active_head = heads[0]
                epoch_num += 1
                for totf in range(MAX_EPISODE_LENGTH):
                    action = action_getter.pt_get_action(frame_number, state=atari.state, active_head=active_head)
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
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number >  info['MIN_HISTORY_TO_LEARN']:
                        print("++++++++++++++++++++++++++++++++++++++++++++++++")
                        print('updating target network at %s'%frame_number)
                        target_net.load_state_dict(policy_net.state_dict())
                        #network_updater.update_networks(sess)

                    if terminal:
                        terminal = False
                        break

                et = time.time()
                ep_time = et-st
                rewards.append(episode_reward_sum)
                #print(epoch_num, "FRAME NUM", frame_number, episode_reward_sum)
                steps.append(frame_number)
                episode_step.append(frame_number-start_steps)
                episode_head.append(active_head)
                episode_loss.append(np.mean(ptloss_list))
                episode_reward.append(episode_reward_sum)
                episode_times.append(ep_time)
                episode_relative_times.append(time.time()-info['START_TIME'])
                avg_rewards.append(np.mean(rewards[-100:]))
                if not epoch_num%10:
                    # TODO plot title
                    plot_dict_losses({'episode steps':{'index':np.arange(epoch_num), 'val':episode_step}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
                    plot_dict_losses({'episode steps':{'index':np.arange(epoch_num), 'val':episode_relative_times}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
                    plot_dict_losses({'episode head':{'index':np.arange(epoch_num), 'val':episode_head}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
                    plot_dict_losses({'steps loss':{'index':steps, 'val':episode_loss}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
                    plot_dict_losses({'steps reward':{'index':steps, 'val':episode_reward}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
                    plot_dict_losses({'episode reward':{'index':np.arange(epoch_num), 'val':episode_reward}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
                    plot_dict_losses({'episode times':{'index':np.arange(epoch_num), 'val':episode_times}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
                    plot_dict_losses({'steps avg reward':{'index':steps, 'val':avg_rewards}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
                    print('avg reward', avg_rewards[-1])

                # Output the progress:
                if not epoch_num % 10:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={
                                                   PTLOSS_PH:np.mean(ptloss_list),
                                                   REWARD_PH:np.mean(rewards[-100:])})


                        #summ = sess.run(PERFORMANCE_SUMMARIES,
                        #                feed_dict={TFLOSS_PH:np.mean(tfloss_list),
                        #                           PTLOSS_PH:np.mean(ptloss_list),
                        #                           REWARD_PH:np.mean(rewards[-100:])})

                        SUMM_WRITER.add_summary(summ, frame_number)
                        tfloss_list = []
                        ptloss_list = []
                    # Histogramm summaries for tensorboard
                    #summ_param = sess.run(PARAM_SUMMARIES)
                    #SUMM_WRITER.add_summary(summ_param, frame_number)

                    print("TO", len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, np.mean(rewards[-100:]), file=reward_file)

            #########################
            ####### Evaluation ######
            #########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False

                # Fire (action 1), when a life was lost or the game just started,
                # so that the agent does not stand around doing nothing. When playing
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else action_getter.pt_get_action(frame_number,
                                                                               atari.state,
                                                                                 active_head=None,
                                                                               evaluation=True)

                #action = 1 if terminal_life_lost else action_getter.get_action(sess, frame_number,
                #                                                               atari.state,
                #                                                               MAIN_DQN,
                #                                                               evaluation=True)
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif:
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif

            print("Evaluation score:\n", np.mean(eval_rewards))
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], SPATH)
            except IndexError:
                print("No evaluation game finished")

            #Save the network parameters
            #saver.save(sess, os.path.join(PATH, 'my_model'), global_step=frame_number)
            frames_for_gif = []

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)

if __name__ == '__main__':
    # original taks ~ 3 hours to get to 2mill frames and avg value is 20

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
        #"GAME":'roms/pong.bin', # gym prefix
        "GAME":'Breakout', # gym prefix
        "DEVICE":device,
        "NAME":'DEBUGFRANKBreakout_9PTA_init', # start files with name
        "DUELING":True,
        "DOUBLE_DQN":True,
        "N_ENSEMBLE":9,
        "LEARN_EVERY_STEPS":4, # should be 1, but is 4 in fg91
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target
        #"MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "MIN_HISTORY_TO_LEARN":500, # in environment frames
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
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
    #env = Environment(info['GAME'])
    #action_space = np.arange(env.num_actions)


    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])


    ENV_NAME = '%sDeterministic-v4'%info['GAME']
    #ENV_NAME = 'PongDeterministic-v4'
    # You can increase the learning rate to 0.00025 in Pong for quicker results
    TRAIN = True


    ###########################################
    tf.reset_default_graph()

    # Control parameters
    MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
    EVAL_STEPS = 10000               # Number of frames for one evaluation
    NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                     # According to Mnih et al. 2015 this is measured in the number of
                                     # parameter updates (every four actions), however, in the
                                     # DeepMind code, it is clearly measured in the number
                                     # of actions the agent choses
    #DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
    #REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
    REPLAY_MEMORY_START_SIZE = info['MIN_HISTORY_TO_LEARN'] # Number of completely random actions,
                                     # before the agent starts learning
    MAX_FRAMES = 30000000            # Total number of frames the agent sees
    MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
    NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                     # evaluation episode
    #UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
    HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                     # has the shape (1,1,1024) which is split into two streams. Both
                                     # the advantage stream and value stream have the shape
                                     # (1,1,512). This is slightly different from the original
                                     # implementation but tests I did with the environment Pong
                                     # have shown that this way the score increases more quickly
    #LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
                                     # Hessel et al. 2017 used 0.0000625
    #BS = 32                          # Batch size

    PATH = "output/"                 # Gifs and checkpoints will be saved here
    SUMMARIES = "summaries"          # logdir for tensorboard
    RUNID = os.path.split(model_base_filedir)[1]
    #RUNID = 'run_frank'
    #RUNID = 'frank1h'
    #os.makedirs(PATH, exist_ok=True)
    #os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    SPATH = os.path.join(SUMMARIES, RUNID)
    if not os.path.exists(SPATH):
        os.makedirs(SPATH)
    print("WRITING TO DIR")
    SUMM_WRITER = tf.summary.FileWriter(SPATH)

    atari = Atari(ENV_NAME, NO_OP_STEPS)
    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=atari.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=atari.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

   # policy_net.apply(weights_init)
    target_net.load_state_dict(policy_net.state_dict())

    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])


#    rbuffer = ReplayBuffer(max_buffer_size=info['BUFFER_SIZE'],
#                           history_size=info['HISTORY_SIZE'],
#                           min_sampling_size=info['MIN_HISTORY_TO_LEARN'],
#                           num_masks=info['N_ENSEMBLE'],
#                           bernoulli_probability=info['BERNOULLI_PROBABILITY'],
#                           device=info['DEVICE'])
#
#
    random_state = np.random.RandomState(info["SEED"])
    #board_logger = TensorBoardLogger(model_base_filedir)
    last_target_update = 0
    all_rewards = []

#    print("Starting training")
#    for epoch_num in range(epoch_start, info['N_EPOCHS']):
#        ep_reward, total_steps, etime = run_training_episode(epoch_num, total_steps)
#        all_rewards.append(ep_reward)
#        overall_time += etime
#        last_mean = np.mean(all_rewards[-100:])
#        #board_logger.scalar_summary("avg reward last 100 episodes", epoch_num, last_mean)
#        last_save = handle_checkpoint(last_save, total_steps, epoch_num, last_mean)


    print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                    atari.env.unwrapped.get_action_meanings()))

    # main DQN and target DQN networks:
    #with tf.variable_scope('mainDQN'):
    #    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, info['ADAM_LEARNING_RATE'])   #
    #with tf.variable_scope('targetDQN'):
    #    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               #

    #init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #init = tf.global_variables_initializer()
    #saver = tf.train.Saver()


    #MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    #TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

    #LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage",
    #             "denseAdvantageBias", "denseValue", "denseValueBias"]

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
    #PERFORMANCE_SUMMARIES = tf.summary.merge([TFLOSS_SUMMARY, PTLOSS_SUMMARY, REWARD_SUMMARY])

    # Histogramm summaries for tensorboard: parameters
    #with tf.name_scope('Parameters'):
    #    ALL_PARAM_SUMMARIES = []
    #    for i, Id in enumerate(LAYER_IDS):
    #        with tf.name_scope('mainDQN/'):
    #            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
    #        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
    #PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)

    if TRAIN:
        train()
    else:
        gif_path = "GIF/"
        os.makedirs(gif_path,exist_ok=True)

        if ENV_NAME == 'BreakoutDeterministic-v4':
            trained_path = "trained/breakout/"
            save_file = "my_model-15845555.meta"

        elif ENV_NAME == 'PongDeterministic-v4':
            trained_path = "trained/pong/"
            save_file = "my_model-3217770.meta"
