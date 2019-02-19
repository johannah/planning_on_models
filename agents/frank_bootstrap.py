from __future__ import print_function
"""
Implementation of DeepMind's Deep Q-Learning by Fabio M. Graetz, 2018
If you have questions or suggestions, write me a mail fabiograetzatgooglemaildotcom
"""
import os
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
from dqn_model import EnsembleNet
from dqn_utils import seed_everything, write_info_file
from env import Environment
sys.path.append('../models')
from lstm_utils import plot_dict_losses
import config
from ae_utils import save_checkpoint

def matplotlib_plot_all(p):
    epoch_num = len(p['steps'])
    epochs = np.arange(epoch_num)
    steps = p['steps']
    plot_dict_losses({'episode steps':{'index':epochs, 'val':p['episode_step']}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
    plot_dict_losses({'episode steps':{'index':epochs, 'val':p['episode_relative_times']}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
    plot_dict_losses({'episode head':{'index':epochs,  'val':p['episode_head']}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
    plot_dict_losses({'steps loss':{'index':steps,     'val':p['episode_loss']}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
    plot_dict_losses({'steps eps':{'index':steps,      'val':p['eps_list']}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
    plot_dict_losses({'steps reward':{'index':steps,   'val':p['episode_reward']}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
    plot_dict_losses({'episode reward':{'index':epochs, 'val':p['episode_reward']}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
    plot_dict_losses({'episode times':{'index':epochs,  'val':p['episode_times']}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
    plot_dict_losses({'steps avg reward':{'index':steps,'val':p['avg_rewards']}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
    plot_dict_losses({'eval rewards':{'index':p['eval_steps'], 'val':p['eval_rewards']}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)

def handle_checkpoint(last_save, cnt):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        st = time.time()
        print("beginning checkpoint", st)
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'policy_net_state_dict':policy_net.state_dict(),
                 'target_net_state_dict':target_net.state_dict(),
                 'perf':perf,
                }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
        save_checkpoint(state, filename)
        # npz will be added
        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer"%cnt)
        replay_memory.save_buffer(buff_filename)
        print("finished checkpoint", time.time()-st)
        return last_save
    else: return last_save

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
            # TODO check this
            if step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*step_number + self.intercept
            elif step_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*step_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            #state = torch.transpose(torch.Tensor(state.astype(np.float)),2,0)[None,:].to(info['DEVICE'])
            state = torch.Tensor(state.astype(np.float)/255.)[None,:].to(info['DEVICE'])
            vals = policy_net(state, active_head)
            if active_head is not None:
                action = torch.argmax(vals, dim=1).item()
                return eps, action
            else:
                # vote
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                action = most_common(acts)
                return eps, action

class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32, num_heads=1, bernoulli_probability=1.0):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
            num_heads: integer number of heads needed in mask
            bernoulli_probability: bernoulli probability that an experience will go to a particular head
        """
        self.bernoulli_probability = bernoulli_probability
        assert(self.bernoulli_probability > 0)
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.count = 0
        self.current = 0
        self.num_heads = num_heads
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.masks = np.empty((self.size, self.num_heads), dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(batch_size, dtype=np.int32)
        self.random_state = np.random.RandomState(393)

    def save_buffer(self, filepath):
        st = time.time()
        print("starting save of buffer to %s"%filepath, st)
        np.savez(filepath,
                 frames=self.frames, actions=self.actions, rewards=self.rewards,
                 terminal_flags=self.terminal_flags, masks=self.masks,
                 count=self.count, current=self.current,
                 agent_history_length=self.agent_history_length,
                 frame_height=self.frame_height, frame_width=self.frame_width,
                 num_heads=self.num_heads, bernoulli_probability=self.bernoulli_probability,
                 )
        print("finished saving buffer", time.time()-st)

    def load_buffer(self, filepath):
        st = time.time()
        print("starting load of buffer from %s"%filepath, st)
        npfile = np.load(filepath)
        self.frames = npfile['frames']
        self.actions = npfile['actions']
        self.rewards = npfile['rewards']
        self.terminal_flags = npfile['terminal_flags']
        self.masks = npfile['masks']
        self.count = npfile['count']
        self.current = npfile['current']
        self.agent_history_length = npfile['agent_history_length']
        self.frame_height = npfile['frame_height']
        self.frame_width = npfile['frame_width']
        self.num_heads = npfile['num_heads']
        self.bernoulli_probability = npfile['bernoulli_probability']

        print("finished loading buffer", time.time()-st)
        print("loaded buffer current is", self.current)

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
        mask = self.random_state.binomial(1, self.bernoulli_probability, self.num_heads)
        self.masks[self.current] = mask
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size


    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self, batch_size):
        if batch_size != self.indices.shape[0]:
             self.indices = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            while True:
                index = self.random_state.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # dont add if there was a terminal flag in previous
                # history_length steps
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self, batch_size):
        """
        Returns a minibatch of batch_size
        """
        if batch_size != self.states.shape[0]:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)

        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices(batch_size)

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.masks[self.indices]

def ptlearn(states, actions, rewards, next_states, terminal_flags, masks):
    states = torch.Tensor(states.astype(np.float)/255.).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)/255.).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    opt.zero_grad()
    q_policy_vals = policy_net(states, None)
    next_q_target_vals = target_net(next_states, None)
    next_q_policy_vals = policy_net(next_states, None)
    cnt_losses = []
    for k in range(info['N_ENSEMBLE']):
        #TODO finish masking
        total_used = torch.sum(masks[:,k])
        if total_used > 0.0:
            next_q_vals = next_q_target_vals[k].data
            if info['DOUBLE_DQN']:
                next_actions = next_q_policy_vals[k].data.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
            else:
                next_qs = next_q_vals.max(1)[0] # max returns a pair

            preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
            targets = rewards + info['GAMMA'] * next_qs * (1-terminal_flags)
            # clip loss from original lua code - seems unstable here...
            # https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
            #loss = torch.clamp(targets-preds, -1, 1)
            #loss = torch.mean(loss**2)

            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            full_loss = masks[:,k]*l1loss
            loss = torch.sum(full_loss/total_used)
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    return np.mean(losses)

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

def train(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal = False
            life_lost = True
            state = env.reset()
            start_steps = step_number
            st = time.time()
            episode_reward_sum = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            epoch_num += 1
            ep_eps_list = []
            ptloss_list = []
            while not terminal:
                if life_lost:
                    action = 1
                    eps = 0
                else:
                    eps,action = action_getter.pt_get_action(step_number, state=state, active_head=active_head)
                ep_eps_list.append(eps)
                next_state, reward, life_lost, terminal = env.step(action)
                # Store transition in the replay memory
                replay_memory.add_experience(action=action,
                                                frame=next_state[-1],
                                                reward=np.sign(reward),
                                                terminal=life_lost)

                step_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                state = next_state

                if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags, _masks = replay_memory.get_minibatch(info['BATCH_SIZE'])
                    ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags, _masks)
                    ptloss_list.append(ptloss)
                if step_number % info['TARGET_UPDATE'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s'%step_number)
                    target_net.load_state_dict(policy_net.state_dict())

            et = time.time()
            ep_time = et-st
            perf['steps'].append(step_number)
            perf['episode_step'].append(step_number-start_steps)
            perf['episode_head'].append(active_head)
            perf['eps_list'].append(np.mean(ep_eps_list))
            perf['episode_loss'].append(np.mean(ptloss_list))
            perf['episode_reward'].append(episode_reward_sum)
            perf['episode_times'].append(ep_time)
            perf['episode_relative_times'].append(time.time()-info['START_TIME'])
            perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))
            last_save = handle_checkpoint(last_save, step_number)

            if not epoch_num%50 and step_number > info['MIN_HISTORY_TO_LEARN']:
                # TODO plot title
                print('avg reward', perf['avg_rewards'][-1])

                # Scalar summaries for tensorboard
                summ = sess.run(PERFORMANCE_SUMMARIES,
                                feed_dict={
                                           PTLOSS_PH:np.mean(ptloss_list),
                                           REWARD_PH:np.mean(perf['avg_rewards'][-1])})

                SUMM_WRITER.add_summary(summ, step_number)
                print("Adding tensorboard", len(perf['episode_reward']), step_number, perf['avg_rewards'][-1])
                with open('rewards.txt', 'a') as reward_file:
                    print(len(perf['episode_reward']), step_number, perf['avg_rewards'][-1], file=reward_file)
        avg_eval_reward = evaluate(step_number)
        perf['eval_rewards'].append(avg_eval_reward)
        perf['eval_steps'].append(step_number)
        matplotlib_plot_all(perf)

def evaluate(step_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    frames_for_gif = []
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        while not terminal:
            if life_lost:
                action = 1
            else:
                eps,action = action_getter.pt_get_action(step_number, state, active_head=None, evaluation=True)
            next_state, reward, life_lost, terminal = env.step(action)
            evaluate_step_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            if not i:
                # only save first
                frames_for_gif.append(env.ale.getScreenRGB())
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
        eval_rewards.append(episode_reward_sum)

    print("Evaluation score:\n", np.mean(eval_rewards))
    generate_gif(step_number, frames_for_gif, eval_rewards[0], name='test')

    # Show the evaluation score in tensorboard
    summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
    SUMM_WRITER.add_summary(summ, step_number)
    efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), file=eval_reward_file)
    return np.mean(eval_rewards)

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
        "GAME":'roms/breakout.bin', # gym prefix
        "DEVICE":device,
        "NAME":'FRANKbootstrap_norm', # start files with name
        "DUELING":True,
        "DOUBLE_DQN":True,
        "N_ENSEMBLE":9,
        "LEARN_EVERY_STEPS":4, # should be 1, but is 4 in fg91
        "BERNOULLI_PROBABILITY": 1., # Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE":10000, # how often to update target network
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "EPS_INITIAL":1.0,
        "EPS_FINAL":0.1,
        "EPS_EVAL":0.0,
        "EPS_ANNEALING_FRAMES":int(1e6),
        "EPS_FINAL_FRAME":0.01,
        "NUM_EVAL_EPISODES":1,
        "BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":500000,
        "EVAL_FREQUENCY":250000,
        "ADAM_LEARNING_RATE":6.25e-5,
        "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "EPSILON_MAX":1.0, # Epsilon greedy exploration ~prob of random action, 0. disables
        "EPSILON_MIN":.1,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1,
        "NETWORK_INPUT_SIZE":(84,84),
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS":18000, # Equivalent of 5 minutes of gameplay at 60 frames per second
        "FRAME_SKIP":4,
        "MAX_NO_OP_FRAMES":30,
        "DEAD_AS_END":True,
        }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    # create replay buffer
    replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                 frame_height=info['NETWORK_INPUT_SIZE'][0],
                                 frame_width=info['NETWORK_INPUT_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'])

    random_state = np.random.RandomState(info["SEED"])
    action_getter = ActionGetter(n_actions=env.num_actions,
                                 eps_initial=info['EPS_INITIAL'],
                                 eps_final=info['EPS_FINAL'],
                                 eps_final_frame=info['EPS_FINAL_FRAME'],
                                 eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                                 eps_evaluation=info['EPS_EVAL'],
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])

    if args.model_loadpath != '':
        # load data from loadpath - save model load for later. we need some of
        # these parameters to setup other things
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        info['DEVICE'] = device
        # set a new random seed
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        start_step_number = start_last_save = model_dict['cnt']
        info['loaded_from'] = args.model_loadpath
        perf = model_dict['perf']
        start_step_number = perf['steps'][-1]
    else:
        # create new project
        perf = {'steps':[],
                'avg_rewards':[],
                'episode_step':[],
                'episode_head':[],
                'eps_list':[],
                'episode_loss':[],
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

    target_net.load_state_dict(policy_net.state_dict())
    # create optimizer
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])
    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])

    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.npz')
            print("auto loading buffer from:%s" %args.buffer_loadpath)
            try:
                replay_memory.load_buffer(args.buffer_loadpath)
            except Exception as e:
                print(e)
                print('not able to load from buffer: %s. exit() to continue with empty buffer' %args.buffer_loadpath)

    # Scalar summaries for tensorboard: loss, average reward and evaluation score
    ############################################
    tf.reset_default_graph()
    SUMM_WRITER = tf.summary.FileWriter(model_base_filedir)
    with tf.name_scope('Performance'):
        PTLOSS_PH = tf.placeholder(tf.float32, shape=None, name='ptloss_summary')
        PTLOSS_SUMMARY = tf.summary.scalar('ptloss', PTLOSS_PH)
        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
        EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
        EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

    PERFORMANCE_SUMMARIES = tf.summary.merge([PTLOSS_SUMMARY, REWARD_SUMMARY])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ############################################
        train(start_step_number, start_last_save)

