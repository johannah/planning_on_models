import os
import numpy as np
import datetime
import time
from collections import Counter

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_value_

from IPython import embed



def choose_action(ch, state, head, action_function, random_state, eps):
    # we are not changing policy_net
    r = random_state.rand()
    # take random action with some prob
    if r < eps:
        action = random_state.randint(0, ch.cfg.actions)
        return random_state, action
    else:
        return random_state, action_function(ch, state, head)

def full_state_action_function(ch, state, head):
    ch.policy_net.eval()
    with torch.no_grad():
        # assume state is given in proper format?
        vals = ch.policy_net(state, None)
        # use given active head
        if type(head) is int:
            action = torch.argmax(vals[head]).item()
            return action
        # if head is a list of head indexes, vote on these heads (done in eval)
        if type(head) is list:
            acts = [torch.argmax(vals[h],dim=1).item() for h in range(head)]
            data = Counter(acts)
            # TODO -  is most_common biased towards low action values?
            action =  data.most_common(1)[0][0]
            return action

def prepare_minibatch(ch, minibatch):
    states, actions, rewards, next_states, terminal_flags, masks = minibatch
    assert(states.dtype == np.uint8)
    assert(next_states.dtype == np.uint8)
    # move states between 0 and 1 - they are stored as uint8
    states = torch.Tensor(states.astype(np.float)/ch.norm_by).to(ch.device)
    next_states = torch.Tensor(next_states.astype(np.float)/ch.norm_by).to(ch.device)
    assert(states.max() <= 1)
    assert(next_states.max() <= 1)
    assert(states.min() >= 0)
    assert(next_states.max() >= 0)

    rewards = torch.Tensor(rewards).to(ch.device)
    assert(rewards.max() <= 1)
    assert(rewards.min() >= 0)
    actions = torch.LongTensor(actions).to(ch.device)
    # TODO - check actions are valid
    #TODO - which way are terminals?
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(ch.device)
    masks = torch.FloatTensor(masks.astype(np.int)).to(ch.device)
    minibatch = states, actions, rewards, next_states, terminal_flags, masks
    return minibatch

def dqn_learn(ch):

    if ch.step_number > ch.cfg['DQN']['min_steps_to_learn']:
        if not ch.step_number%ch.cfg['DQN']['learn_every_steps']:
            prepared_minibatch = prepare_minibatch(train_replay_memory.get_minibatch())
            states, actions, rewards, next_states, terminal_flags, masks = prepared_minibatch
            # min history to learn is 200,000 frames in dqn - 50000 steps
            losses = [0.0 for _ in range(ch.cfg['DQN']['n_ensemble'])]
            opt.zero_grad()
            q_policy_vals = policy_net(states, None)
            next_q_target_vals = target_net(next_states, None)
            next_q_policy_vals = policy_net(next_states, None)
            cnt_losses = []
            for k in range(ch.cfg['DQN']['n_ensemble']):
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
                    targets = rewards + ch.cfg['DQN']['gamma'] * next_qs * (1-terminal_flags)
                    l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
                    full_loss = masks[:,k]*l1loss
                    loss = torch.sum(full_loss/total_used)
                    cnt_losses.append(loss)
                    losses[k] = loss.cpu().detach().item()
            loss = sum(cnt_losses)/float(ch.cfg['DQN']['n_ensemble'])
            loss.backward()

            for param in policy_net.core_net.parameters():
                if param.grad is not None:
                    # divide grads in core
                    param.grad.data *=1.0/float(ch.cfg['DQN']['n_ensemble'])
            nn.utils.clip_grad_norm_(policy_net.parameters(), ch.cfg['DQN']['clip_grad'])
            opt.step()
            ch.episode_losses.append(loss.mean().item())
        if not ch.step_number%self.cfg['DQN']['TARGET_UPDATE']:
            print("---%s updating target net"%ch.step_number)
            target_net.load_state_dict(policy_net.state_dict())
    return ch


def train(train_env, train_replay_memory, steps_to_train):
    # load stored random_state
    random_state = np.random.RandomState()
    random_state.set_state(ch.state_dict['train_random_state'])
    heads = ch.cfg['DQN']['N_ENSEMBLE']
    # TODO - this maybe should be zero or perhaps need function for changing
    train_eps = ch.cfg['DQN']['EPS_INIT']
    while cf.state_dict['step_number'] < steps_to_train:
        # figure out which head to make decisions with
        random_state.shuffle(heads)
        active_head = heads[0]
        # tmp holding
        mean_losses = []
        episode_st = time.time()
        start_step = ch.state_dict['step_number']
        #### START MAIN LOOP #####################################################################
        state = train_env.reset()
        terminal = False
        life_lost = True
        episodic_reward = 0
        while not terminal:
            random_state, action = choose_action(state, policy_net, active_head, random_state, train_eps)
            next_state, reward, life_lost, terminal = train_env.step(action)
            train_replay_memory.add_experience(action=action,
                                         frame=next_state[-1],
                                         reward=1+np.sign(reward),
                                         terminal=life_lost,
                                         )
            ch.state_dict['step_number']+=1
            episodic_reward += reward
            #### END MAIN LOOP #####################################################################
            ch = dqn_learn(ch)
            #### END TRAIN LOOP #####################################################################

        # end of episode is here
        episode_et = time.time()
        ch.state_dict['episode_number']+=1
        ch.perf_dict['steps'].append(ch.state_dict['step_number'])
        ch.perf_dict['episode_step'].append(ch.state_dict['step_number']-start_step)
        ch.perf_dict['episode_head'].append(active_head)
        # before dqn is trained - this list has nothing in it
        if len(mean_losses): avg_episode_loss = np.mean(mean_losses)
        else: avg_episode_loss = 0.0
        perf_dict['episode_loss'].append(avg_episode_loss)
        perf_dict['episode_reward'].append(episodic_reward)
        perf_dict['episode_time'].append(episode_et-episode_st)
        perf_dict['avg_reward'].append(np.mean(perf_dict['episode_reward'][-100:]))
        # TODO save
        # TODO - plot every episode individually
        # TODO - fix x axis plot
        plot_episode()
        if not episode_num or (ch.state_dict['step_number']-ch.state_dict['last_save']) >= ch.cfg['RUN']['checkpoint_every_steps']:
            train_replay_buffer.save_buffer(ch.get_replay_buffer_path('train', ch.state_dict['step_number']))
            save_checkpoint(ch)
        # TODO plot every
    # end of this training period
    ch.state_dict['train_random_state'] = random_state.get_state()
    return ch, train_env, train_replay_memory




if __name__ == '__main__':
    from argparse import ArgumentParser
    from config_handler import ConfigHandler
    parser = ArgumentParser()
    parser.add_argument('config_path', help='pass name of config file that will be used to generate random data')
    args = parser.parse_args()
    assert os.path.exists(args.config_path)
    ch = ConfigHandler(args.config_path)

    # this will load latest availalbe buffer - if none available - it will
    # create or load a random replay for this seed
    train_seed = ch.cfg['RUN']['train_seed']
    train_replay_memory = ch.load_replay_memory('train')
    train_env = ch.create_environment(train_seed)

    # this will load latest availalbe buffer - if none available - it will
    # create or load a random replay for this seed
    eval_seed = ch.cfg['RUN']['train_seed']
    eval_replay_memory = ch.load_replay_memory('eval')
    eval_env = ch.create_environment(eval_seed)

    tsteps =
    train_out = train(train_env, train_replay_memory, tsteps, state_dict, perf_dict)
    train_env, train_replay_memory = train_out
