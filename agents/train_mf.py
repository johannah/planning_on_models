"""
TODO - replay_memory_loadpath
"""
import os
import sys
import numpy as np
import datetime
import time
from collections import Counter
from copy import deepcopy

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_model import EnsembleNet, NetWithPrior

from IPython import embed
def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_action_vals(policy_net, state):
    # TODO handle state for action
    with torch.no_grad():
        policy_net.eval()
        # assume state is given in proper format?
        vals = policy_net(state, None)
        return vals

def single_head_action_function(vals, head):
    # use given active head
    action = torch.argmax(vals[head]).item()
    return action

def vote_action_function(vals):
    # if head is a list of head indexes, vote on these heads (done in eval)
    acts = [torch.argmax(vals[h],dim=1).item() for h in range(len(vals))]
    values,counts = np.unique(acts, return_counts=True)
    maxvals = values[counts == counts.max()]
    # TODO - make random state
    action = sm.random_state.choice(maxvals)
    print(values, counts)
    print(maxvals, 'action', action)
    return action

def get_frame_prepared_minibatch(ch, minibatch):
    #states, actions, rewards, next_states, terminal_flags, masks = sm.memory_buffer.get_minibatch(sm.ch.cfg['DQN']['batch_size'])
    states, actions, rewards, next_states, terminal_flags, masks = minibatch
    states = prepare_state(ch, states)
    next_states = prepare_state(ch, next_states)
    # move states between 0 and 1 - they are stored as uint8
    assert(states.max() <= 1)
    assert(next_states.max() <= 1)
    assert(states.min() >= 0)
    assert(next_states.max() >= 0)

    rewards = torch.Tensor(rewards).to(ch.device)
    assert(rewards.max() <= 1)
    assert(rewards.min() >= -1)
    actions = torch.LongTensor(actions).to(ch.device)
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(ch.device)
    masks = torch.FloatTensor(masks.astype(np.int)).to(ch.device)
    minibatch = states, actions, rewards, next_states, terminal_flags, masks
    return  minibatch

def dqn_learn(sm, model_dict):
    s_loss = 0
    if sm.step_number > sm.ch.cfg['DQN']['min_steps_to_learn']:
        if not sm.step_number%sm.ch.cfg['DQN']['learn_every_steps']:
            minibatch = sm.memory_buffer.get_minibatch(sm.ch.cfg['DQN']['batch_size'])
            prepared_minibatch = get_frame_prepared_minibatch(sm.ch, minibatch)
            states, actions, rewards, next_states, terminal_flags, masks = prepared_minibatch
            # min history to learn is 200,000 frames in dqn - 50000 steps
            losses = [0.0 for _ in range(sm.ch.cfg['DQN']['n_ensemble'])]
            model_dict['opt'].zero_grad()
            q_policy_vals = model_dict['policy_net'](states, None)
            next_q_target_vals = model_dict['target_net'](next_states, None)
            next_q_policy_vals = model_dict['policy_net'](next_states, None)
            cnt_losses = []
            for k in range(sm.ch.cfg['DQN']['n_ensemble']):
                #TODO finish masking
                total_used = torch.sum(masks[:,k])
                if total_used > 0.0:
                    next_q_vals = next_q_target_vals[k].data
                    if sm.ch.cfg['DQN']['double_dqn']:
                        next_actions = next_q_policy_vals[k].data.max(1, True)[1]
                        next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
                    else:
                        next_qs = next_q_vals.max(1)[0] # max returns a pair

                    preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
                    targets = rewards + sm.ch.cfg['DQN']['gamma'] * next_qs * (1-terminal_flags)
                    l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
                    full_loss = masks[:,k]*l1loss
                    loss = torch.sum(full_loss/total_used)
                    cnt_losses.append(loss)
                    losses[k] = loss.cpu().detach().item()
            loss = sum(cnt_losses)/float(sm.ch.cfg['DQN']['n_ensemble'])
            loss.backward()
            for param in model_dict['policy_net'].core_net.parameters():
                if param.grad is not None:
                    # divide grads in core
                    param.grad.data *=1.0/float(sm.ch.cfg['DQN']['n_ensemble'])
            nn.utils.clip_grad_norm_(model_dict['policy_net'].parameters(), sm.ch.cfg['DQN']['clip_grad'])
            model_dict['opt'].step()
            s_loss = loss.mean().item()
        if not sm.step_number%sm.ch.cfg['DQN']['target_update_every_steps']:
            print("---%s updating target net"%sm.step_number)
            model_dict['target_net'].load_state_dict(model_dict['policy_net'].state_dict())
    sm.episode_losses.append(s_loss)
    return sm, model_dict

def prepare_state(ch, state):
    assert(state.dtype == np.uint8)
    pt_state = torch.Tensor(state.astype(np.float)/ch.norm_by).to(ch.device)
    return pt_state

def run_agent(sm, model_dict, phase, max_count, count_type='steps'):
    print('training at S%s'%sm.step_number)
    start_step = deepcopy(sm.step_number)
    start_episode = deepcopy(sm.episode_number)
    count = 0
    while count < max_count:
        #### START MAIN LOOP #####################################################################
        sm.start_episode()
        while not sm.terminal:
            is_random = sm.is_random_action()
            if is_random and sm.step_number > 0:
                action = sm.random_action()
            else:
                # state coming from the model looks like (4,84,84) and is a uint8
                pt_state = prepare_state(sm.ch, sm.state[None])
                vals = get_action_vals(model_dict['policy_net'], pt_state)
            if phase == 'train':
                action = single_head_action_function(vals, sm.active_head)
            else:
                action = vote_action_function(vals)
            sm.step(action)
            if phase == 'train':
                sm, model_dict =  dqn_learn(sm, model_dict)
        sm.end_episode()
        sm.handle_plotting()
        if phase == 'eval':
            sm.plot_current_episode()
        if count_type == 'steps':
            count = sm.step_number - start_step
        else:
            count = sm.episode_number - start_episode
        # TODO plot every
    return sm, model_dict

def save_models(checkpoint_basepath, model_dict):
    model_state_dict = {}
    for model_name in model_dict.keys():
        model_state_dict[model_name] = model_dict[model_name].state_dict()
    models_savepath = checkpoint_basepath+'.pt'
    print("saving models at: %s"%models_savepath)
    torch.save(model_state_dict, models_savepath)

def load_models(filepath, model_dict):
    model_state_dict = torch.load(filepath)
    for model_name in model_dict.keys():
        try:
            model_dict[model_name].load_state_dict(model_state_dict[model_name])
        except:
            # forgot to make function when saving in some versions
            model_dict[model_name].load_state_dict(model_state_dict[model_name]())
    return model_dict

def create_dqn_model_dict(ch, num_actions, model_dict={}):
    model_dict['policy_net'] = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      reshape_size=ch.cfg['DQN']['reshape_size'],
                                      num_channels=ch.cfg['ENV']['history_size'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

    model_dict['target_net'] = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      reshape_size=ch.cfg['DQN']['reshape_size'],
                                      num_channels=ch.cfg['ENV']['history_size'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

    if ch.cfg['DQN']['prior']:
        print("using randomized prior")
        prior_net = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      reshape_size=ch.cfg['DQN']['reshape_size'],
                                      num_channels=ch.cfg['ENV']['history_size'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

        model_dict['policy_net'] = NetWithPrior(model_dict['policy_net'], prior_net, ch.cfg['DQN']['prior_scale'])
        model_dict['target_net'] = NetWithPrior(model_dict['target_net'], prior_net, ch.cfg['DQN']['prior_scale'])

    model_dict['target_net'].load_state_dict(model_dict['policy_net'].state_dict())
    model_dict['opt'] = optim.Adam(model_dict['policy_net'].parameters(), lr=ch.cfg['DQN']['adam_learning_rate'])
    return model_dict


if __name__ == '__main__':
    """
    TODO -
    - when loading a model - write a file to indicate where it was reloadedi if training and write new files in same dir
    - load and eval all files
    - keep track of each heads avg score
    """
    from argparse import ArgumentParser
    from handler import ConfigHandler, StateManager
    parser = ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/mf_rep_breakout_config.txt', help='path of config file that will be used to generate random data')
    parser.add_argument('-lp', '--load_path', default='', help='path of .pkl state manager file to load checkpoint')
    parser.add_argument('-ll', '--load_last_path', default='', help='given working directory of last checkpoint, load last checkpoints')
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='plot a loaded model')
    parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='eval a loaded model')
    parser.add_argument('-mp', '--model_path', default='', help='path of .pt model file to load checkpoint')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='flag to use cuda device')
    # TODO - add reload and continue of previous projects
    args = parser.parse_args()
    if args.cuda: device = 'cuda';
    else: device='cpu'
    num_train_steps = 0
    # this will load latest availalbe buffer - if none available - it will
    # create or load a random replay for this seed

    if args.evaluate:
        phase = 'eval'
        # breakout_S014342_N0001754345_train.pkl[
    else:
        phase = 'train'

    print("phase is %s"%phase)
    sm = StateManager()
    # load given configuration file and create experiment directory
    ch = ConfigHandler(args.config_path, device=device)
    if args.load_path == '' or phase == 'eval':
        print('creating new state instance')
        sm.create_new_state_instance(config_handler=ch, phase=phase)
    else:
        print('loading checkpoint with specified config - always load train')
        sm.load_checkpoint(filepath=args.load_path, config_handler=ch)
    seed_everything(ch.cfg['RUN']['train_seed'])
    num_actions = sm.env.num_actions
    num_rewards = sm.ch.num_rewards
    model_dict = create_dqn_model_dict(ch, num_actions=sm.env.num_actions)
    if args.model_path != '':
        num_train_steps = int(os.path.split(args.model_path)[1].split('.')[0].split('_')[-1][1:])
        model_dict = load_models(args.model_path, model_dict)
    elif args.load_path != '':
        lpath = args.load_path.replace('_train.pkl', '.pt')
        num_train_steps = int(os.path.split(lpath)[1].split('.')[0].split('_')[-2][1:])
        model_dict = load_models(lpath, model_dict)
    else:
        print('fresh models')

    if phase == 'train':
        max_count = ch.cfg['RUN']['eval_and_checkpoint_every_steps']
        count_type = 'steps'
    else:
        max_count = ch.cfg['EVAL']['num_eval_episodes']
        count_type = 'episodes'

    ll_idx = 0
    #if args.evaluate and args.load_last_path != '':
    #    while True:
    #        avail_models = sorted(glob(os.path.join(args.load_last_path, '*.pt')))
    #        if ll_idx > len(avail_models)-1:
    #            break
    #        cur_path = avail_models[ll_idx]
    #        num_train_steps = int(os.path.split(cur_path)[1].split('.')[0].split('_')[-2][1:])
    #        # todo - load last checkpoint
    #        checkpoint_basepath = ch.get_checkpoint_basepath(num_train_steps)+'_'+phase
    #        if not os.path.exists(checkpoint_basepath+'.pkl'):
    #            sm.load_checkpoint(filepath=cur_path, config_handler=ch)
    #            model_dict = load_models(cur_path, model_dict)
    #            sm, model_dict = run_agent(sm, model_dict,  phase=phase, max_count=max_count, count_type=count_type)
    #            sm.save_checkpoint(checkpoint_basepath)
    #        if args.plot:
    #            print('plotting last episode')
    #            sm.plot_last_episode()
    if args.evaluate:
        print("-----------starting eval--------------")
        sm.active_head = 0
        sm, model_dict = run_agent(sm, model_dict,  phase=phase, max_count=max_count, count_type=count_type)
        checkpoint_basepath = ch.get_checkpoint_basepath(num_train_steps)+'_'+phase
        sm.save_checkpoint(checkpoint_basepath)
        if args.plot:
            print('plotting last episode')
            sm.plot_last_episode()
    else:
        print("TRAINING")
        if num_train_steps < ch.cfg['RUN']['total_train_steps'] and phase == 'train':
            sm, model_dict = run_agent(sm, model_dict,  phase=phase, max_count=max_count, count_type=count_type)
            num_train_steps = sm.step_number
            checkpoint_basepath = ch.get_checkpoint_basepath(num_train_steps)+'_'+phase
            sm.save_checkpoint(checkpoint_basepath)
            save_models(checkpoint_basepath, model_dict)
            if args.plot:
                print('plotting last episode')
                sm.plot_last_episode()

#    from argparse import ArgumentParser
#    from handler import ConfigHandler, StateManager
#    parser = ArgumentParser()
#    parser.add_argument('-cp', '--config_path', help='path of config file that will be used to generate random data')
#    parser.add_argument('-lp', '--load_path', default='', help='path of .pkl state manager file to load checkpoint')
#    parser.add_argument('-mp', '--model_path', default='', help='path of .pt model file to load checkpoint')
#    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='flag to use cuda device')
#    # TODO - add reload and continue of previous projects
#    args = parser.parse_args()
#    if args.cuda: device = 'cuda';
#    else: device='cpu'
#
#
#    # this will load latest availalbe buffer - if none available - it will
#    # create or load a random replay for this seed
#    train_sm = StateManager()
#    #eval_sm = StateManager()
#
#    if args.config_path == '':
#        print('loading checkpoint and its config')
#        train_sm.load_checkpoint(filepath=args.load_path)
#        #eval_sm.load_checkpoint(filepath=args.load_path.replace('train', 'eval'))
#        ch = train_sm.ch
#        ch.device = device
#    else:
#        # load given configuration file and create experiment directory
#        ch = ConfigHandler(args.config_path, device=device)
#        if args.load_path == '':
#            print('creating new state instance')
#            train_sm.create_new_state_instance(config_handler=ch, phase='train')
#            #eval_sm.create_new_state_instance(config_handler=ch, phase='eval')
#        else:
#            print('loading checkpoint with specified config')
#            train_sm.load_checkpoint(filepath=args.load_path, phase='train', config_handler=ch)
#            #eval_sm.load_checkpoint(filepath=args.load_path.replace('train', 'eval'), phase='eval', config_handler=ch)
#
#    seed_everything(ch.cfg['RUN']['train_seed'])
#    model_dict = create_dqn_model_dict(ch, num_actions=train_sm.env.num_actions)
#    if args.model_path != '':
#        model_dict = load_models(args.model_path, model_dict)
#    elif args.load_path != '':
#        model_dict = load_models(args.load_path.replace('_train', '.pt'), model_dict)
#    else:
#        print('fresh models')
#
#    checkpoint_basepath = ch.get_checkpoint_basepath(train_sm.step_number)
#    save_models(checkpoint_basepath, model_dict)
#
#    steps_to_train = ch.cfg['RUN']['eval_and_checkpoint_every_steps']
#    num_eval_episodes = ch.cfg['EVAL']['num_eval_episodes']
#    while train_sm.step_number < ch.cfg['RUN']['total_train_steps']:
#        train_sm, model_dict = run_agent(train_sm, model_dict, steps_to_train)
#        # we save according to train num - so train should come before eval
#        #eval_sm = eval_agent(eval_sm, model_dict, num_eval_episodes)
#
#        checkpoint_basepath = ch.get_checkpoint_basepath(train_sm.step_number)
#        save_models(checkpoint_basepath, model_dict)
#        train_sm.save_checkpoint(checkpoint_basepath+'_train')
#        #eval_sm.save_checkpoint(checkpoint_basepath+'_eval')
#        #eval_sm.plot_current_episode(plot_basepath=checkpoint_basepath+'_eval')
#
