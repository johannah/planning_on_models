"""
TODO add annealing

in this version, we are testing learning a mf policy based on reconstructions from a trained
reconstruction model
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
sys.path.append('../models')
from train_breakout_conv_acn_pcnn_bce_prediction_actgrad import ConvVAE, GatedPixelCNN, PriorNetwork
from train_breakout_conv_acn_pcnn_bce_prediction_actgrad import make_state as rep_make_state


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
    data = Counter(acts)
    # TODO -  is most_common biased towards low action values?
    action =  data.most_common(1)[0][0]
    print("vote", data, action)
    return action

def get_frame_prepared_minibatch(ch, minibatch):
    #states, actions, rewards, next_states, terminal_flags, masks = sm.memory_buffer.get_minibatch(sm.ch.cfg['DQN']['batch_size'])
    states, actions, rewards, next_states, terminal_flags, masks, pred_states, pred_next_states = minibatch
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
    minibatch = states, actions, rewards, next_states, terminal_flags, masks, pred_states, pred_next_states
    return  minibatch

def acn_pcnn_pred_next_state_function(batch):
    states, actions, rewards, next_states = rep_make_state(batch, device, 255.0)
    with torch.no_grad():
        z, u_q, s_q = rep_dict['vae_model'](states)
        pred_next_states = torch.sigmoid(rep_dict['pcnn_decoder'](x=next_states, float_condition=z))
    # convert back to 0 to 255. so it can be put in replay buffer
    return pred_next_states


def dqn_learn(sm, model_dict):
    s_loss = 0
    if sm.step_number > sm.ch.cfg['DQN']['min_steps_to_learn']:
        if not sm.step_number%sm.ch.cfg['DQN']['learn_every_steps']:
            minibatch = sm.memory_buffer.get_minibatch(sm.ch.cfg['DQN']['batch_size'])
            prepared_minibatch = get_frame_prepared_minibatch(sm.ch, minibatch)
            states, actions, rewards, next_states, terminal_flags, masks, pred_states, pred_next_states = prepared_minibatch
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
    # resize using max pool
    dummy_next_state = state
    make_atari_channel_action_reward_state(state, prev_action, prev_reward, dummy_next_state, info['device'], info['num_actions'], info['num_rewards'])

def train_agent(sm, model_dict, steps_to_train):
    print('training at S%s'%sm.step_number)
    start_step = deepcopy(sm.step_number)
    while (sm.step_number-start_step) < steps_to_train:
        #### START MAIN LOOP #####################################################################
        sm.start_episode()
        while not sm.terminal:
            is_random, action = sm.is_random_action()
            if not is_random:
                # state coming from the env looks like (4,84,84) and is a uint8
                pt_state = prepare_state(sm.ch, sm.state[None])
                vals = get_action_vals(model_dict['policy_net'], pt_state)
                action = single_head_action_function(vals, sm.active_head)
            sm.step_pred_next_state(action)
            sm, model_dict =  dqn_learn(sm, model_dict)
        sm.end_episode()
        sm.handle_plotting()
        # TODO plot every
    return sm, model_dict

def eval_agent(sm, model_dict, num_episodes_to_eval):
    start_episode = deepcopy(sm.episode_number)
    while (sm.episode_number-start_episode) < num_episodes_to_eval:
        #### START MAIN LOOP #####################################################################
        sm.start_episode()
        while not sm.terminal:
            is_random, action = sm.is_random_action()
            if not is_random:
                pt_state = prepare_state(sm.ch, sm.state[None])
                vals = get_action_vals(model_dict['policy_net'], pt_state)
                action = vote_action_function(vals)
            sm.step_pred_next_state(action)
        sm.end_episode()
        print('eval', np.sum(sm.episode_rewards))
        print(sm.episode_rewards)
        print(sm.episode_actions)
    return sm

def save_models(checkpoint_basepath, model_dict):
    model_state_dict = {}
    for model_name in model_dict.keys():
        model_state_dict[model_name] = model_dict[model_name].state_dict
    models_savepath = checkpoint_basepath+'.pt'
    print("saving models at: %s"%models_savepath)
    torch.save(model_state_dict, models_savepath)

def load_models(filepath, model_dict):
    model_state_dict = torch.load(filepath)
    for model_name in model_dict.keys():
        model_dict[model_name].load_state_dict(model_state_dict[model_name])
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

def load_uvdeconv_representation_model(representation_model_path):
    # model trained with this file:
    # ../models/train_atari_uvdeconv_tacn_twgradloss.py
    # will output a acn flat float representation and a vq discrete
    # representation - which to use?
    from train_atari_uvdeconv_tacn_twgradloss import create_models, forward_pass, make_atari_channel_action_reward_state
    rep_info = {'device':device}
    rep_model_dict, _, rep_info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(rep_info, representation_model_path)
    return rep_model_dict, rep_info


if __name__ == '__main__':
    from argparse import ArgumentParser
    from handler import ConfigHandler, StateManager
    parser = ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/mf_rep_breakout_config.txt', help='path of config file that will be used to generate random data')
    parser.add_argument('-lp', '--load_path', default='', help='path of .pkl state manager file to load checkpoint')
    parser.add_argument('-mp', '--model_path', default='', help='path of .pt model file to load checkpoint')
    parser.add_argument('-rmp', '--representation_model_path',
                        default='../../model_savedir/results_train_breakout_acn_bce_pcnn_pred_actgrad/sigcacn_breakout_binary_bce_pcnn_pred_actgrad_0012001280ex.pt',
                        help='path of reconstruction .pt model file to load checkpoint')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='flag to use cuda device')
    # TODO - add reload and continue of previous projects


    args = parser.parse_args()
    if args.cuda: device = 'cuda';
    else: device='cpu'

    rep_model_dict, rep_info = load_uvdeconv_representation_model(args.representation_model_path)

    # this will load latest availalbe buffer - if none available - it will
    # create or load a random replay for this seed
    train_sm = StateManager()
    eval_sm = StateManager()

    if args.config_path == '':
        print('loading checkpoint and its config')
        train_sm.load_checkpoint(filepath=args.load_path)
        eval_sm.load_checkpoint(filepath=args.load_path.replace('train', 'eval'))
        ch = train_sm.ch
        ch.device = device
    else:
        # load given configuration file and create experiment directory
        ch = ConfigHandler(args.config_path, device=device, pred_next_state_function=acn_pcnn_pred_next_state_function)
        if args.load_path == '':
            print('creating new state instance')
            train_sm.create_new_state_instance(config_handler=ch, phase='train')
            eval_sm.create_new_state_instance(config_handler=ch, phase='eval')
        else:
            print('loading checkpoint with specified config')
            train_sm.load_checkpoint(filepath=args.load_path, phase='train', config_handler=ch)
            eval_sm.load_checkpoint(filepath=args.load_path.replace('train', 'eval'), phase='eval', config_handler=ch)

    # make sure given info in the config file is the same as what the
    # representation model was trained on
    #assert (rep_info['wsize'] ==  ch.cfg['ENV']['obs_width'])
    #assert (rep_info['hsize'] ==  ch.cfg['ENV']['obs_height'])
    # JRH Jan 2020 - for the uvdeconv model, it should be resized once by
    # env.py, then resized again with max pooling to get it down to the
    # prescribed size - this is available in replay.py

    seed_everything(ch.cfg['RUN']['train_seed'])
    model_dict = create_dqn_model_dict(ch, num_actions=train_sm.env.num_actions)
    if args.model_path != '':
        model_dict = load_models(args.model_path, model_dict)
    elif args.load_path != '':
        model_dict = load_models(args.load_path.replace('_train', '.pt'), model_dict)
    else:
        print('fresh models')


    #TODO load model_dict

    steps_to_train = ch.cfg['RUN']['eval_and_checkpoint_every_steps']
    num_eval_episodes = ch.cfg['EVAL']['num_eval_episodes']
    while train_sm.step_number < ch.cfg['RUN']['total_train_steps']:
        train_sm, model_dict = train_agent(train_sm, model_dict, steps_to_train)
        # we save according to train num - so train should come before eval
        eval_sm = eval_agent(eval_sm, model_dict, num_eval_episodes)

        checkpoint_basepath = ch.get_checkpoint_basepath(train_sm.step_number)
        save_models(checkpoint_basepath, model_dict)
        train_sm.save_checkpoint(checkpoint_basepath+'_train')
        eval_sm.save_checkpoint(checkpoint_basepath+'_eval')
        eval_sm.plot_current_episode(plot_basepath=checkpoint_basepath+'_eval')

