"""
TODO add annealing

in this version, we are testing learning a mf policy based on reconstructions from a trained
reconstruction model

in this version - we use the past x z states for decision making - the hope is that this will balance some jitter and also give a long perspective
"""
import os
import sys
import numpy as np
import datetime
import time
from glob import glob
from collections import Counter
from copy import deepcopy

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_acn_model import EnsembleNet, NetWithPrior

sys.path.append('../models')
from acn_utils import count_parameters, sample_from_discretized_mix_logistic, set_model_mode
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

def prepare_details(actions, rewards, terminal_flags, masks, device):
    rewards = torch.Tensor(rewards).to(device)
    assert(rewards.max() <= 1)
    assert(rewards.min() >= -1)
    actions = torch.LongTensor(actions).to(device)
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(device)
    masks = torch.FloatTensor(masks.astype(np.int)).to(device)
    return actions, rewards, terminal_flags, masks

def prepare_past_details(actions, rewards, terminal_flags, masks, device):
    rewards = torch.Tensor(rewards).to(device)
    assert(rewards.max() <= 1)
    assert(rewards.min() >= -1)
    embed()
    actions = torch.LongTensor(actions).to(device)
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(device)
    masks = torch.FloatTensor(masks.astype(np.int)).to(device)
    return actions, rewards, terminal_flags, masks


def dqn_learn(sm, model_dict):
    # TODO - put rep model in eval / no_grad
    s_loss = 0
    if sm.step_number > sm.ch.cfg['DQN']['min_steps_to_learn']:
        if not sm.step_number%sm.ch.cfg['DQN']['learn_every_steps']:
            minibatch = sm.memory_buffer.get_history_minibatch(sm.ch.cfg['DQN']['batch_size'])
            states, actions, rewards, next_states, terminal_flags, masks = minibatch
            # pt_states is acn representation bs,3,8,8
            pt_states = prepare_state_fn(states)
            pt_next_states = prepare_state_fn(next_states)
            actions, rewards, terminal_flags, masks = prepare_details(actions, rewards, terminal_flags, masks, ch.device)
            # min history to learn is 200,000 frames in dqn - 50000 steps
            losses = [0.0 for _ in range(sm.ch.cfg['DQN']['n_ensemble'])]
            model_dict['opt'].zero_grad()
            q_policy_vals = model_dict['policy_net'](pt_states, None)
            next_q_target_vals = model_dict['target_net'](pt_next_states, None)
            next_q_policy_vals = model_dict['policy_net'](pt_next_states, None)
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

def save_models(checkpoint_basepath, model_dict):
    model_state_dict = {}
    for model_name in model_dict.keys():
        model_state_dict[model_name] = model_dict[model_name].state_dict()
    models_savepath = checkpoint_basepath+'.pt'
    print("saving models at: %s"%models_savepath)
    torch.save(model_state_dict, models_savepath)

def load_models(filepath, model_dict):
    model_state_dict = torch.load(filepath, map_location=lambda storage, loc:storage)
    print('loading dqn models from', filepath)
    for model_name in model_dict.keys():
        print('loaded', model_name)
        model_dict[model_name].load_state_dict(model_state_dict[model_name])
    return model_dict

def create_dqn_model_dict(ch, num_actions, model_dict={}):
    cl = 192 # todo load this
    rsize = cl*ch.cfg['REP']['num_prev_steps']
    model_dict['policy_net'] = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      code_length=rsize,
                                      num_hidden=ch.cfg['DQN']['n_hidden'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

    model_dict['target_net'] = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      code_length=rsize,
                                      num_hidden=ch.cfg['DQN']['n_hidden'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

    if ch.cfg['DQN']['prior']:
        print("using randomized prior")
        prior_net = EnsembleNet(n_ensemble=ch.cfg['DQN']['n_ensemble'],
                                      n_actions=num_actions,
                                      code_length=rsize,
                                      num_hidden=ch.cfg['DQN']['n_hidden'],
                                      dueling=ch.cfg['DQN']['dueling']).to(ch.device)

        model_dict['policy_net'] = NetWithPrior(model_dict['policy_net'], prior_net, ch.cfg['DQN']['prior_scale'])
        model_dict['target_net'] = NetWithPrior(model_dict['target_net'], prior_net, ch.cfg['DQN']['prior_scale'])

    model_dict['target_net'].load_state_dict(model_dict['policy_net'].state_dict())

    for name,model in model_dict.items():
        print('created %s model with %s parameters' %(name,count_parameters(model)))
        model.eval()

    model_dict['opt'] = optim.Adam(model_dict['policy_net'].parameters(), lr=ch.cfg['DQN']['adam_learning_rate'])
    return model_dict

def load_uvdeconv_representation_model(representation_model_path):
    # model trained with this file:
    # ../models/train_atari_uvdeconv_tacn_midtwgradloss.py
    # will output a acn flat float representation and a vq discrete
    # representation - which to use?
    rep_info = {'device':device, 'args':args}
    rep_model_dict, _, rep_info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(rep_info, representation_model_path, load_data=False)
    rep_model_dict = set_model_mode(rep_model_dict, 'valid')
    #return rep_model_dict, rep_info, prepare_uv_state_latents, rescale, rescale_inv
    return rep_model_dict, rep_info, prepare_uv_state_hist_latents, rescale, rescale_inv

def prepare_uv_state_hist_latents(states):
    # resize using max pool
    bs,nh,c,h,w =  states.shape
    if c == 0:
        print("BAD size of channels")
    with torch.no_grad():
        # rep model is trained on past 4 states, we have prev "4-frame-states"
        # here and need to find the z for each of them -
        for bi in range(bs):
            # treat prev frames like batch and do all at once if the batch size
            # is tiny
            z, u_q = rep_model_dict['mid_vq_acn_model'](torch.FloatTensor(rescale(states[bi])).to(device))
            # reshape so that zs are stacked in the channel dim
            # this will be shape = (1,3*nh, 8, 8)
            z = z.permute(2,3,0,1).reshape(8,8,-1).permute(2,0,1)[None]
            if not bi:
                zo = z
            else:
                zo = torch.cat((zo, z), dim=0)
    return zo

def prepare_uv_state_latents(states):
    # resize using max pool
    bs,c,h,w =  states.shape
    if c == 0:
        print("BAD size of channels")
        embed()
    with torch.no_grad():
        z, u_q = rep_model_dict['mid_vq_acn_model'](torch.FloatTensor(rescale(states)).to(device))
    return z

def get_dummy_representation(states, actions, rewards):
    return states

def pad_and_split(arr, bs):
    tot = arr.shape[0]
    leftover = tot % bs
    pad = bs - leftover
    pad_arr = torch.Tensor(np.zeros((pad, arr.shape[1], arr.shape[2], arr.shape[3])))
    arr = torch.cat([arr, pad_arr])
    return torch.split(arr, bs)

def get_latent_pred_representation(states, actions, rewards, ch):
    bs = rep_info['batch_size']
    tot = states.shape[0]
    # split states, action_cond, and reward_cond into batch size chunks
    # before returning to prevent overrun on gpu
    # next state is the corresponding action
    action_cond = torch.zeros((tot,num_actions,8,8))
    reward_cond = torch.zeros((tot,num_rewards,8,8))
    # add in actions/reward as conditioning
    for i in range(tot):
        a = actions[i]
        r = rewards[i]
        action_cond[i,a]=1.0
        reward_cond[i,r]=1.0
    # without chunking
    #with torch.no_grad():
    #    z = prepare_uv_state_latents(states)
    #    rec_dml, z_e_x, z_q_x, latents =  rep_model_dict['mid_vq_acn_model'].decode(z, action_cond.to(device), reward_cond.to(device))
    #    rec_yhat = sample_from_discretized_mix_logistic(rec_dml, rep_info['nr_logistic_mix'], only_mean=False, sampling_temperature=0.1)
    #    rec_yhat = rescale_inv(rec_yhat).cpu().numpy()[:,0]
    #return rec_yhat, z, latents
    state_chunks = pad_and_split(torch.FloatTensor(states), bs)
    reward_chunks = pad_and_split(reward_cond, bs)
    action_chunks = pad_and_split(action_cond, bs)

    with torch.no_grad():
        fktot = bs*len(state_chunks)
        rec_out = np.empty((fktot, ch.frame_height, ch.frame_width), dtype=np.uint8)
        z_out = np.zeros((fktot, 3, 8, 8))
        latent_out = np.zeros((fktot, 8, 8))
        for chunk in range(len(state_chunks)):
            z = prepare_uv_state_latents(state_chunks[chunk])
            rec_dml, z_e_x, z_q_x, latents =  rep_model_dict['mid_vq_acn_model'].decode(z, action_chunks[chunk].to(device), reward_chunks[chunk].to(device))
            rec_yhat = sample_from_discretized_mix_logistic(rec_dml, rep_info['nr_logistic_mix'], only_mean=False, sampling_temperature=0.1)
            rec_yhat = rescale_inv(rec_yhat).cpu().numpy()[:,0].astype(np.uint8)
            rec_out[chunk*bs:(chunk+1)*bs] = rec_yhat
            z_out[chunk*bs:(chunk+1)*bs] = z.cpu().numpy()
            latent_out[chunk*bs:(chunk+1)*bs] = latents.cpu().numpy()
    return rec_out[:tot], z_out[:tot], latent_out[:tot]


def run_agent(sm, model_dict, phase, max_count, count_type='steps'):
    """
    num_to_run: number of steps or episodes to run - training is conventionally
    measured in steps while evaluation is measured in episodes.
    """
    print('running %s agent at S%s'%(phase, sm.step_number))
    start_step = deepcopy(sm.step_number)
    start_episode = deepcopy(sm.episode_number)
    count = 0
    while count < max_count:
        #### START MAIN LOOP #####################################################################
        sm.start_episode()
        total_reward = 0
        while not sm.terminal:
            is_random = sm.is_random_action()
            # make first step go thru agent for debugging
            if is_random and sm.step_number > 0:
                action = sm.random_action()
            else:
                # state coming from the env looks like (4,40,40) and is a uint8
                if sm.state[None].shape != (1, sm.ch.num_prev_steps, sm.ch.history_length, sm.ch.frame_height, sm.ch.frame_width):
                    print('wrong state size')
                    embed()
                pt_state = prepare_state_fn(sm.state[None])
                total_reward += sm.prev_reward

                vals = get_action_vals(model_dict['policy_net'], pt_state)
                if phase == 'train':
                    action = single_head_action_function(vals, sm.active_head)
                else:
                    action = vote_action_function(vals)
            #step_count = sm.step_number-start_step
            #print(step_count, total_reward, sm.active_head, pt_state.sum(), sm.state.sum(), action)
            sm.step(action)
            if phase == 'train':
                sm, model_dict =  dqn_learn(sm, model_dict)
            if not sm.step_number%1000:
                print('ah', sm.active_head, 'A', action, 'sn', sm.step_number, 'tR', sum(sm.episode_rewards), 'stsum', pt_state.sum().data)
        sm.end_episode()
        sm.handle_plotting()
        print(phase, 'total reward', np.sum(sm.episode_rewards))
        if phase == 'eval':
            sm.plot_current_episode()
            print('rewards', np.histogram(sm.episode_rewards))
        if count_type == 'steps':
            count = sm.step_number-start_step
        else:
            count = sm.episode_number-start_episode
    return sm, model_dict


if __name__ == '__main__':
    """
    TODO -
    - when loading a model - write a file to indicate where it was reloadedi if training and write new files in same dir
    - load and eval all files
    - keep track of each heads avg score
    - track of min/max/avg reward
    """

    from train_atari_uvdeconv_tacn_midtwgradloss import create_models, forward_pass, make_atari_channel_action_reward_state
    from argparse import ArgumentParser
    from handler_hist import ConfigHandler, StateManager
    parser = ArgumentParser()
    parser.add_argument('config_path', help='path of config file that will be used to generate random data')
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
    #if args.config_path == '':
    #    print('loading checkpoint and its config')
    #    sm.load_checkpoint(filepath=args.load_path)
    #    ch = sm.ch
    #    ch.device = device
    #else:
    # load given configuration file and create experiment directory
    ch = ConfigHandler(args.config_path, device=device)
    if args.load_path == '' or phase == 'eval':
        print('creating new state instance')
        sm.create_new_state_instance(config_handler=ch, phase=phase)
    else:
        print('loading checkpoint with specified config - always load train')
        sm.load_checkpoint(filepath=args.load_path, config_handler=ch)
    # make sure given info in the config file is the same as what the
    # representation model was trained on
    #assert (rep_info['wsize'] ==  ch.cfg['ENV']['obs_width'])
    #assert (rep_info['hsize'] ==  ch.cfg['ENV']['obs_height'])
    # JRH Jan 2020 - for the uvdeconv model, it should be resized once by
    # env.py, then resized again with max pooling to get it down to the
    # prescribed size - this is available in replay.py
    rep_model_path = ch.cfg['REP']['rep_model_path']
    #rep_model_dict, rep_info, prepare_state_fn, rescale, rescale_inv = load_uvdeconv_representation_model(rep_model_path)
    rep_model_dict, rep_info, prepare_state_fn, rescale, rescale_inv = load_uvdeconv_representation_model(rep_model_path)
    sm.latent_representation_function = get_latent_pred_representation
    seed_everything(ch.cfg['RUN']['train_seed'])
    num_actions = sm.env.num_actions
    num_rewards = sm.ch.num_rewards
    model_dict = create_dqn_model_dict(ch, num_actions=sm.env.num_actions)
    if args.model_path != '':
        #TODO - not sure it is actually loading model
        num_train_steps = int(os.path.split(args.model_path)[1].split('.')[0].split('_')[-1][1:])
        model_dict = load_models(args.model_path, model_dict)
    elif args.load_path != '':
        lpath = args.load_path.replace('_train.pkl', '.pt')
        num_train_steps = int(os.path.split(lpath)[1].split('.')[0].split('_')[-2][1:])
        model_dict = load_models(lpath, model_dict)
    else:
        print('fresh models')


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
        max_count = ch.cfg['EVAL']['num_eval_episodes']
        count_type = 'episodes'
        print("-----------starting eval--------------")
        sm.active_head = 0
        sm, model_dict = run_agent(sm, model_dict,  phase=phase, max_count=max_count, count_type=count_type)
        checkpoint_basepath = ch.get_checkpoint_basepath(num_train_steps)+'_'+phase
        sm.save_checkpoint(checkpoint_basepath)
        if args.plot:
            print('plotting last episode')
            sm.plot_last_episode()
        sys.exit()
    else:
        max_count = ch.cfg['RUN']['eval_and_checkpoint_every_steps']
        count_type = 'steps'
        if num_train_steps < ch.cfg['RUN']['total_train_steps']:
            sm, model_dict = run_agent(sm, model_dict,  phase=phase, max_count=max_count, count_type=count_type)
            num_train_steps = sm.step_number
            print("TRAINING on step number %s"%num_train_steps)
            checkpoint_basepath = ch.get_checkpoint_basepath(num_train_steps)+'_'+phase
            sm.save_checkpoint(checkpoint_basepath)
            save_models(checkpoint_basepath, model_dict)
            if args.plot:
                print('plotting last episode')
                sm.plot_last_episode()

