from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from skvideo.io import vwrite
import cv2
import sys
import numpy as np
from IPython import embed
from collections import Counter
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from torch.nn.utils.clip_grad import clip_grad_value_
#from state_managers import VQEnv
from mb_dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint, linearly_decaying_epsilon, plot_dict_losses, matplotlib_plot_all
from env import Environment
from replay import ReplayMemory
import config
sys.path.append('../models')

from train_atari_vqvae_diff_action_reward import prepare_state_for_vq, train_vqvae, VQVAErl, run_vqvae, save_vqvae

def handle_checkpoint(cnt):
    st = time.time()
    print("beginning checkpoint", st)
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
    return cnt

#def full_state_norm_function(state):
#    return  torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])

#def mb_state_norm_function(latent_state):
#    return latent_state.long().to(info['DEVICE'])
#    return latent_state.to(info['DEVICE'])

def get_action(policy_net, state, active_head=None):
    # run on all heads to get values
    policy_net.eval()
    with torch.no_grad():
        vals = policy_net(state, None)
    if active_head is not None:
        action = torch.argmax(vals[active_head]).item()
    else:
        # vote
        acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
        data = Counter(acts)
        action = data.most_common(1)[0][0]
    return action, torch.stack(vals)[:,0].detach().cpu().numpy()

def pt_latent_learn(vqvae_model, opt):# latent_states, actions, rewards, latent_next_states, terminal_flags, masks):
    opt.zero_grad()
    batch = replay_memory.get_minibatch(info['BATCH_SIZE'])
    terminal_flags = torch.Tensor(batch[4].astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(batch[5].astype(np.int)).to(info['DEVICE'])
    # do losses based on next state estimate
    # backward() for state_representation happens in train_vqvae - still need to
    # step
    avg_vq_train_losses, next_z_q_x, next_z_e_x, pt_data = train_vqvae(vqvae_model, info, batch)
    states, actions, rewards, next_states = pt_data
    rewards = rewards.float()
    # get representation for current state
    x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(states)
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    q_policy_vals = policy_net(z_e_x, None)
    next_q_target_vals = target_net(next_z_e_x, None)
    next_q_policy_vals = policy_net(next_z_e_x, None)
    z_e_x.retain_grad()
    next_z_e_x.retain_grad()
    # trying to work on e rather than q to look at grads
    # z_q_x dows not give .grads to vqvae_model
    #q_policy_vals = policy_net(z_q_x, None)
    #next_q_target_vals = target_net(next_z_q_x, None)
    #next_q_policy_vals = policy_net(next_z_q_x, None)
    #z_q_x.retain_grad()
    #next_z_q_x.retain_grad()
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
            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            full_loss = masks[:,k]*l1loss
            loss = torch.sum(full_loss/total_used)
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    vqvae_parameters = list(vqvae_model.parameters())
    clip_grad_value_(vqvae_parameters, 5)
    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    return vqvae_model, opt, np.mean(losses)+np.sum(avg_vq_train_losses), avg_vq_train_losses

def train_sim(step_number, last_save, vqvae_model, opt):
    """Contains the training and evaluation loops"""
    print("starting agent training at step", step_number)
    avg_vq_train_losses = [0.0 for x in range(8)]
    epoch_num = len(perf['steps'])
    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < info['EVAL_FREQUENCY']:
            st = time.time()
            terminal = False
            life_lost = True
            # use real state
            state = env.reset()
            #latent_state, x_d = vqenv.get_state_representation(state[None])[None]
            #latent_state, x_d = vqenv.get_state_representation(state[None])[None]
            state_input = prepare_state_for_vq(state[None], info['DEVICE'], info['NORM_BY'])
            with torch.no_grad():
                x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(state_input)
            start_steps = step_number
            episode_reward_sum = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            ptloss_list = []
            print("Gathering data with head=%s"%active_head)
            # at every new episode - recalculate action/reward weight
            rewards_weight = 1-np.array(replay_memory.percentages_rewards(range(info['num_rewards'])))
            actions_weight = 1-np.array(replay_memory.percentages_actions(range(info['num_actions'])))
            actions_weight = torch.FloatTensor(actions_weight).to(info['DEVICE'])
            rewards_weight = torch.FloatTensor(rewards_weight).to(info['DEVICE'])
            info['actions_weight'] = actions_weight
            info['rewards_weight'] = rewards_weight
            while not terminal:
                eps = random_state.rand()
                if eps < info['EPS_INIT']:
                    action = random_state.randint(0, env.num_actions)
                    print("random action eval", action)
                else:
                    action, state_value = get_action(policy_net=policy_net, state=z_e_x, active_head=active_head)
                next_state, reward, life_lost, terminal = env.step(action)
                next_state_input = prepare_state_for_vq(next_state[None], info['DEVICE'], info['NORM_BY'])
                with torch.no_grad():
                    x_d, z_e_x, z_e_x, latents, pred_actions, pred_rewards = vqvae_model(next_state_input)
                #next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
                # Store transition in the replay memory
                #TODO - add latents from initial training buffer to replay buffer
                replay_memory.add_experience(action=action,
                                             frame=next_state[-1],
                                             reward=np.sign(reward),
                                             terminal=life_lost,
                                             )
                step_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                if step_number == info['MIN_STEPS_TO_LEARN']:
                    print("TRAIN VQVAE")
                    #TODO
                    last_save = handle_checkpoint(step_number)
                    valid_batch = valid_replay_memory.get_minibatch(info['VQ_BATCH_SIZE'])
                    save_vqvae(info, info['vq_train_cnt'], vqvae_model, opt, avg_vq_train_losses, valid_batch)
                    vqvae_model, opt = run_vqvae(info, vqvae_model, opt, replay_memory, valid_replay_memory,
                                                 num_samples_to_train=30000000, save_every_samples=250000, batches=0)
                elif step_number > info['MIN_STEPS_TO_LEARN']:
                    if step_number % info['LEARN_EVERY_STEPS'] == 0:
                        #_latent_states, _actions, _rewards, _latent_next_states, _terminal_flags, _masks,_latent_states, _latent_next_states  = replay_memory.get_minibatch(info['BATCH_SIZE'])
                        vqvae_model, opt, ptloss, avg_vq_train_losses = pt_latent_learn(vqvae_model, opt)#_latent_states, _actions, _rewards, _latent_next_states, _terminal_flags, _masks)
                        info['vq_train_cnt'] += info['VQ_BATCH_SIZE']
                        ptloss_list.append(ptloss)
                    if step_number % info['TARGET_UPDATE'] == 0:
                        print("++++++++++++++++++++++++++++++++++++++++++++++++")
                        print('updating target network at %s'%step_number)
                        target_net.load_state_dict(policy_net.state_dict())


            print('END EPISODE', epoch_num, step_number, episode_reward_sum)
            et = time.time()
            ep_time = et-st
            perf['steps'].append(step_number)
            perf['episode_step'].append(step_number-start_steps)
            perf['episode_head'].append(active_head)
            #perf['eps_list'].append(np.mean(ep_eps_list))
            if len(ptloss_list):
                lmean = np.mean(ptloss_list)
            else:
                lmean = 0.0
            perf['episode_loss'].append(lmean)
            perf['episode_reward'].append(episode_reward_sum)
            perf['head_rewards'][active_head].append(episode_reward_sum)
            perf['episode_times'].append(ep_time)
            perf['episode_relative_times'].append(time.time()-info['START_TIME'])
            perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))

            if not epoch_num or (step_number-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
                if not epoch_num or step_number > info['MIN_STEPS_TO_LEARN']:
                    last_save = handle_checkpoint(step_number)

            if not epoch_num or not epoch_num%info['PLOT_EVERY_EPISODES']:
                matplotlib_plot_all(perf, model_base_filedir)
                # TODO plot title
                print('avg reward', perf['avg_rewards'][-1])
                print('last rewards', perf['episode_reward'][-info['PLOT_EVERY_EPISODES']:])
                with open('rewards.txt', 'a') as reward_file:
                    print(len(perf['episode_reward']), step_number, perf['avg_rewards'][-1], file=reward_file)
            epoch_num += 1
        avg_eval_reward = evaluate(step_number, vqvae_model)
        perf['eval_rewards'].append(avg_eval_reward)
        perf['eval_steps'].append(step_number)

def evaluate(step_number, vqvae_model):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    for eval_run in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()

        state_input = prepare_state_for_vq(state[None], info['DEVICE'], info['NORM_BY'])
        with torch.no_grad():
            x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(state_input)
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        frames_for_gif = []
        results_for_eval = []
        x_ds = []
        rec_frames_for_gif = []
        evaluate_step_number = 0
        actions = [[-1,-1] for x in range(4)]
        exp_dir = os.path.join(model_base_filedir, "eval_step%010d" %step_number)
        os.makedirs(exp_dir)
        while not terminal:
            eps = random_state.rand()
            if eps < info['EPS_EVAL']:
               action = random_state.randint(0, env.num_actions)
               print("random action eval", action)
            else:
               action, mb_state_value = get_action(policy_net=policy_net, state=z_e_x, active_head=None)
               #action = expert_action
            actions.append(action)
            next_state, reward, life_lost, terminal = env.step(action)
            if not evaluate_step_number%100:
                print('eval,step,action,reward')
                print(evaluate_step_number,action,reward)
            next_state = prepare_state_for_vq(next_state[None], info['DEVICE'], info['NORM_BY'])
            #next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
            with torch.no_grad():
                x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(next_state)
            #if not eval_run:
            #    if evaluate_step_number < 300:
            #        #hx_d,pa,pr=vqenv.decode_vq_from_latents(latent_state)
            #        hmean = (vqenv.sample_mean_from_latents(x_d)[:,0]*255).astype(np.uint8)
            #        f,ax = plt.subplots(2,4, figsize=(4.5,1.5))
            #        for i in range(hmean.shape[0]):
            #            ax[0,i].imshow(hmean[i])
            #            ax[0,i].set_title('a%s' %(actions[i-4]))
            #            ax[0,i].axis('off')
            #            ax[1,i].imshow(next_state[i])
            #            ax[1,i].axis('off')
            #        fname = os.path.join(exp_dir, "%06d.png"%evaluate_step_number)
            #        plt.savefig(fname)
            #        plt.close()

            episode_steps +=1
            episode_reward_sum += reward
            if not eval_run:
                # add current observation
                frames_for_gif.append(cv2.resize(env.ale.getScreenRGB(), (80, 100)).astype(np.uint8) )
                #x_ds.append(x_d)
                #if len(x_ds) >= info['BATCH_SIZE']:
                #    # todo - stop converting back and forth from np and
                #    # precalculate array sizes
                #    rec_frames_for_gif.extend(reconstruct_from_latents(x_ds))
                #    x_ds = []

                if len(frames_for_gif) >= 1000:
                    # create videos from previous 1000 frames
                    create_video(frames_for_gif, 'obs', step_number, evaluate_step_number, episode_reward_sum)
                    create_video(rec_frames_for_gif, 'rec', step_number, evaluate_step_number, episode_reward_sum)
                    # reset lists
                    frames_for_gif = []
                    rec_frames_for_gif = []

            results_for_eval.append("%s, %s, %s, %s" %(action, reward, life_lost, terminal))
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
            evaluate_step_number += 1

        if not eval_run:
            #if len(x_ds):
            #    rec_frames_for_gif.extend(reconstruct_from_latents(x_ds))
            create_video(frames_for_gif, 'obs', step_number, evaluate_step_number, episode_reward_sum)
            #create_video(rec_frames_for_gif, 'rec', step_number, evaluate_step_number, episode_reward_sum)
            frames_for_gif = []
            rec_frames_for_gif = []
        print("Evaluation score:\n", episode_reward_sum)
        #efile = os.path.join(model_base_filedir, 'eval_rewards_%010d_%s.txt'%(step_number, i))
        #with open(efile, 'a') as eval_reward_file:
        #    print(step_number, np.mean(eval_rewards), file=eval_reward_file)
        eval_rewards.append(episode_reward_sum)
    avg_reward = np.sum(eval_rewards)
    print("sum of eval rewards", avg_reward)
    if avg_reward != 0:
        avg_reward /= float(len(eval_rewards))
    return avg_reward

#def reconstruct_from_latents(x_ds):
#    pt_latents = torch.stack(x_ds)[:,0]
#    return list((vqenv.sample_mean_from_latents(pt_latents)[:,0]*255).astype(np.uint8))

def create_video(frames, name, train_step_number, last_index, episode_reward_sum):
    if len(frames):
        tmp4_fname = os.path.join(model_base_filedir, "ATARI_step%010d_%s_%06d_%03d.mp4"%(train_step_number, name, last_index, episode_reward_sum))
        vwrite(tmp4_fname, np.array(frames, dtype=np.uint8))

def init_vq_model(info):
    if info['VQ_MODEL_LOADPATH'] == '':
         run_num = 0
         vq_model_base_filedir = os.path.join(info['model_base_filedir'], 'VQ%02d'%run_num)
         while os.path.exists(vq_model_base_filedir):
             run_num +=1
             vq_model_base_filedir = os.path.join(info['model_base_filedir'], 'VQ%02d'%run_num)
         os.makedirs(vq_model_base_filedir)
         #model_base_filepath = os.path.join(model_base_filedir, args.savename)
         #print("MODEL BASE FILEPATH", model_base_filepath)
         info['BETA']=0.25
         info['ALPHA_REC']=1.0
         info['ALPHA_ACT']=2.0
         info['ALPHA_REW']=1.0
         info['NUM_Z']=64
         info['NUM_K']=512
         info['NR_LOGISTIC_MIX']=10
         info['VQ_BATCH_SIZE']=64
         info['NUMBER_CONDITION']=4
         info['VQ_LEARNING_RATE']=1e-4
         info['NUM_SAMPLES']=40
         info['VQ_NUM_EXAMPLES_TO_TRAIN']=10000000
         info['VQ_SAVE_EVERY']=250000
         info['VQ_MIN_BATCHES_BEFORE_SAVE']=1000
         info['LATENT_SIZE']=10
         info['vq_train_cnt'] = 0
         info['vq_train_cnts'] = []
         info['vq_train_losses_list'] = []
         info['vq_valid_cnts'] = []
         info['vq_valid_losses_list'] = []
         info['vq_save_times'] = []
         info['vq_last_save'] = 0
         info['vq_last_plot'] = 0
    else:
        print('loading model from: %s' %info['VQ_MODEL_LOADPATH'])
        model_dict = torch.load(info['VQ_MODEL_LOADPATH'])
        vq_model_base_filedir = os.path.split(info['VQ_MODEL_LOADPATH'] )[0]

        # load previously loaded vq
        vqinfo = model_dict['vq_info']
        info['vq_train_cnt'] = vqinfo['vq_train_cnts'][-1]
        info['vq_loaded_from'] = info['VQ_MODEL_LOADPATH']

        old_keys = info.keys()
        for key in vqinfo.keys():
            print('loading vq pararmeter into info:%s'%key)
            if key in old_keys:
                if type(vqinfo[key]) is not list:
                    print('was', info[key])
                    print('now', vqinfo[key])
            info[key] = vqinfo[key]
        #if 'reward_weights' not in info.keys():
        #    info['reward_weights'] = [1,100]
    # create replay buffer
    info['VQ_MODEL_BASE_FILEDIR'] = vq_model_base_filedir
    # old name is some places in code - should fix this
    info['vq_model_base_filepath'] = vq_model_base_filedir
    #train_buffer = ReplayMemory(load_file=train_data_file)
    #valid_buffer = ReplayMemory(load_file=valid_data_file)

    #info['num_actions'] = train_buffer.num_actions()
    #info['size_training_set'] = train_buffer.num_examples()
    #info['hsize'] = train_buffer.frame_height
    #info['wsize'] = train_buffer.frame_width
    #info['num_rewards'] = train_buffer.num_rewards()

    #rewards_weight = 1-np.array(train_buffer.percentages_rewards())
    #actions_weight = 1-np.array(train_buffer.percentages_actions())
    #actions_weight = torch.FloatTensor(actions_weight).to(DEVICE)
    #rewards_weight = torch.FloatTensor(rewards_weight).to(DEVICE)
    #info['actions_weight'] = actions_weight
    #info['rewards_weight'] = rewards_weight

    # output mixtures should be 2*nr_logistic_mix + nr_logistic mix for each
    # decorelated channel
    info['num_output_mixtures']= (2*info['NR_LOGISTIC_MIX']+info['NR_LOGISTIC_MIX'])*info['HISTORY_SIZE']
    nmix = int(info['num_output_mixtures']/info['HISTORY_SIZE'])
    info['nmix'] = nmix
    vqvae_model = VQVAErl(num_clusters=info['NUM_K'],
                        encoder_output_size=info['NUM_Z'],
                        num_output_mixtures=info['num_output_mixtures'],
                        in_channels_size=info['NUMBER_CONDITION'],
                        n_actions=info['num_actions'],
                        int_reward=info['num_rewards'],
                        ).to(info['DEVICE'])

    #parameters = list(vqvae_model.parameters())
    #opt = optim.Adam(parameters, lr=info['VQ_LEARNING_RATE'])
    if args.model_loadpath != '':
        print("loading weights from:%s" %args.model_loadpath)
        vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
        #opt.load_state_dict(model_dict['vq_optimizer'])
        vqvae_model.embedding = model_dict['vq_embedding']
    return info, vqvae_model


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
        "N_PLAYOUT":50,
        "MIN_SCORE_GIF":-1, # min score to plot gif in eval
        "DEVICE":device, #cpu vs gpu set by argument
        #"NAME":"MBFreeway_replay",
        "NAME":"MBBreakout_train_init",
        "DUELING":True, # use dueling dqn
        "DOUBLE_DQN":True, # use double dqn
        "PRIOR":True, # turn on to use randomized prior
        "PRIOR_SCALE":10, # what to scale prior by
        "N_ENSEMBLE":9, # number of bootstrap heads to use. when 1, this is a normal dqn
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE":10000, # how often to update target network
        # 500000 may be too much
        # could consider each of the heads once
        #"MIN_STEPS_TO_LEARN":100000, # min steps needed to start training neural nets
        "MIN_STEPS_TO_LEARN":40000, # min steps needed to start training neural nets
        #"MIN_STEPS_TO_LEARN":400, # min steps needed to start training neural nets
        "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
        "VQ_LEARN_EVERY_STEPS":6400, # update every 6400 steps in simple
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        # I think this randomness might need to be higher
        "EPS_INIT":0.01,
        "EPS_FINAL":0.01, # 0.01 in osband
        "EPS_EVAL":0.0, # 0 in osband, .05 in others....
        "NUM_EVAL_EPISODES":1, # num examples to average in eval
        #"BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "BUFFER_SIZE":int(500000), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":50000, # how often to write pkl of model and npz of data buffer
        #"CHECKPOINT_EVERY_STEPS":1e6, # how often to write pkl of model and npz of data buffer
        #"EVAL_FREQUENCY":500000, # how often to run evaluation episodes
        "EVAL_FREQUENCY":100000, # how often to run evaluation episodes
        #"EVAL_FREQUENCY":1, # how often to run evaluation episodes
        "ADAM_LEARNING_RATE":6.25e-5,
        #"ADAM_LEARNING_RATE":1e-4,
        #"RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        #"RMS_DECAY":0.95,
        #"RMS_MOMENTUM":0.0,
        #"RMS_EPSILON":0.00001,
        #"RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":64, # Batch size to use for learning
        "GAMMA":.99, # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 5,
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":14342,
        "RANDOM_HEAD":-1, # just used in plotting as demarcation
        "OBS_SIZE":(84,84),
        "RESHAPE_SIZE":10*10*16,
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP":4, # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
        "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life
        "REWARD_SPACE":[0,1], #[-1,0,1]
         ##################### for vqvae model
        #"VQ_MODEL_LOADPATH":'../../model_savedir/MBR01/MBvqbt01/MBvqbt_0033756480ex.pt',
        # worked on poorly trained model below
        #"VQ_MODEL_LOADPATH":'../../model_savedir/MBvqbt_reward_0041007872ex.pt',
        #"VQ_MODEL_LOADPATH":'../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0035503692ex.pt',
        #"VQ_MODEL_LOADPATH":"../../model_savedir/MBBreakout00/BreakoutVQ02/BreakoutVQ_0049509504ex.pt",
        #"VQ_MODEL_LOADPATH":"../../model_savedir/MBBreakout_init_dataset/BreakoutVQ02/BreakoutVQ_0103019776ex.pt",
        #"VQ_MODEL_LOADPATH":"../../model_savedir/MBBreakout_init_dataset/BreakoutVQ02/BreakoutVQ_0103019776ex.pt",
        #"VQ_MODEL_LOADPATH":"../../model_savedir/MBFreeway_init_train/FreewayR4LR1e400/FreewayR4LR1e4_0029255680ex.pt",
        "VQ_MODEL_LOADPATH":"",#"../../model_savedir/MBFreeway_init_train/FreewayR4LR1e400/FreewayR4LR1e4_0058261248ex.pt",
        "REPLAY_MEMORY_LOADPATH":"", #"../../model_savedir/MBFreeway_init_train/MBFreeway_data_0000040001q_train_buffer.npz",
        "REPLAY_MEMORY_VALID_LOADPATH":"/usr/local/data/jhansen/planning/model_savedir/MBBreakout_valid_init00/MBBreakout_valid_init_0000005000q_train_buffer.npz",
        "USE_EMBEDDING":False,# if use_embedding in ensemble model - input should be float, otherwise long
        "FORWARD_DROPOUT":0.25,
        "FORWARD_LEARNING_RATE":1e-5,
        "FORWARD_BATCH_SIZE":0.25,
    }

    start_step_number = 0
    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])
    info['num_rewards'] = len(info['REWARD_SPACE'])

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    valid_replay_memory = ReplayMemory(load_file=info['REPLAY_MEMORY_VALID_LOADPATH'])
    # create replay buffer
    if info['REPLAY_MEMORY_LOADPATH'] == "":
        replay_memory = ReplayMemory(action_space=env.action_space,
                                 size=info['BUFFER_SIZE'],
                                 frame_height=info['OBS_SIZE'][0],
                                 frame_width=info['OBS_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'],
                                 #latent_frame_height=info['LATENT_SIZE'],
                                 #latent_frame_width=info['LATENT_SIZE'])
                                  )

    else:
        replay_memory = ReplayMemory(load_file=info['REPLAY_MEMORY_LOADPATH'])
        valid_replay_memory = ReplayMemory(load_file=info['REPLAY_MEMORY_VALID_LOADPATH'])
        start_step_number = replay_memory.count
    random_state = np.random.RandomState(info["SEED"])

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
                'eval_steps':[],
                'head_rewards':[[] for x in range(info['N_ENSEMBLE'])],
                }

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


    info['model_base_filepath'] = model_base_filepath
    info['model_base_filedir'] = model_base_filedir
    info['num_actions'] = env.num_actions
    info['action_space'] = range(info['num_actions'])

    #vqenv = VQEnv(info, vq_model_loadpath=info['VQ_MODEL_LOADPATH'], device='cpu')
    #vq_model_dict = torch.load(info['VQ_MODEL_LOADPATH'], map_location=lambda storage, loc: storage)
    ############################################VQ##################################
    info, vqvae_model = init_vq_model(info)

    ############################################VQ##################################

    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      #num_channels=info['HISTORY_SIZE'],
                                      num_channels=info['NUM_Z'],
                                      dueling=info['DUELING'],
                                      num_clusters=info['NUM_K'],
                                      use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      #num_channels=info['HISTORY_SIZE'],
                                      num_channels=info['NUM_Z'],
                                      dueling=info['DUELING'],
                                      num_clusters=info['NUM_K'],
                                      use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])
    if info['PRIOR']:
        prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                reshape_size=info['RESHAPE_SIZE'],
                                #num_channels=info['HISTORY_SIZE'],
                                num_channels=info['NUM_Z'],
                                dueling=info['DUELING'],
                                num_clusters=info['NUM_K'],
                                use_embedding=info['USE_EMBEDDING']).to(info['DEVICE'])

        print("using randomized prior")
        policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
        target_net = NetWithPrior(target_net, prior_net, info['PRIOR_SCALE'])

    target_net.load_state_dict(policy_net.state_dict())
    # create optimizer
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])

    parameters = list(policy_net.parameters())+list(vqvae_model.parameters())
    opt = optim.Adam(parameters, lr=info['ADAM_LEARNING_RATE'])
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

    train_sim(start_step_number, start_last_save, vqvae_model, opt)

