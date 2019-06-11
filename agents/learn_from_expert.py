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
from state_managers import VQEnv
from mb_dqn_model import EnsembleNet as mbEnsembleNet
from mb_dqn_model import NetWithPrior as mbNetWithPrior
from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, generate_video, save_checkpoint, linearly_decaying_epsilon, matplotlib_plot_all
from env import Environment
from replay import ReplayMemory
import config

#def pt_latent_learn(latent_states, actions, rewards, latent_next_states, terminal_flags, masks):
#    #latent_states = torch.Tensor(latent_states[:,-1:].astype(np.float)/float(info['NUM_K'])).to(info['DEVICE'])
#    #latent_next_states = torch.Tensor(latent_next_states[:,-1:].astype(np.float)/float(info['NUM_K'])).to(info['DEVICE'])
#    # dont normalize because we are using embedding
#    latent_states = torch.LongTensor(latent_states).to(info['DEVICE'])
#    latent_next_states = torch.LongTensor(latent_next_states).to(info['DEVICE'])
#    rewards = torch.Tensor(rewards).to(info['DEVICE'])
#    actions = torch.LongTensor(actions).to(info['DEVICE'])
#    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
#    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
#    # min history to learn is 200,000 frames in dqn - 50000 steps
#    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
#    opt.zero_grad()
#    q_policy_vals = mb_policy_net(latent_states, None)
#    next_q_target_vals = mb_target_net(latent_next_states, None)
#    next_q_policy_vals = mb_policy_net(latent_next_states, None)
#    cnt_losses = []
#    for k in range(info['N_ENSEMBLE']):
#        #TODO finish masking
#        total_used = torch.sum(masks[:,k])
#        if total_used > 0.0:
#            next_q_vals = next_q_target_vals[k].data
#            if info['DOUBLE_DQN']:
#                next_actions = next_q_policy_vals[k].data.max(1, True)[1]
#                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
#            else:
#                next_qs = next_q_vals.max(1)[0] # max returns a pair
#
#            preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
#            targets = rewards + info['GAMMA'] * next_qs * (1-terminal_flags)
#            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
#            full_loss = masks[:,k]*l1loss
#            loss = torch.sum(full_loss/total_used)
#            cnt_losses.append(loss)
#            losses[k] = loss.cpu().detach().item()
#
#    loss = sum(cnt_losses)/info['N_ENSEMBLE']
#    loss.backward()
#    for param in mb_policy_net.core_net.parameters():
#        if param.grad is not None:
#            # divide grads in core
#            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
#    nn.utils.clip_grad_norm_(mb_policy_net.parameters(), info['CLIP_GRAD'])
#    opt.step()
#    return np.mean(losses)


def handle_checkpoint(cnt):
    st = time.time()
    print("beginning checkpoint", st)
    state = {'info':info,
             'optimizer':opt.state_dict(),
             'cnt':cnt,
             'policy_net_state_dict':mb_policy_net.state_dict(),
             'target_net_state_dict':mb_target_net.state_dict(),
             'perf':perf,
            }
    filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
    save_checkpoint(state, filename)
    # npz will be added
    buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer"%cnt)
    replay_memory.save_buffer(buff_filename)
    print("finished checkpoint", time.time()-st)
    return cnt


def full_state_norm_function(state):
    return  torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])

def mb_state_norm_function(latent_state):
    return latent_state.long().to(info['DEVICE'])

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

def kl_latent_learn(step_number):
    _states, _actions, _rewards, _next_states, _terminal_flags, _masks, _latent_states, _latent_next_states = replay_memory.get_minibatch(info['BATCH_SIZE'])
    states = full_state_norm_function(_states)[0]
    latent_states = mb_state_norm_function(torch.Tensor(_latent_states))
    expert_policy_net.eval()
    with torch.no_grad():
        expert_values = expert_policy_net(states, None)
    mb_q_policy_vals = mb_policy_net(latent_states, None)
    # referenced -
    # https://github.com/NervanaSystems/distiller/blob/master/distiller/knowledge_distillation.py#L148
    distillation_losses = []
    for head in range(info['N_ENSEMBLE']):
        dloss = F.kl_div(mb_q_policy_vals[head], expert_values[head].detach(), reduction='batchmean')
        distillation_losses.append(dloss)

    loss = sum(distillation_losses)/info['N_ENSEMBLE']
    loss.backward()
    for param in mb_policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(mb_policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    ind = np.random.randint(0,_latent_states.shape[0])
    hx_d,pa,pr=vqenv.decode_vq_from_latents(latent_states[ind].cpu())
    hmean = (vqenv.sample_mean_from_latents(hx_d)[:,0]*255).astype(np.uint8)
    f,ax = plt.subplots(2,4, figsize=(4.5,1.5))
    for i in range(hmean.shape[0]):
        expert_action, expert_state_value = get_action(policy_net=expert_policy_net, state=states[ind][None], active_head=None)
        mb_action, mb_state_value = get_action(policy_net=mb_policy_net, state=latent_states[ind][None], active_head=None)
        ax[0,i].imshow(hmean[i])
        ax[0,i].axis('off')
        ax[1,i].imshow(_states[ind][i])
        ax[1,i].axis('off')
    ax[0,i].set_title('e%sm%s' %(expert_action, mb_action))
    fname = os.path.join(model_base_filedir, "train_step%010d.png"%step_number)
    plt.savefig(fname)
    plt.close()
    return float(loss)

def train_student(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        same = [0 for pp in range(1000)]
        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal = False
            life_lost = True
            # use real state
            state = env.reset()
            latent_state, x_d = vqenv.get_state_representation(state[None])
            latent_hist_state = torch.stack((latent_state, latent_state, latent_state, latent_state), dim=1)
            start_steps = step_number
            st = time.time()
            episode_reward_sum = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            ptloss_list = []
            print("Gathering data with head=%s"%active_head)
            while not terminal:
               # eps
                eps = random_state.rand()
                if eps < info['EPS_INIT']:
                    action = random_state.randint(0, env.num_actions)
                    print(step_number, "random action train", action)
                else:
                    expert_action, expert_state_value = get_action(policy_net=expert_policy_net, state=full_state_norm_function(state), active_head=active_head)
                    mb_action, mb_state_value = get_action(policy_net=mb_policy_net, state=mb_state_norm_function(latent_hist_state), active_head=active_head)
                    if mb_action==expert_action:
                        same.append(1)
                    else:
                        same.append(0)
                    same = same[1:]
                next_state, reward, life_lost, terminal = env.step(expert_action)
                next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
                # Store transition in the replay memory
                #TODO - add latents from initial training buffer to replay buffer
                replay_memory.add_experience(action=expert_action,
                                             frame=next_state[-1],
                                             reward=np.sign(reward),
                                             terminal=life_lost,
                                             latent_frame=next_latent_state[0].cpu().numpy())
                latent_hist_state = torch.cat((latent_hist_state[:,1:], next_latent_state[:,None]), dim=1)

                step_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                state = next_state
                if step_number > info['MIN_STEPS_TO_LEARN']:
                    if step_number % info['LEARN_EVERY_STEPS'] == 0:
                        ptloss = kl_latent_learn(step_number)
                        ptloss_list.append(ptloss)
                    if step_number % info['TARGET_UPDATE'] == 0:
                        print("++++++++++++++++++++++++++++++++++++++++++++++++")
                        print('updating target network at %s'%step_number)
                        mb_target_net.load_state_dict(mb_policy_net.state_dict())
            print('END EPISODE', epoch_num, step_number, episode_reward_sum)
            print("SAME/1000", np.sum(same))
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
        avg_eval_reward = evaluate(step_number)
        perf['eval_rewards'].append(avg_eval_reward)
        perf['eval_steps'].append(step_number)
        matplotlib_plot_all(perf, model_base_filedir)

def reconstruct_from_latents(x_ds):
    pt_latents = torch.stack(x_ds)[:,0]
    return list((vqenv.sample_mean_from_latents(pt_latents)[:,0]*255).astype(np.uint8))

def create_video(frames, name, train_step_number, last_index, episode_reward_sum):
    if len(frames):
        tmp4_fname = os.path.join(model_base_filedir, "ATARI_step%010d_%s_%06d_%03d.mp4"%(train_step_number, name, last_index, episode_reward_sum))
        vwrite(tmp4_fname, np.array(frames, dtype=np.uint8))

def evaluate(step_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    # with 9000 steps - expert gets about 250 pts
    eval_rewards = []
    # only run one
    for eval_run in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()
        latent_state, x_d = vqenv.get_state_representation(state[None])
        latent_hist_state = torch.stack((latent_state, latent_state, latent_state, latent_state), dim=1)

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
               mb_action, expert_action = -1, -1

            else:
               expert_action, expert_state_value = get_action(policy_net=expert_policy_net, state=full_state_norm_function(state), active_head=None)
               mb_action, mb_state_value = get_action(policy_net=mb_policy_net, state=mb_state_norm_function(latent_hist_state), active_head=None)
               print(expert_state_value)
               print(mb_state_value)
               action = mb_action
               #action = expert_action
            actions.append((expert_action, mb_action))
            next_state, reward, life_lost, terminal = env.step(action)
            if not evaluate_step_number%100:
                print('eval,step,mb_action,expert_action,reward')
                print(evaluate_step_number,mb_action,expert_action,reward)
            next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
            latent_hist_state = torch.cat((latent_hist_state[:,1:], next_latent_state[:,None]), dim=1)
            #latent_hist_state is of size [1,4,10,10]
            if not eval_run:
                if evaluate_step_number < 300:
                    hx_d,pa,pr=vqenv.decode_vq_from_latents(latent_hist_state[0])
                    hmean = (vqenv.sample_mean_from_latents(hx_d)[:,0]*255).astype(np.uint8)
                    f,ax = plt.subplots(2,4, figsize=(4.5,1.5))
                    for i in range(hmean.shape[0]):
                        ax[0,i].imshow(hmean[i])
                        ax[0,i].set_title('e%sm%s' %(actions[i-4][0], actions[i-4][1]))
                        ax[0,i].axis('off')
                        ax[1,i].imshow(next_state[i])
                        ax[1,i].axis('off')
                    fname = os.path.join(exp_dir, "%06d.png"%evaluate_step_number)
                    plt.savefig(fname)
                    plt.close()

            episode_steps +=1
            episode_reward_sum += reward
            if not eval_run:
                # add current observation
                frames_for_gif.append(cv2.resize(env.ale.getScreenRGB(), (80, 100)).astype(np.uint8) )
                x_ds.append(x_d)
                if len(x_ds) >= 48:
                    # todo - stop converting back and forth from np and
                    # precalculate array sizes
                    rec_frames_for_gif.extend(reconstruct_from_latents(x_ds))
                    x_ds = []

                if len(frames_for_gif) >= 500:
                    # create videos from previous 500 frames
                    create_video(frames_for_gif, 'obs', step_number, evaluate_step_number, episode_reward_sum)
                    create_video(rec_frames_for_gif, 'rec', step_number, evaluate_step_number, episode_reward_sum)
                    # reset lists
                    frames_for_gif = []
                    rec_frames_for_gif = []

            results_for_eval.append("%s, %s, %s, %s, %s" %(mb_action, expert_action, reward, life_lost, terminal))
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
            evaluate_step_number += 1

        if not eval_run:
            if len(x_ds):
                rec_frames_for_gif.extend(reconstruct_from_latents(x_ds))
            create_video(frames_for_gif, 'obs', step_number, evaluate_step_number, episode_reward_sum)
            create_video(rec_frames_for_gif, 'rec', step_number, evaluate_step_number, episode_reward_sum)
            frames_for_gif = []
            rec_frames_for_gif = []
        print("Evaluation score:\n", episode_reward_sum)
        efile = os.path.join(model_base_filedir, 'eval_rewards_%010d_%s.txt'%(step_number, i))
        with open(efile, 'a') as eval_reward_file:
            print(step_number, np.mean(eval_rewards), file=eval_reward_file)
        eval_rewards.append(episode_reward_sum)
    avg_reward = np.sum(eval_rewards)
    print("sum of eval rewards", avg_reward)
    if avg_reward != 0:
        avg_reward /= float(len(eval_rewards))
    return avg_reward

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
        #"NAME":'MBReward_RUN_rerunwithnewstatemanager', # start files with name
        #"NAME":'MBReward_RUN_rerunwithnewstatemanager_fullytrainedvqvae_lower_checkpoint', # start files with name
        #"NAME":'MBReward_embedding_hist_SEED14_GAMMAp99_prior1MLReps', # start files with name
        "NAME":"MBBreakout_learn_from_expert",
        #"NAME":"DEBUG",
        "EXPERT":"../../model_savedir/BreakoutNewActionAnnealingPRIOR00/BreakoutNewActionNoAnnealingPRIOR_0002501995q.pkl",
        #"REPLAY_BUFFER_LOADPATH":"../../model_savedir/BreakoutNewActionAnnealingPRIOR00/BreakoutNewActionNoAnnealingPRIOR_0001501367q_train_buffer.npz",
        "DUELING":True, # use dueling dqn
        "DOUBLE_DQN":True, # use double dqn
        "PRIOR":True, # turn on to use randomized prior
        "PRIOR_SCALE":1, # what to scale prior by
        "N_ENSEMBLE":9, # number of bootstrap heads to use. when 1, this is a normal dqn
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE":10000, # how often to update target network
        # 500000 may be too much
        # could consider each of the heads once
        #"MIN_STEPS_TO_LEARN":100000, # min steps needed to start training neural nets
        "MIN_STEPS_TO_LEARN":500, # min steps needed to start training neural nets
        "LEARN_EVERY_STEPS":1, # updates every 4 steps in osband
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        # I think this randomness might need to be higher
        "EPS_INIT":0.01,
        "EPS_FINAL":0.01, # 0.01 in osband
        "EPS_EVAL":0.1, # 0 in osband, .05 in others....
        "NUM_EVAL_EPISODES":1, # num examples to average in eval
        #"BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "BUFFER_SIZE":int(500000), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":500000, # how often to write pkl of model and npz of data buffer
        #"CHECKPOINT_EVERY_STEPS":1e6, # how often to write pkl of model and npz of data buffer
        #"EVAL_FREQUENCY":500000, # how often to run evaluation episodes
        "EVAL_FREQUENCY":50000, # how often to run evaluation episodes
        #"EVAL_FREQUENCY":1, # how often to run evaluation episodes
        #"ADAM_LEARNING_RATE":6.25e-5,
        "ADAM_LEARNING_RATE":1e-4,
        "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":64, # Batch size to use for learning
        "GAMMA":.99, # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 5,
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":14,
        "RANDOM_HEAD":-1, # just used in plotting as demarcation
        "OBS_SIZE":(84,84),
        "RESHAPE_SIZE":10*10*16,
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        #"MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "MAX_EPISODE_STEPS":9000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
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
        "VQ_MODEL_LOADPATH":"../../model_savedir/MBBreakout_init_dataset/BreakoutVQ02/BreakoutVQ_0103269824ex.pt",
        "BETA":0.25,
        "ALPHA_REC":1.0,
        "ALPHA_ACT":2.0,
        "NUM_Z":64,
        "NUM_K":512,
        "NR_LOGISTIC_MIX":10,
        "VQ_BATCH_SIZE":84,
        "NUMBER_CONDITION":4,
        # learning rate can probably go higher than 2e-4
        "VQ_LEARNING_RATE":2e-5,
        "NUM_SAMPLES":40,
        "VQ_NUM_EXAMPLES_TO_TRAIN":100000000,
        "VQ_SAVE_EVERY":500000,
        "VQ_MIN_BATCHES_BEFORE_SAVE":1000,
        "LATENT_SIZE":10,
        "VQ_SAVENAME":"VQ",
        "VQ_GAMMA":0.9,
    }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])
    info['num_rewards'] = len(info['REWARD_SPACE'])

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    # create replay buffer
    replay_memory = ReplayMemory(action_space=env.action_space,
                                 size=info['BUFFER_SIZE'],
                                 frame_height=info['OBS_SIZE'][0],
                                 frame_width=info['OBS_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'],
                                 latent_frame_height=info['LATENT_SIZE'],
                                 latent_frame_width=info['LATENT_SIZE'])
   # latent_replay_memory = ReplayMemory(action_space=env.action_space,
   #                              size=info['BUFFER_SIZE'],
   #                              frame_height=info['LATENT_SIZE'],
   #                              frame_width=info['LATENT_SIZE'],
   #                              agent_history_length=info['HISTORY_SIZE'],
   #                              batch_size=info['BATCH_SIZE'],
   #                              num_heads=info['N_ENSEMBLE'],
   #                              bernoulli_probability=info['BERNOULLI_PROBABILITY'])


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


    info['model_base_filepath'] = model_base_filepath
    info['num_actions'] = env.num_actions
    info['action_space'] = range(info['num_actions'])

    vqenv = VQEnv(info, vq_model_loadpath=info['VQ_MODEL_LOADPATH'], device='cpu')

    mb_policy_net = mbEnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                      num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])
    mb_target_net = mbEnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                      num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])
    if info['PRIOR']:
        mb_prior_net = mbEnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                reshape_size=info['RESHAPE_SIZE'],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])

        print("using randomized prior")
        mb_policy_net = mbNetWithPrior(mb_policy_net, mb_prior_net, info['PRIOR_SCALE'])
        mb_target_net = mbNetWithPrior(mb_target_net, mb_prior_net, info['PRIOR_SCALE'])

    mb_target_net.load_state_dict(mb_policy_net.state_dict())
    # create optimizer
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])
    opt = optim.Adam(mb_policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])



    ########################################expert#############################
    expert_model_dict = torch.load(info['EXPERT'])
    expert_info = expert_model_dict['info']
    if 'RESHAPE_SIZE' not in expert_info.keys():
        expert_info['RESHAPE_SIZE'] = 64*7*7
    expert_policy_net = EnsembleNet(n_ensemble=expert_info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=expert_info['RESHAPE_SIZE'],
                                      num_channels=expert_info['HISTORY_SIZE'],
                                      dueling=expert_info['DUELING'],
                                      ).to(info['DEVICE'])
    expert_target_net = EnsembleNet(n_ensemble=expert_info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=expert_info['RESHAPE_SIZE'],
                                      num_channels=expert_info['HISTORY_SIZE'],
                                      dueling=expert_info['DUELING'],
                                      ).to(info['DEVICE'])
    if info['PRIOR']:
        expert_prior_net = EnsembleNet(n_ensemble=expert_info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                reshape_size=expert_info['RESHAPE_SIZE'],
                                num_channels=expert_info['HISTORY_SIZE'], dueling=expert_info['DUELING'],
                                ).to(info['DEVICE'])

        print("using randomized prior")
        expert_policy_net = NetWithPrior(expert_policy_net, expert_prior_net, expert_info['PRIOR_SCALE'])
        expert_target_net = NetWithPrior(expert_target_net, expert_prior_net, expert_info['PRIOR_SCALE'])

    expert_target_net.load_state_dict(expert_policy_net.state_dict())
    expert_target_net.load_state_dict(expert_model_dict['target_net_state_dict'])
    expert_policy_net.load_state_dict(expert_model_dict['policy_net_state_dict'])

    train_student(start_step_number, start_last_save)

