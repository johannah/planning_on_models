from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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
from mb_dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint, linearly_decaying_epsilon, plot_dict_losses, matplotlib_plot_all
from env import Environment
from replay import ReplayMemory
import config

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


#def pt_get_latent_action(latent_state, active_head=None):
#    # comes in as vq
#    #vals = policy_net(latent_state[None].float().to(info['DEVICE'])/float(info['NUM_K']), active_head)
#    # goes in with channels = 0
#    vals = policy_net(latent_state.long().to(info['DEVICE']), active_head)
#    if active_head is not None:
#        action = torch.argmax(vals, dim=1).item()
#        return action
#    else:
#        # vote
#        acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
#        print(acts)
#        data = Counter(acts)
#        action = data.most_common(1)[0][0]
#        return action

def pt_latent_learn():# latent_states, actions, rewards, latent_next_states, terminal_flags, masks):
    #latent_states = torch.Tensor(latent_states[:,-1:].astype(np.float)/float(info['NUM_K'])).to(info['DEVICE'])
    #latent_next_states = torch.Tensor(latent_next_states[:,-1:].astype(np.float)/float(info['NUM_K'])).to(info['DEVICE'])
    # dont normalize because we are using embedding
    _states, _actions, _rewards, _next_states, _terminal_flags, _masks, _latent_states, _latent_next_states = replay_memory.get_minibatch(info['BATCH_SIZE'])
    latent_states = torch.LongTensor(_latent_states).to(info['DEVICE'])
    latent_next_states = torch.LongTensor(_latent_next_states).to(info['DEVICE'])
    rewards = torch.Tensor(_rewards).to(info['DEVICE'])
    actions = torch.LongTensor(_actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(_terminal_flags.astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(_masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    opt.zero_grad()
    q_policy_vals = policy_net(latent_states, None)
    next_q_target_vals = target_net(latent_next_states, None)
    next_q_policy_vals = policy_net(latent_next_states, None)
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
    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    return np.mean(losses)

def ptlearn(states, actions, rewards, next_states, terminal_flags, masks):
    states = torch.Tensor(states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
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
#
def train_sim(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    while step_number < info['MAX_STEPS']:
        #avg_eval_reward = evaluate(step_number)
        #perf['eval_rewards'].append(avg_eval_reward)
        #perf['eval_steps'].append(step_number)
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
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
                #if step_number < info['MIN_STEPS_TO_LEARN'] and random_state.rand() < info['EPS_INIT']:
                #    action = random_state.randint(0, env.num_actions)
                #else:
                eps = random_state.rand()
                if eps < info['EPS_INIT']:
                    action = random_state.randint(0, env.num_actions)
                    print("random action eval", action)
                else:
                    #action, value = pt_get_latent_action(latent_state=latent_hist_state, active_head=active_head)
                    action, state_value = get_action(policy_net=policy_net, state=mb_state_norm_function(latent_hist_state), active_head=active_head)
                next_state, reward, life_lost, terminal = env.step(action)
                next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
                # Store transition in the replay memory
                #TODO - add latents from initial training buffer to replay buffer
                replay_memory.add_experience(action=action,
                                             frame=next_state[-1],
                                             reward=np.sign(reward),
                                             terminal=life_lost,
                                             latent_frame=next_latent_state[0].cpu().numpy()
                                             )
                # dump oldest state
                latent_hist_state = torch.cat((latent_hist_state[:,1:], next_latent_state[:,None]), dim=1)

                step_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                state = next_state
                #if step_number == info['MIN_STEPS_TO_LEARN']:
                #    if info['VQ_MODEL_LOADPATH'] == '':
                #        # need to write npz file and train vqvae on it
                #        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer"%step_number)
                #        replay_memory.save_buffer(buff_filename)
                #        # now the vq is trained -
                #        vqenv.train_vq_model(buff_filename+'.npz')

                if step_number > info['MIN_STEPS_TO_LEARN']:
                    if step_number % info['LEARN_EVERY_STEPS'] == 0:
                        #_latent_states, _actions, _rewards, _latent_next_states, _terminal_flags, _masks,_latent_states, _latent_next_states  = replay_memory.get_minibatch(info['BATCH_SIZE'])
                        ptloss = pt_latent_learn()#_latent_states, _actions, _rewards, _latent_next_states, _terminal_flags, _masks)
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

def evaluate(step_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()
        latent_state, x_d = vqenv.get_state_representation(state[None])
        latent_hist_state = torch.stack((latent_state, latent_state, latent_state, latent_state), dim=1)

        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        frames_for_gif = []
        results_for_eval = []
        latents = []
        x_ds = []
        latents.append(latent_state.cpu().numpy())
        while not terminal:
            eps = random_state.rand()
            if eps < info['EPS_EVAL']:
               action = random_state.randint(0, env.num_actions)
               print("random action eval", action)
            else:
               action = pt_get_latent_action(latent_hist_state, active_head=None)
            next_state, reward, life_lost, terminal = env.step(action)
            next_latent_state, x_d = vqenv.get_state_representation(next_state[None])
            latents.append(next_latent_state.cpu().numpy())
            x_ds.append(x_d)
            latent_hist_state = torch.cat((latent_hist_state[:,1:], next_latent_state[:,None]), dim=1)
            evaluate_step_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            frames_for_gif.append(env.ale.getScreenRGB())
            results_for_eval.append("%s, %s, %s, %s" %(action, reward, life_lost, terminal))
            if not episode_steps%1000:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state

        # only save best if we've seen this round
        if not i:
            generate_gif(model_base_filedir, step_number, np.array(latents)[:,0], episode_reward_sum, name='test_latents', results=results_for_eval, resize=True)
            #rec_est, rec_mean = vqenv.sample_from_latents(torch.cat(x_ds))
            #rec = (255*rec_est[:,0]).astype(np.uint8)
            #generate_gif(model_base_filedir, step_number, rec, episode_reward_sum, name='test_reconstruct', results=results_for_eval, resize=False)
        #    generate_gif(model_base_filedir, step_number, frames_for_gif, episode_reward_sum, name='test', results=results_for_eval, resize=False)
        #eval_rewards.append(episode_reward_sum)
    #print("Evaluation score:\n", eval_rewards)
    #efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
    #with open(efile, 'a') as eval_reward_file:
    #    print(step_number, np.mean(eval_rewards), file=eval_reward_file)
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
        "N_PLAYOUT":50,
        "MIN_SCORE_GIF":-1, # min score to plot gif in eval
        "DEVICE":device, #cpu vs gpu set by argument
        #"NAME":'MBReward_RUN_rerunwithnewstatemanager', # start files with name
        #"NAME":'MBReward_RUN_rerunwithnewstatemanager_fullytrainedvqvae_lower_checkpoint', # start files with name
        #"NAME":'MBReward_embedding_hist_SEED14_GAMMAp99_prior1MLReps', # start files with name
        "NAME":"MBBreakout_replay",
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
        "MIN_STEPS_TO_LEARN":50000, # min steps needed to start training neural nets
        "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        # I think this randomness might need to be higher
        "EPS_INIT":0.01,
        "EPS_FINAL":0.01, # 0.01 in osband
        "EPS_EVAL":0.0, # 0 in osband, .05 in others....
        "NUM_EVAL_EPISODES":1, # num examples to average in eval
        #"BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "BUFFER_SIZE":int(500000), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":500000, # how often to write pkl of model and npz of data buffer
        #"CHECKPOINT_EVERY_STEPS":1e6, # how often to write pkl of model and npz of data buffer
        #"EVAL_FREQUENCY":500000, # how often to run evaluation episodes
        "EVAL_FREQUENCY":200000, # how often to run evaluation episodes
        #"EVAL_FREQUENCY":1, # how often to run evaluation episodes
        "ADAM_LEARNING_RATE":6.25e-5,
        #"ADAM_LEARNING_RATE":1e-4,
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
        "VQ_MODEL_LOADPATH":"../../model_savedir/MBBreakout_init_dataset/BreakoutVQ02/BreakoutVQ_0103019776ex.pt",
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

    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                      num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      reshape_size=info['RESHAPE_SIZE'],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                      num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])
    if info['PRIOR']:
        prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                reshape_size=info['RESHAPE_SIZE'],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING'],
                                num_clusters=vqenv.vq_info['NUM_K']).to(info['DEVICE'])

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

    train_sim(start_step_number, start_last_save)

