import os
import sys
import numpy as np
from IPython import embed

import math
from logger import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from experience_handler import experience_replay
from prepare_atari import DMAtariEnv
sys.path.append('../models')
from glob import glob
import config
from ae_utils import save_checkpoint
from dqn_model import EnsembleNet

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

def train_batch(batch, cnt):
    st = time.time()
    losses = [0.0 for k in range(info['N_ENSEMBLE'])]
    inputs_pt = torch.Tensor(batch[0]).to(info['DEVICE'])
    nexts_pt =  torch.Tensor(batch[1]).to(info['DEVICE'])
    #print('state',inputs_pt.sum(), nexts_pt.sum())
    actions_pt = torch.LongTensor(batch[2][:,0][:, None]).to(info['DEVICE'])
    rewards_pt = torch.Tensor(batch[2][:,1].astype(np.float32)).to(info['DEVICE'])

    ongoing_flags_pt = torch.Tensor(batch[2][:,2]).to(info['DEVICE'])
    mask_pt = torch.FloatTensor(batch[3]).to(info['DEVICE'])


    q_values = policy_net_ensemble(inputs_pt, None)
    next_q_values = policy_net_ensemble(nexts_pt, None)
    next_q_state_values = target_net_ensemble(nexts_pt, None)

    opt.zero_grad()
    cnt_losses = []
    for k in range(info['N_ENSEMBLE']):
        # TODO mask
        total_used = torch.sum(mask_pt[:, k])
        if total_used > 0.0:
            q_value = q_values[k].gather(1, actions_pt).squeeze(1)
            next_q_value = next_q_state_values[k].gather(1, next_q_values[k].max(1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = rewards_pt + (info["GAMMA"] * next_q_value * ongoing_flags_pt)
            #full_loss = (q_value-expected_q_value.detach()).pow(2).mean()
            full_loss = F.smooth_l1_loss(q_value,expected_q_value.detach())
            full_loss = mask_pt[:,k]*full_loss
            loss = torch.sum(full_loss/total_used)
            cnt_losses.append(loss)
            #loss.backward(retain_graph=True)
            losses[k] = loss.cpu().detach().item()
    all_loss = torch.stack(cnt_losses).sum()
    all_loss.backward()
    # trains faster if core_net is divided by 1/k vs the entire network
    for param in policy_net_ensemble.core_net.parameters():
        if param.grad is not None:
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    torch.nn.utils.clip_grad_value_(policy_net_ensemble.parameters(), info['CLIP_GRAD'])
    opt.step()
    if not cnt%info['TARGET_UPDATE']:
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print('updating target network')
        target_net_ensemble.load_state_dict(policy_net_ensemble.state_dict())
    et = time.time()
    board_logger.scalar_summary('Loss by step', cnt, np.mean(losses))
    board_logger.scalar_summary('Train batch time by step', cnt, et-st)
    return losses

def handle_checkpoint(last_save, cnt, epoch):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        print("checkpoint")
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'epoch':epoch,
                 'policy_net_ensemble_state_dict':policy_net_ensemble.state_dict(),
                 'target_net_ensemble_state_dict':target_net_ensemble.state_dict(),
                 }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
        save_checkpoint(state, filename)
        return last_save,filename
    else: return last_save, ''

def handle_step(cnt, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer,checkpoint=''):
    # mask to determine which head can use this experience
    exp_mask = random_state.binomial(1, info['BERNOULLI_P'], info['N_ENSEMBLE']).astype(np.uint8)
    # at this observed state
    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
    batch = replay_buffer.send((checkpoint, experience))
    # update so "state" representation is past history_size frames
    S_hist.pop(0)
    S_hist.append(S_prime)
    episodic_reward += reward
    cnt+=1
    return cnt, S_hist, batch, episodic_reward

def run_training_episode(epoch_num, total_steps, last_save):
    epsilon = 0.0
    start = time.time()
    episodic_losses = []
    start_steps = total_steps
    episode_steps = 0
    random_state.shuffle(heads)
    active_head = heads[0]
    episodic_reward = 0.0
    S, action, reward, finished = env.reset()
    episode_actions = [action]
    # init current state buffer with initial frame
    S_hist = [S for _ in range(info['HISTORY_SIZE'])]
    epoch_losses = [0. for k in range(info['N_ENSEMBLE'])]
    epoch_steps = [1. for k in range(info['N_ENSEMBLE'])]
    policy_net_ensemble.train()
    total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S, action, reward, finished, info['RANDOM_HEAD'], info['FAKE_ACTS'], 0, exp_replay)
    print("start action while loop")
    checkpoint_times = []

    while not finished:
        est = time.time()
        with torch.no_grad():
            # always do this calculation - as it is used for debugging
            S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(info['DEVICE'])
            # get all values for logging
            vals = policy_net_ensemble(S_hist_pt, None)
            acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
            action = acts[active_head]
            #embed()
            #vals = [q.cpu().data.numpy() for q in policy_net(S_hist_pt, None)]
            #acts = [np.argmax(v, axis=-1)[0] for v in vals]

        if info['USE_EPSILON']:
            epsilon = epsilon_by_frame(total_steps)
            if (random_state.rand() < epsilon):
                action = random_state.choice(action_space)
            k_used = info['RANDOM_HEAD']
        #else:
        #    action = acts[active_head]
            #k_used = active_head
        S_prime, reward, finished = env.step4(action)
        cst = time.time()
        last_save, checkpoint = handle_checkpoint(last_save, total_steps, epoch_num)
        #total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, exp_replay, checkpoint)
        total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S_prime, action, reward, finished, active_head, acts, episodic_reward, exp_replay, checkpoint)
        if batch:
            train_batch(batch, total_steps)
        eet = time.time()
        checkpoint_times.append(time.time()-cst)
        #if not total_steps % 100:
        #    # CPU 40 seconds to complete 1 action when buffer is 6000
        #    # CPU .0008 seconds to complete 1 action when buffer is 0
        #    print('time', eet-est)
        #    print(total_steps, 'head', active_head,'action', action, 'so far reward', episodic_reward)
        #    print('epsilon', epsilon)
    stop = time.time()
    ep_time =  stop - start
    board_logger.scalar_summary('Reward by step', total_steps, episodic_reward)
    board_logger.scalar_summary('Reward per episode', epoch_num, episodic_reward)
    board_logger.scalar_summary('head per episode', epoch_num, active_head)
    board_logger.scalar_summary('Avg get batch time', epoch_num, np.mean(checkpoint_times))
    board_logger.scalar_summary('epoch time', epoch_num, ep_time)
    board_logger.scalar_summary('epoch steps', epoch_num, total_steps-start_steps)
    print("EPISODE:%s HEAD %s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, active_head, episodic_reward, total_steps-start_steps, total_steps))
    #print('actions',episode_actions)
    print("time for episode", ep_time)
    return active_head, episodic_reward, total_steps, ep_time, last_save, np.mean(episodic_losses)

def write_info_file(cnt):
    info_filename = model_base_filepath + "_%010d_info.txt"%cnt
    info_f = open(info_filename, 'w')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz model file full path')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    # takes about 24 steps after "fire" for game to end
    # 3 is right
    # 2 is left
    # 1 is fire
    # 0 is noop
    info = {
        'USE_EPSILON':False,
        "GAME":'Breakout', # gym prefix
        "DEVICE":device,
        "NAME":'_Breakout7RMS', # start files with name
        "N_ENSEMBLE":7, # number of heads to use
        "BERNOULLI_P": 0.9, # Probability of experience to go to each head
        "TARGET_UPDATE":10000, # TARGET_UPDATE how often to use replica target TODO - what should this be
        "CHECKPOINT_EVERY_STEPS":100000,
        "RMS_LEARNING_RATE": .00025,
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "CLIP_REWARD_MAX":1,
        "CLIP_REWARD_MAX":-1,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "PRINT_EVERY":1, # How often to print statistics
        "PRIOR_SCALE":0.0, # Weight for randomized prior, 0. disables
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
        "EPSILON_MAX":1.0, # Epsilon greedy exploration ~prob of random action, 0. disables
        "EPSILON_MIN":.01,
        "EPSILON_DECAY":30000,
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":18,
        "RANDOM_HEAD":-1,
        "NETWORK_INPUT_SIZE":(84,84),
        }

    if not info['USE_EPSILON'] and info['N_ENSEMBLE'] == 1:
        print('no epsilon and single epsilon')
        embed()
    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    accumulation_rewards = []
    overall_time = 0.
    info['load_time'] = datetime.date.today().ctime()

    if args.model_loadpath != '':
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        total_steps = model_dict['cnt']
        info['DEVICE'] = device
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        last_save = model_dict['cnt']
        info['loaded_from'] = args.model_loadpath
        epoch_start = model_dict['epoch']
    else:
        total_steps = 0
        last_save = 0
        epoch_start = 0
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(total_steps)
    env = DMAtariEnv(info['GAME'],random_seed=info['SEED'])
    action_space = np.arange(env.env.action_space.n)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])
    #policy_net = DDQNCoreNet(info['HISTORY_SIZE'], env.env.action_space.n).to(info['DEVICE'])
    #target_net = DDQNCoreNet(info['HISTORY_SIZE'], env.env.action_space.n).to(info['DEVICE'])
    policy_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])
    target_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE']).to(info['DEVICE'])


    #opt = optim.Adam(policy_net_ensemble.parameters(), lr=info['ADAM_LEARNING_RATE'])
    opt = optim.RMSprop(policy_net_ensemble.parameters(),
                        lr=info["RMS_LEARNING_RATE"],
                        momentum=info["RMS_MOMENTUM"],
                        eps=info["RMS_EPSILON"],
                        centered=info["RMS_CENTERED"],
                        alpha=info["RMS_DECAY"])

    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net_ensemble.load_state_dict(model_dict['target_net_ensemble_state_dict'])
        policy_net_ensemble.load_state_dict(model_dict['policy_net_ensemble_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            args.buffer_loadpath = glob(args.model_loadpath.replace('.pkl', '*.npz'))[0]
            print("auto loading buffer from:%s" %args.buffer_loadpath)
    exp_replay = experience_replay(batch_size=info['BATCH_SIZE'],
                                   max_size=info['BUFFER_SIZE'],
                                   history_size=info['HISTORY_SIZE'],
                                   name='train_buffer', random_seed=info['SEED'],
                                   buffer_file=args.buffer_loadpath)

    random_state = np.random.RandomState(info["SEED"])
    next(exp_replay) # Start experience-replay coroutines

    board_logger = TensorBoardLogger(model_base_filedir)
    last_target_update = 0
    print("Starting training")
    all_rewards = []

    epsilon_by_frame = lambda frame_idx: info['EPSILON_MIN'] + (info['EPSILON_MAX'] - info['EPSILON_MIN']) * math.exp( -1. * frame_idx / info['EPSILON_DECAY'])
    total_head_reward = [0.0 for x in range(info['N_ENSEMBLE'])]
    head_count = [0 for x in range(info['N_ENSEMBLE'])]
    for epoch_num in range(epoch_start, info['N_EPOCHS']):
        active_head, ep_reward, total_steps, etime, last_save, mean_loss = run_training_episode(epoch_num, total_steps, last_save)
        head_count[active_head] +=1
        total_head_reward[active_head]+=ep_reward
        avg_by_head = [total_head_reward[k]/(head_count[k]+.01) for k in range(info['N_ENSEMBLE'])]
        all_rewards.append(ep_reward)
        overall_time += etime
        board_logger.scalar_summary("avg reward last 100 episodes", epoch_num, np.mean(all_rewards[-100:]))
        print("HEAD AVERAGE")
        print(avg_by_head)

#        if (total_steps - last_target_update) >= info['TARGET_UPDATE']:
#            print("Updating target network at {}".format(epoch_num))
#            target_net.load_state_dict(policy_net.state_dict())
#            last_target_update = total_steps
