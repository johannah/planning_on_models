# extending on code from
# https://github.com/58402140/Fruit

import matplotlib
matplotlib.use('Agg')
import sys
# TODO - fix install
sys.path.append('../models')
import config
from ae_utils import save_checkpoint
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prepare_atari import DMAtariEnv
from dqn_model import EnsembleNet, NetWithPrior
from experience_handler import experience_replay
from IPython import embed
from imageio import imwrite
from glob import glob

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

def plot_evals(accumulation_rewards):
    plt.figure()
    trace = np.array(accumulation_rewards)
    #xs = np.array([int(n * EVALUATE_EVERY) for n in range(N_EPOCHS // EVALUATE_EVERY + 1)])
    xs = np.array([int(n * info['EVALUATE_EVERY']) for n in range(trace.shape[0])])
    plt.plot(xs, trace, label="Reward")
    plt.legend()
    plt.ylabel("Average Evaluation Reward ({})".format(info['N_EVALUATIONS']))
    model = "Double DQN" if info['USE_DOUBLE_DQN'] else "DQN"
    if info['N_ENSEMBLE'] > 1:
        model = "Bootstrap " + model
    if info['PRIOR_SCALE'] > 0.:
        model = model + " with randomized prior {}".format(info['PRIOR_SCALE'])
    footnote_text = "Episodes\n"
    footnote_text += "\n"
    footnote_text += "\n"
    footnote_text += "Settings:\n"
    footnote_text += "{}\n".format(model)
    footnote_text += "Number of heads {}\n".format(info['N_ENSEMBLE'])
    footnote_text += "Epsilon-greedy {}\n".format(info['EPSILON'])
    if info['N_ENSEMBLE'] > 1:
        footnote_text += "Sharing mask probability {}\n".format(info['BERNOULLI_P'])
    footnote_text += "Gamma decay {}\n".format(info['GAMMA'])
    footnote_text += "Grad clip {}\n".format(info['CLIP_GRAD'])
    footnote_text += "Adam, learning rate {}\n".format(info['ADAM_LEARNING_RATE'])
    footnote_text += "Batch size {}\n".format(info['BATCH_SIZE'])
    footnote_text += "Experience replay buffer size {}\n".format(info['BUFFER_SIZE'])
    footnote_text += "Training time {}\n".format(overall_time)
    plt.xlabel(footnote_text)
    plt.tight_layout()
    plt.savefig("reward_traces.png")


def train_batch(batch, epoch_losses, epoch_steps):
    st = time.time()
    inputs_pt = torch.Tensor(batch[0]).to(info['DEVICE'])
    nexts_pt =  torch.Tensor(batch[1]).to(info['DEVICE'])
    ongoing_flags_pt = torch.Tensor(batch[2][:,2]).to(info['DEVICE'])
    mask_pt = torch.FloatTensor(batch[3]).to(info['DEVICE'])
    actions_pt = torch.LongTensor(batch[2][:,0][:, None]).to(info['DEVICE'])
    rewards_pt = torch.Tensor(batch[2][:,1].astype(np.float32)).to(info['DEVICE'])
    all_target_next_Qs = [n.detach() for n in target_net(nexts_pt, None)]
    all_Qs = policy_net(inputs_pt, None)
    if info['USE_DOUBLE_DQN']:
        all_policy_next_Qs = [n.detach() for n in policy_net(nexts_pt, None)]
    # set grads to 0 before iterating heads
    opt.zero_grad()
    for k in range(info['N_ENSEMBLE']):
        total_used = torch.sum(mask_pt[:, k])
        if total_used:
            if info['USE_DOUBLE_DQN']:
                policy_next_Qs = all_policy_next_Qs[k]
                next_Qs = all_target_next_Qs[k]
                policy_actions = policy_next_Qs.max(1)[1][:, None]
                next_max_Qs = next_Qs.gather(1, policy_actions)
                next_max_Qs = next_max_Qs.squeeze()
            else:
                next_Qs = all_target_next_Qs[k]
                next_max_Qs = next_Qs.max(1)[0]
                next_max_Qs = next_max_Qs.squeeze()

            # mask based on if it is end of episode or not
            next_max_Qs = ongoing_flags_pt * next_max_Qs
            target_Qs = rewards_pt + info['GAMMA'] * next_max_Qs

            # get current step predictions
            Qs = all_Qs[k]
            Qs = Qs.gather(1, actions_pt)
            Qs = Qs.squeeze()

            # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
            # need to do updates on each head based on experience mask
            full_loss = (Qs - target_Qs) ** 2
            full_loss = mask_pt[:, k] * full_loss
            #loss = torch.mean(full_loss)

            loss = torch.sum(full_loss / total_used)
            loss.backward(retain_graph=True)
            loss_np = loss.cpu().detach().numpy()
            if np.isinf(loss_np) or np.isnan(loss_np):
                print('nan')
                embed()
            for param in policy_net.parameters():
                if param.grad is not None:
                    # Multiply grads by 1 / K?
                    param.grad.data *= 1. / info['N_ENSEMBLE']
            epoch_losses[k] += loss.detach().cpu().numpy()
            epoch_steps[k] += 1.
    # After iterating all heads, do the update step
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    et = time.time()
    return epoch_losses, epoch_steps

def handle_checkpoint(last_save, cnt, epoch):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        print("checkpoint")
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'epoch':epoch,
                 'policy_net_state_dict':policy_net.state_dict(),
                 'target_net_state_dict':target_net.state_dict(),
                 }
        filename = model_base_filepath + "_%010dq.pkl"%cnt
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
    start_steps = total_steps
    episode_steps = 0
    start = time.time()
    random_state.shuffle(heads)
    active_head = heads[0]
    episodic_reward = 0.0
    S, action, reward, finished = env.reset()
    # init current state buffer with initial frame
    S_hist = [S for _ in range(info['HISTORY_SIZE'])]
    epoch_losses = [0. for k in range(info['N_ENSEMBLE'])]
    epoch_steps = [1. for k in range(info['N_ENSEMBLE'])]
    policy_net.train()
    total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S, action, reward, finished, info['RANDOM_HEAD'], info['FAKE_ACTS'], 0, exp_replay)
    episode_actions = []
    while not finished:
        with torch.no_grad():
            # always do this calculation - as it is used for debugging
            S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(info['DEVICE'])
            vals = [q.cpu().data.numpy() for q in policy_net(S_hist_pt, None)]
            acts = [np.argmax(v, axis=-1)[0] for v in vals]

        if (random_state.rand() < info['EPSILON']):
            action = random_state.choice(action_space)
            k_used = info['RANDOM_HEAD']
        else:
            action = acts[active_head]
            k_used = active_head
        S_prime, reward, finished = env.step4(action)
        last_save, checkpoint = handle_checkpoint(last_save, total_steps, epoch_num)
        total_steps, S_hist, batch, episodic_reward = handle_step(total_steps, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, exp_replay)
        episode_actions.append(action)
        if batch:
            epoch_losses, epoch_steps = train_batch(batch, epoch_losses, epoch_steps)
        if not total_steps % 100:
            print(total_steps, 'head', active_head,'action', action, 'so far reward', episodic_reward)

    stop = time.time()
    ep_time =  stop - start
    print("EPISODE:%s HEAD %s REWARD:%s ------ ep %04d total %010d steps"%(epoch_num, active_head, episodic_reward, total_steps-start_steps, total_steps))
    print("loss: {}".format([epoch_losses[k] / float(epoch_steps[k]) for k in range(info['N_ENSEMBLE'])]))
    print('actions',episode_actions)
    return total_steps, ep_time, last_save


#def run_eval_episode(eval_epoch):
#    eval_cnt = 0
#    evaluate_episodic_reward = 0
#    S, eval_action, reward, finished = eval_env.reset()
#    print("starting new eval epoch", eval_epoch, finished)
#    ongoing_flag = int(finished)
#    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
#    experience =  [S, eval_action, reward, ongoing_flag, exp_mask, RANDOM_HEAD, info['FAKE_ACTS']]
#    _ = eval_exp_replay.send(experience)
#
#    S_hist = [S for _ in range(HISTORY_SIZE)]
#    reward_trace = [reward]
#    policy_net.eval()
#    eval_actions = []
#    while not finished:
#        S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(DEVICE)
#        vals = [q.cpu().data.numpy() for q in policy_net(S_hist_pt, None)]
#        acts = [np.argmax(v, axis=-1)[0] for v in vals]
#        #print('vals',vals,acts)
#        act_counts = Counter(acts)
#        max_count = max(act_counts.values())
#        top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count]
#        # break action ties with random choice
#        random_state.shuffle(top_actions)
#        eval_action = top_actions[0]
#        eval_actions.append(eval_action)
#        S_prime, reward, finished = eval_env.step4(eval_action)
#        ongoing_flag = int(finished)
#        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
#        top_heads = [k for (k,a) in enumerate(acts) if a in top_actions]
#        # randomly choose which head to say was used in case there are
#        # multiple heads that chose same action
#        k_used = top_heads[random_state.randint(len(top_heads))]
#        experience =  [S, eval_action, reward, ongoing_flag, exp_mask, k_used, acts]
#        if not eval_cnt%10:
#            print(eval_cnt,'chose',eval_action,'k',k_used)
#            print('eval actions', acts)
#            print(np.sum(vals), np.sum(S_hist))
#        eval_cnt+=1
#        _ = eval_exp_replay.send(experience)
#        evaluate_episodic_reward += reward
#        reward_trace.append(reward)
#        if SAVE_IMAGES:
#            img_saver.send((S_prime, reward))
#        S_hist.pop(0)
#        S_hist.append(S_prime)
#        S = S_prime
#
#    print("Evaluation Episode eval_epoch {} reward {}".format(eval_epoch, evaluate_episodic_reward))
#    print('num steps in epoch', len(eval_actions), 'ep reward', evaluate_episodic_reward)
#    print('eval actions', eval_actions)
#    return eval_cnt, reward_trace
#

def write_info_file(model_loaded=''):
    info_f = open(os.path.join(model_base_filedir, 'info%s.txt'%model_loaded), 'w')
    info_f.write(datetime.date.today().ctime()+'\n')
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
        "GAME":'Breakout', # gym prefix
        "DEVICE":device,
        "NAME":'_debug', # start files with name
        "N_ENSEMBLE":11, # number of heads to use
        "N_EVALUATIONS":3, # Number of evaluation episodes to run
        "BERNOULLI_P": 0.8, # Probability of experience to go to each head
        "TARGET_UPDATE":30, # TARGET_UPDATE how often to use replica target
        "USE_DOUBLE_DQN":True, # Whether to use double DQN or regular DQN
        "CHECKPOINT_EVERY_STEPS":1000,
        "ADAM_LEARNING_RATE": 0.00001,  #LR from this thread - https://github.com/dennybritz/reinforcement-learning/issues/30
        "CLIP_REWARD_MAX":1,
        "CLIP_REWARD_MAX":-1,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "EVALUATE_EVERY":200, # How often to check and evaluate
        "PRINT_EVERY":1, # How often to print statistics
        "PRIOR_SCALE":0.0, # Weight for randomized prior, 0. disables
        "N_EPOCHS":10000,  # Number of episodes to run
        "BATCH_SIZE":128, # Batch size to use for learning
        "BUFFER_SIZE":1e6, # Buffer size for experience replay
        "EPSILON":0.05, # Epsilon greedy exploration ~prob of random action, 0. disables
        "GAMMA":.99, # Gamma weight in Q update
        "CLIP_GRAD":1, # Gradient clipping setting
        "SEED":18, # Learning rate for Adam
        "RANDOM_HEAD":-1,
        "NETWORK_INPUT_SIZE":(84,84),
        }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    #eval_replay_size = 10000#int(replay_size*.1)
    #eval_write_every = 2000
    #eval_exp_replay = experience_replay(batch_size=BATCH_SIZE, max_size=eval_replay_size,
    #                               history_size=HISTORY_SIZE,
    #                               write_buffer_every=eval_write_every, name=NAME)
    #next(eval_exp_replay)
    #eval_env = DMAtariEnv(info['GAME',random_seed=info['SEED']+1)

    # Stores the total rewards from each evaluation, per head over all epochs

    accumulation_rewards = []
    overall_time = 0.

    if args.model_loadpath:
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(os.path.split(args.model_loadpath)[0])[1]
        model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
        last_save = model_dict['cnt']
    else:
        last_save = 0
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_loadpath, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        model_base_filepath = os.path.join(model_base_filedir, info['NAME'])


    env = DMAtariEnv(info['GAME'],random_seed=info['SEED'])
    action_space = np.arange(env.env.action_space.n)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])
    prior_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                     n_actions=env.env.action_space.n,
                                     network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                     num_channels=info['HISTORY_SIZE'])
    policy_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'])

    policy_net = NetWithPrior(policy_net_ensemble, prior_net_ensemble, info['PRIOR_SCALE']).to(info['DEVICE'])

    target_net_ensemble = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.env.action_space.n,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'])
    target_net = NetWithPrior(target_net_ensemble, prior_net_ensemble, info['PRIOR_SCALE']).to(info['DEVICE'])

    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])

    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        total_steps = model_dict['cnt']
        # set random seed based on how many it has seen
        if args.buffer_loadpath == '':
            args.buffer_loadpath = glob(args.model_loadpath.replace('.pkl', '*.npz'))[0]
            print("auto loading buffer from:%s" %args.buffer_loadpath)
        try:
            epoch_start = info['epoch']
        except:
            epoch_start = 67

    else:
        epoch_start = 0
        total_steps = model_dict['cnt']
    exp_replay = experience_replay(batch_size=info['BATCH_SIZE'],
                                   max_size=info['BUFFER_SIZE'],
                                   history_size=info['HISTORY_SIZE'],
                                   name='train_buffer', random_seed=info['SEED'],
                                   buffer_file=args.buffer_loadpath)

    random_state = np.random.RandomState(info["SEED"])
    next(exp_replay) # Start experience-replay coroutines

    for epoch_num in range(epoch_start, info['N_EPOCHS']):
        total_steps, etime, last_save = run_training_episode(epoch_num, total_steps, last_save)
        overall_time += etime
        print("TOTAL STEPS", total_steps)

        if info['TARGET_UPDATE'] > 1 and epoch_num % info['TARGET_UPDATE'] == 0:
            print("Updating target network at {}".format(epoch_num))
            target_net.load_state_dict(policy_net.state_dict())


        #if (epoch_num % EVALUATE_EVERY == 0 or epoch_num == (N_EPOCHS - 1)) and epoch_num > 0:
        #    evaluation_rewards = []
        #    for eval_epoch in range(N_EVALUATIONS):
        #        ecnt, eval_reward_trace = run_eval_episode(eval_epoch)
        #        eval_total_steps += ecnt
        #        evaluation_rewards.append(np.sum(eval_reward_trace))
        #    accumulation_rewards.append(np.mean(evaluation_rewards))
        #    print("Mean evaluation reward all eval epochs {}".format(accumulation_rewards[-1]))
        #    plot_evals(accumulation_rewards)


