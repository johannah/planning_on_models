# extending on code from
# https://github.com/58402140/Fruit

import matplotlib
matplotlib.use('Agg')

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

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

def save_img(epoch):
    imagedir = 'images_%010d'%epoch
    print("saving images to %s"%imagedir)
    if imagedir not in os.listdir('.'):
        os.mkdir(imagedir)
    frame = 0
    while True:
        if frame < IMAGE_SAVE_LIMIT:
            screen, reward = (yield)
            plt.imshow(screen, interpolation='none')
            plt.title("reward: {}".format(reward))
            plt.savefig(os.path.join(imagedir, '%010d.png'%(frame)))
            frame += 1

def plot_evals(accumulation_rewards):
    plt.figure()
    trace = np.array(accumulation_rewards)
    #xs = np.array([int(n * EVALUATE_EVERY) for n in range(N_EPOCHS // EVALUATE_EVERY + 1)])
    xs = np.array([int(n * EVALUATE_EVERY) for n in range(trace.shape[0])])
    plt.plot(xs, trace, label="Reward")
    plt.legend()
    plt.ylabel("Average Evaluation Reward ({})".format(N_EVALUATIONS))
    model = "Double DQN" if USE_DOUBLE_DQN else "DQN"
    if N_ENSEMBLE > 1:
        model = "Bootstrap " + model
    if PRIOR_SCALE > 0.:
        model = model + " with randomized prior {}".format(PRIOR_SCALE)
    footnote_text = "Episodes\n"
    footnote_text += "\n"
    footnote_text += "\n"
    footnote_text += "Settings:\n"
    footnote_text += "{}\n".format(model)
    footnote_text += "Number of heads {}\n".format(N_ENSEMBLE)
    footnote_text += "Epsilon-greedy {}\n".format(EPSILON)
    if N_ENSEMBLE > 1:
        footnote_text += "Sharing mask probability {}\n".format(BERNOULLI_P)
    footnote_text += "Gamma decay {}\n".format(GAMMA)
    footnote_text += "Grad clip {}\n".format(CLIP_GRAD)
    footnote_text += "Adam, learning rate {}\n".format(ADAM_LEARNING_RATE)
    footnote_text += "Batch size {}\n".format(BATCH_SIZE)
    footnote_text += "Experience replay buffer size {}\n".format(BUFFER_SIZE)
    footnote_text += "Training time {}\n".format(overall_time)
    plt.xlabel(footnote_text)
    plt.tight_layout()
    plt.savefig("reward_traces.png")


def train_batch(batch, epoch_losses, epoch_steps):
    inputs = []
    actions = []
    rewards = []
    nexts = []
    ongoing_flags = []
    masks = []
    st = time.time()
    for b_i in batch:
        s, s_prime, a, r, ongoing_flag, mask = b_i
        rewards.append(r)
        inputs.append(s)
        actions.append(a)
        nexts.append(s_prime)
        ongoing_flags.append(ongoing_flag)
        masks.append(mask)
    mask = torch.Tensor(np.array(masks)).to(DEVICE)
    # precalculate the core Q values for every head
    inputs_pt = torch.Tensor(inputs).to(DEVICE)
    nexts_pt = torch.Tensor(nexts).to(DEVICE)
    all_target_next_Qs = [n.detach() for n in target_net(nexts_pt, None)]
    all_Qs = policy_net(inputs_pt, None)
    if USE_DOUBLE_DQN:
        all_policy_next_Qs = [n.detach() for n in policy_net(nexts_pt, None)]
    # set grads to 0 before iterating heads
    opt.zero_grad()
    for k in range(N_ENSEMBLE):
        total_used = torch.sum(mask[:, k])
        if total_used:
            if USE_DOUBLE_DQN:
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
            next_max_Qs = torch.Tensor(ongoing_flags).to(DEVICE) * next_max_Qs
            target_Qs = torch.Tensor(np.array(rewards).astype("float32")).to(DEVICE) + GAMMA * next_max_Qs

            # get current step predictions
            Qs = all_Qs[k]
            Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32")).to(DEVICE))
            Qs = Qs.squeeze()

            # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
            # need to do updates on each head based on experience mask
            full_loss = (Qs - target_Qs) ** 2
            full_loss = mask[:, k] * full_loss
            #loss = torch.mean(full_loss)

            loss = torch.sum(full_loss / total_used)
            loss.backward(retain_graph=True)
            loss_np = loss.cpu().detach().numpy()
            if np.isinf(loss_np) or np.isnan(loss_np):
                print('nan')
                embed()
            for param in policy_net.parameters():
                if param.grad is not None:
                    # Multiply grads by 1 / K
                    param.grad.data *= 1. / N_ENSEMBLE
            epoch_losses[k] += loss.detach().cpu().numpy()
            epoch_steps[k] += 1.
    # After iterating all heads, do the update step
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), CLIP_GRAD)
    opt.step()
    et = time.time()
    return epoch_losses, epoch_steps

def run_training_episode(epoch_num):
    start = time.time()
    random_state.shuffle(heads)
    active_head = heads[0]
    episodic_reward = 0
    S, action, reward, finished = env.reset()
    ongoing_flag = int(finished)
    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
    experience =  [S, action, reward, ongoing_flag, exp_mask, -1, fake_acts]
    batch = exp_replay.send(experience)
    S_hist = [S for _ in range(HISTORY_SIZE)]
    epoch_losses = [0. for k in range(N_ENSEMBLE)]
    epoch_steps = [1. for k in range(N_ENSEMBLE)]
    policy_net.train()
    episode_actions = []
    cnt = 0
    while not finished:
        with torch.no_grad():
            # always do this calculation - as it is used for debugging
            S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(DEVICE)
            vals = [q.cpu().data.numpy() for q in policy_net(S_hist_pt, None)]
            acts = [np.argmax(v, axis=-1)[0] for v in vals]

        if (random_state.rand() < EPSILON):
            action = random_state.choice(action_space)
            used = -1
        else:
            action = acts[active_head]
            used = active_head
        episode_actions.append(action)
        S_prime, reward, finished = env.step4(action)
        if finished:
            print("got finished epoch")
        episodic_reward+=reward
        ongoing_flag = int(finished)
        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
        experience =  [S_prime, action, reward, ongoing_flag, exp_mask, used, acts]
        S_hist.pop(0)
        S_hist.append(S_prime)
        S = S_prime
        batch = exp_replay.send(experience)
        cnt+=1
        if batch:
            epoch_losses, epoch_steps = train_batch(batch, epoch_losses, epoch_steps)
    stop = time.time()
    ep_time =  stop - start
    print("EPISODIC_REWARD", epoch_num, episodic_reward)
    print("episode steps", cnt)
    print('actions',episode_actions)
    print("Epoch {}, head {}, loss: {}".format(epoch_num + 1, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))
    return cnt, ep_time

def run_eval_episode(eval_epoch):
    eval_cnt = 0
    evaluate_episodic_reward = 0
    S, eval_action, reward, finished = eval_env.reset()
    print("starting new eval epoch", eval_epoch, finished)
    ongoing_flag = int(finished)
    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
    experience =  [S, eval_action, reward, ongoing_flag, exp_mask, -1, fake_acts]
    _ = eval_exp_replay.send(experience)

    S_hist = [S for _ in range(HISTORY_SIZE)]
    reward_trace = [reward]
    policy_net.eval()
    eval_actions = []
    while not finished:
        S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(DEVICE)
        vals = [q.cpu().data.numpy() for q in policy_net(S_hist_pt, None)]
        acts = [np.argmax(v, axis=-1)[0] for v in vals]
        #print('vals',vals,acts)
        act_counts = Counter(acts)
        max_count = max(act_counts.values())
        top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count]
        # break action ties with random choice
        random_state.shuffle(top_actions)
        eval_action = top_actions[0]
        eval_actions.append(eval_action)
        S_prime, reward, finished = eval_env.step4(eval_action)
        ongoing_flag = int(finished)
        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
        top_heads = [k for (k,a) in enumerate(acts) if a in top_actions]
        # randomly choose which head to say was used in case there are
        # multiple heads that chose same action
        k_used = top_heads[random_state.randint(len(top_heads))]
        experience =  [S, eval_action, reward, ongoing_flag, exp_mask, k_used, acts]
        if not eval_cnt%10:
            print(eval_cnt,'chose',eval_action,'k',k_used)
            print('eval actions', acts)
            print(np.sum(vals), np.sum(S_hist))
        eval_cnt+=1
        _ = eval_exp_replay.send(experience)
        evaluate_episodic_reward += reward
        reward_trace.append(reward)
        if SAVE_IMAGES:
            img_saver.send((S_prime, reward))
        S_hist.pop(0)
        S_hist.append(S_prime)
        S = S_prime

    print("Evaluation Episode eval_epoch {} reward {}".format(eval_epoch, evaluate_episodic_reward))
    print('num steps in epoch', len(eval_actions), 'ep reward', evaluate_episodic_reward)
    print('eval actions', eval_actions)
    return eval_cnt, reward_trace

if __name__ == '__main__':
    # 3 is right
    # 2 is left
    # 1 is fire
    # 0 is noop
    NETWORK_INPUT_SIZE = (84,84)
    CLIP_REWARD_MAX = 1
    CLIP_REWARD_MAX = -1
    DEVICE = 'cuda'
    NAME = '_debug1H'
    # how many past frames to use for state input
    HISTORY_SIZE = 4
    # How often to check and evaluate
    EVALUATE_EVERY = 10
    # Number of evaluation episodes to run
    N_EVALUATIONS = 3
    # Save images of policy at each evaluation if True, otherwise only at the end if False
    SAVE_IMAGES = False
    # How often to print statistics
    PRINT_EVERY = 1
    # Whether to use double DQN or regular DQN
    USE_DOUBLE_DQN = True
    # TARGET_UPDATE how often to use replica target
    TARGET_UPDATE = 30
    # Number of heads for ensemble (1 falls back to DQN)
    N_ENSEMBLE = 1
    # Probability of experience to go to each head
    BERNOULLI_P = 1.0
    # Weight for randomized prior, 0. disables
    PRIOR_SCALE = 0.0
    # Number of episodes to run
    N_EPOCHS = 10000
    # Batch size to use for learning
    BATCH_SIZE = 128
    # Buffer size for experience replay
    BUFFER_SIZE = 1e6
    # How often to write data file
    WRITE_BUFFER_EVERY = 10000
    # Epsilon greedy exploration ~prob of random action, 0. disables
    EPSILON = 0.1
    # Gamma weight in Q update
    GAMMA = .99
    # Gradient clipping setting
    CLIP_GRAD = 1
    # Learning rate for Adam
    SEED = 18
    seed_everything(SEED)
    ADAM_LEARNING_RATE = 1E-4
    IMAGE_SAVE_LIMIT = 300
    random_state = np.random.RandomState(11)

    # change minibatch setup to use masking...
    replay_size = int(N_ENSEMBLE * BUFFER_SIZE)
    eval_replay_size = 2000#int(replay_size*.1)
    eval_write_every = 2000
    exp_replay = experience_replay(batch_size=BATCH_SIZE, max_size=replay_size,
                                   history_size=HISTORY_SIZE,
                                   write_buffer_every=WRITE_BUFFER_EVERY,
                                   name=NAME+'train')
    eval_exp_replay = experience_replay(batch_size=BATCH_SIZE, max_size=eval_replay_size,
                                   history_size=HISTORY_SIZE,
                                   write_buffer_every=eval_write_every, name=NAME+'eval')
    next(exp_replay) # Start experience-replay coroutines
    next(eval_exp_replay)

    # Stores the total rewards from each evaluation, per head over all epochs
    accumulation_rewards = []
    overall_time = 0.
    env = DMAtariEnv('Breakout',random_seed=34)
    eval_env = DMAtariEnv('Breakout',random_seed=55)
    action_space = np.arange(env.env.action_space.n)
    heads = list(range(N_ENSEMBLE))
    total_steps = 0
    eval_total_steps = 0
    eval_cnt = 0
    prior_net_ensemble = EnsembleNet(n_ensemble=N_ENSEMBLE, n_actions=env.env.action_space.n,
                                     network_output_size=NETWORK_INPUT_SIZE[0], num_channels=HISTORY_SIZE)
    policy_net_ensemble = EnsembleNet(n_ensemble=N_ENSEMBLE, n_actions=env.env.action_space.n,
                                      network_output_size=NETWORK_INPUT_SIZE[0], num_channels=HISTORY_SIZE)

    policy_net = NetWithPrior(policy_net_ensemble, prior_net_ensemble, PRIOR_SCALE).to(DEVICE)

    target_net_ensemble = EnsembleNet(n_ensemble=N_ENSEMBLE, n_actions=env.env.action_space.n,
                                      network_output_size=NETWORK_INPUT_SIZE[0], num_channels=HISTORY_SIZE)
    target_net = NetWithPrior(target_net_ensemble, prior_net_ensemble, PRIOR_SCALE).to(DEVICE)

    opt = optim.Adam(policy_net.parameters(), lr=ADAM_LEARNING_RATE)
    fake_acts = [-1 for x in range(N_ENSEMBLE)]


    for epoch_num in range(N_EPOCHS):
        tcnt, ttime = run_training_episode(epoch_num)
        total_steps += tcnt
        overall_time += ttime

        if TARGET_UPDATE > 1 and epoch_num % TARGET_UPDATE == 0:
            print("Updating target network at {}".format(epoch_num))
            target_net.load_state_dict(policy_net.state_dict())


        if (epoch_num % EVALUATE_EVERY == 0 or epoch_num == (N_EPOCHS - 1)) and epoch_num > 0:
            evaluation_rewards = []
            for eval_epoch in range(N_EVALUATIONS):
                ecnt, eval_reward_trace = run_eval_episode(eval_epoch)
                eval_total_steps += ecnt
                evaluation_rewards.append(np.sum(eval_reward_trace))
            accumulation_rewards.append(np.mean(evaluation_rewards))
            print("Mean evaluation reward all eval epochs {}".format(accumulation_rewards[-1]))
            plot_evals(accumulation_rewards)

