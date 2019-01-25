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
from prepare_atari import NETWORK_INPUT_SIZE, DMAtariEnv
from experience_handler import experience_replay
from IPython import embed
from imageio import imwrite
DEVICE = 'cuda'
# how many past frames to use for state input
HISTORY_SIZE = 4
# How often to check and evaluate
EVALUATE_EVERY = 100
# Save images of policy at each evaluation if True, otherwise only at the end if False
SAVE_IMAGES = False
# How often to print statistics
PRINT_EVERY = 1

# Whether to use double DQN or regular DQN
USE_DOUBLE_DQN = True
# TARGET_UPDATE how often to use replica target
TARGET_UPDATE = 30
# Number of evaluation episodes to run
N_EVALUATIONS = 10
# Number of heads for ensemble (1 falls back to DQN)
N_ENSEMBLE = 9
# Probability of experience to go to each head
BERNOULLI_P = 0.8
# Weight for randomized prior, 0. disables
PRIOR_SCALE = 1.
# Number of episodes to run
N_EPOCHS = 10000
# Batch size to use for learning
BATCH_SIZE = 128
# Buffer size for experience replay
BUFFER_SIZE = 1e5
# How often to write data file
WRITE_BUFFER_EVERY = 1000
# Epsilon greedy exploration ~prob of random action, 0. disables
EPSILON = .1
# Gamma weight in Q update
GAMMA = .99
# Gradient clipping setting
CLIP_GRAD = 1
# Learning rate for Adam
ADAM_LEARNING_RATE = 1E-3
IMAGE_SAVE_LIMIT = 300
random_state = np.random.RandomState(11)

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(44)

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

#The first convolution layer convolves the input with 32 filters of size 8 (stride 4),
#the second layer has 64 layers of size 4
#(stride 2), the final convolution layer has 64 filters of size 3 (stride
#1). This is followed by a fully-connected hidden layer of 512 units.
class CoreNet(nn.Module):
    def __init__(self, chans=4, network_output_size=NETWORK_INPUT_SIZE[0]):
        super(CoreNet, self).__init__()
        self.network_output_size = network_output_size
        self.chans = chans
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(self.chans, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #embed
        # size after conv3
        reshape = 64*7*7
        x = x.view(-1, reshape)
        return x

class HeadNet(nn.Module):
    def __init__(self, n_actions=4):
        super(HeadNet, self).__init__()
        mult = 64*7*7
        self.fc1 = nn.Linear(mult, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet()
        self.net_list = nn.ModuleList([HeadNet() for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        return self.net_list[k](self.core_net(x))


class NetWithPrior(nn.Module):
    def __init__(self, net, prior, prior_scale=1.):
        super(NetWithPrior, self).__init__()
        self.net = net
        self.prior_scale = prior_scale
        if self.prior_scale > 0.:
            self.prior = prior

    def forward(self, x, k):
        if hasattr(self.net, "net_list"):
            if k is not None:
                if self.prior_scale > 0.:
                    return self.net(x, k) + self.prior_scale * self.prior(x, k).detach()
                else:
                    return self.net(x, k)
            else:
                core_cache = self.net._core(x)
                net_heads = self.net._heads(core_cache)
                if self.prior_scale <= 0.:
                    return net_heads
                else:
                    prior_core_cache = self.prior._core(x)
                    prior_heads = self.prior._heads(prior_core_cache)
                    return [n + self.prior_scale * p.detach() for n, p in zip(net_heads, prior_heads)]
        else:
            raise ValueError("Only works with a net_list model")

prior_net_ensemble = EnsembleNet(N_ENSEMBLE)
policy_net_ensemble = EnsembleNet(N_ENSEMBLE)
policy_net = NetWithPrior(policy_net_ensemble, prior_net_ensemble, PRIOR_SCALE).to(DEVICE)

target_net_ensemble = EnsembleNet(N_ENSEMBLE)
target_net = NetWithPrior(target_net_ensemble, prior_net_ensemble, PRIOR_SCALE).to(DEVICE)

opt = optim.Adam(policy_net.parameters(), lr=ADAM_LEARNING_RATE)

# change minibatch setup to use masking...
exp_replay = experience_replay(batch_size=BATCH_SIZE, max_size=N_ENSEMBLE * BUFFER_SIZE,
                               history_size=HISTORY_SIZE, write_buffer_every=WRITE_BUFFER_EVERY)
next(exp_replay) # Start experience-replay coroutines

# Stores the total rewards from each evaluation, per head over all epochs
accumulation_rewards = []
overall_time = 0.
env = DMAtariEnv('Breakout',random_seed=34)
eval_env = DMAtariEnv('Breakout',random_seed=55)
action_space = np.arange(env.env.action_space.n)

for i in range(N_EPOCHS):
    start = time.time()
    #ep = episode()
    episodic_reward = 0
    S, action, reward, finished = env.reset()
    ongoing_flag = int(finished)
    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
    experience =  [S, action, reward, ongoing_flag, exp_mask, -1]
    batch = exp_replay.send(experience)
    S_hist = [S for _ in range(HISTORY_SIZE)]
    epoch_losses = [0. for k in range(N_ENSEMBLE)]
    epoch_steps = [1. for k in range(N_ENSEMBLE)]
    heads = list(range(N_ENSEMBLE))
    random_state.shuffle(heads)
    active_head = heads[0]
    finished = False
    policy_net.train()
    episode_actions = []
    cnt = 0
    while not finished:
        if (random_state.rand() < EPSILON):
            action = random_state.choice(action_space)
            used = -1
        else: # Get the index of the maximum q-value of the model.
            with torch.no_grad():
                used = active_head
                policy_net_output = policy_net(torch.Tensor(S_hist)[None].to(DEVICE),active_head).detach().cpu().data.numpy()
                action = np.argmax(policy_net_output, axis=-1)[0]

        episode_actions.append(action)
        S_prime, reward, finished = env.step4(action)
        if not cnt%10:
            print('%05d'%cnt,action,used,reward)
        episodic_reward+=reward
        ongoing_flag = int(finished)
        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
        experience =  [S_prime, action, reward, ongoing_flag, exp_mask,used]
        S_hist.pop(0)
        S_hist.append(S_prime)
        S = S_prime
        batch = exp_replay.send(experience)
        cnt+=1
        if batch:
            #print('batch')
            inputs = []
            actions = []
            rewards = []
            nexts = []
            ongoing_flags = []
            masks = []
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
                    #next_max_Qs = next_max_Qs.cpu()
                    next_max_Qs = torch.Tensor(ongoing_flags).to(DEVICE) * next_max_Qs
                    target_Qs = torch.Tensor(np.array(rewards).astype("float32")).to(DEVICE) + GAMMA * next_max_Qs

                    # get current step predictions
                    #Qs = all_Qs[k].cpu()
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
    stop = time.time()
    overall_time += stop - start
    print("EPISODIC_REWARD", episodic_reward)
    print("episode steps", cnt)
    print(episode_actions)

    if TARGET_UPDATE > 1 and i % TARGET_UPDATE == 0:
        print("Updating target network at {}".format(i))
        target_net.load_state_dict(policy_net.state_dict())

    if i % PRINT_EVERY == 0:
        print("Epoch {}, head {}, loss: {}".format(i + 1, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))

    if i % EVALUATE_EVERY == 0 or i == (N_EPOCHS - 1):
        if i == (N_EPOCHS - 1):
            # save images at the end for sure
            SAVE_IMAGES = True
            ORIG_N_EVALUATIONS = N_EVALUATIONS
            N_EVALUATIONS = 5
        if SAVE_IMAGES:
            img_saver = save_img(i)
            next(img_saver)
        evaluation_rewards = []
        for eval_epoch in range(N_EVALUATIONS):
            evaluate_episodic_reward = 0
            S, action, reward, finished = eval_env.reset()
            ongoing_flag = int(finished)
            S_hist = [S]

            #S, reward = next(g)
            reward_trace = [reward]
            if SAVE_IMAGES:
                img_saver.send((S, reward))
            policy_net.eval()
            while not finished:
                if len(S_hist) == HISTORY_SIZE:
                    S_hist_pt = torch.Tensor(np.array(S_hist)[None]).to(DEVICE)
                    acts = [np.argmax(q.data.numpy(), axis=-1)[0] for q in policy_net(S_hist_pt, None)]
                else:
                    acts = [random_state.choice(action_space) for h in range(N_ENSEMBLE)]
                act_counts = Counter(acts)
                max_count = max(act_counts.values())
                top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count]
                # break action ties with random choice
                random_state.shuffle(top_actions)
                act = top_actions[0]
                #S, reward = g.send(act)
                S_prime, reward, finished = env.step4(act)
                evaluate_episodic_reward += reward
                reward_trace.append(reward)
                if SAVE_IMAGES:
                    img_saver.send((S_prime, reward))
                S_hist.pop(0)
                S_hist.append(S_prime)
                S = S_prime
            evaluation_rewards.append(np.sum(reward_trace))
            print("Evaluation Episode eval_epoch {} reward {}".format(eval_epoch, evaluation_rewards[-1]))
        accumulation_rewards.append(np.mean(evaluation_rewards))
        print("Mean evaluation reward all eval epochs {}".format(accumulation_rewards[-1]))
        if SAVE_IMAGES:
            img_saver.close()

    if i == (N_EPOCHS - 1):
        plt.figure()
        trace = np.array(accumulation_rewards)
        xs = np.array([int(n * EVALUATE_EVERY) for n in range(N_EPOCHS // EVALUATE_EVERY + 1)])
        plt.plot(xs, trace, label="Reward")
        plt.legend()
        plt.ylabel("Average Evaluation Reward ({})".format(ORIG_N_EVALUATIONS))

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

