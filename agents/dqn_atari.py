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

DEVICE = 'cuda'
# how many past frames to use for state input
HISTORY_SIZE = 4
# How often to check and evaluate
EVALUATE_EVERY = 10
# Save images of policy at each evaluation if True, otherwise only at the end if False
SAVE_IMAGES = False
# How often to print statistics
PRINT_EVERY = 1

# Whether to use double DQN or regular DQN
USE_DOUBLE_DQN = True
# TARGET_UPDATE how often to use replica target
TARGET_UPDATE = 30
# Number of evaluation episodes to run
N_EVALUATIONS = 30
# Number of heads for ensemble (1 falls back to DQN)
N_ENSEMBLE = 3
# Probability of experience to go to each head
BERNOULLI_P = 1.
# Weight for randomized prior, 0. disables
PRIOR_SCALE = 1.
# Number of episodes to run
N_EPOCHS = 1000
# Batch size to use for learning
BATCH_SIZE = 16
# Buffer size for experience replay
BUFFER_SIZE = 1000
# Epsilon greedy exploration ~prob of random action, 0. disables
EPSILON = .0
# Gamma weight in Q update
GAMMA = .8
# Gradient clipping setting
CLIP_GRAD = 1
# Learning rate for Adam
ADAM_LEARNING_RATE = 1E-3

random_state = np.random.RandomState(11)

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(22)

def save_img(epoch):
    if 'images_{}'.format(epoch) not in os.listdir('.'):
        os.mkdir('images_{}'.format(epoch))
    frame = 0
    while True:
        screen, reward = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.title("reward: {}".format(reward))
        plt.savefig('images_{}/{}.png'.format(epoch, frame))
        frame += 1

class CoreNet(nn.Module):
    def __init__(self, chans=4, n=16, network_output_size=48):
        super(CoreNet, self).__init__()
        self.network_output_size = network_output_size
        self.n = n
        self.chans = chans
        self.kernel_size = 3
        self.conv1 = nn.Conv2d(self.chans, self.n, self.kernel_size, 1, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.n, self.n, self.kernel_size, 1, padding=(1, 1))
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(-1, self.network_output_size * self.network_output_size * self.n)
        return x


class HeadNet(nn.Module):
    def __init__(self, n=16, network_output_size=48, n_actions=4):
        super(HeadNet, self).__init__()
        self.network_output_size = network_output_size
        self.n = n
        mult = self.network_output_size * self.network_output_size
        self.fc1 = nn.Linear(mult * self.n, mult)
        self.fc2 = nn.Linear(mult, n_actions)

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
exp_replay = experience_replay(batch_size=BATCH_SIZE, max_size=N_ENSEMBLE * BUFFER_SIZE, history_size=HISTORY_SIZE)
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
    S, action, reward, finished = env.reset()
    ongoing_flag = int(finished)
    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
    experience =  [S, action, reward, ongoing_flag, exp_mask]
    batch = exp_replay.send(experience)
    S_hist = [S]
    epoch_losses = [0. for k in range(N_ENSEMBLE)]
    epoch_steps = [1. for k in range(N_ENSEMBLE)]
    heads = list(range(N_ENSEMBLE))
    random_state.shuffle(heads)
    active_head = heads[0]
    finished = False
    policy_net.train()
    while not finished:
        action = random_state.choice(action_space)
        if (random_state.rand() < EPSILON) or (len(S_hist) < HISTORY_SIZE):
            action = random_state.choice(action_space)
        else: # Get the index of the maximum q-value of the model.
            # Subtract one because actions are either -1, 0, or 1
            with torch.no_grad():
                policy_net_output = policy_net(torch.Tensor(S_hist)[None].to(DEVICE),active_head).detach().cpu().data.numpy()
                action = np.argmax(policy_net_output, axis=-1)[0]

        #S_prime, won = ep.send(action)
        S_prime, reward, finished = env.step4(action)
        ongoing_flag = int(finished)
        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
        #experience = (S, action, reward, S_prime, ongoing_flag, exp_mask)
        experience =  [S_prime, action, reward, ongoing_flag, exp_mask]
        S_hist.append(S_prime)
        if len(S_hist) >= HISTORY_SIZE:
            S_hist.pop(0)
        S = S_prime
        batch = exp_replay.send(experience)
        if batch:
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
            mask = torch.Tensor(np.array(masks))
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
                next_max_Qs = next_max_Qs.cpu()
                next_max_Qs = torch.Tensor(ongoing_flags) * next_max_Qs
                target_Qs = torch.Tensor(np.array(rewards).astype("float32")) + GAMMA * next_max_Qs

                # get current step predictions
                Qs = all_Qs[k].cpu()
                Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32")))
                Qs = Qs.squeeze()

                # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
                # need to do updates on each head based on experience mask
                full_loss = (Qs - target_Qs) ** 2
                full_loss = mask[:, k] * full_loss
                #loss = torch.mean(full_loss)
                loss = torch.sum(full_loss / torch.sum(mask[:, k]))
                #loss = F.smooth_l1_loss(Qs, target_Qs[:, None])

                loss.backward(retain_graph=True)
                for param in policy_net.parameters():
                    if param.grad is not None:
                        # Multiply grads by 1 / K
                        param.grad.data *= 1. / N_ENSEMBLE
                epoch_losses[k] += loss.detach().cpu().numpy()
                epoch_steps[k] += 1.
            # After iterating all heads, do the update step
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), CLIP_GRAD)
            opt.step()
    #except StopIteration:
    #    # add the end of episode experience
    #    ongoing_flag = 0.
    #    # just put in S, since it will get masked anyways
    #    exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
    #    experience = [S, action, won, S, ongoing_flag, exp_mask]
    #    exp_replay.send(experience)

    stop = time.time()
    overall_time += stop - start

#    if TARGET_UPDATE > 0 and i % TARGET_UPDATE == 0:
#        print("Updating target network at {}".format(i))
#        target_net.load_state_dict(policy_net.state_dict())
#
#    if i % PRINT_EVERY == 0:
#        print("Epoch {}, head {}, loss: {}".format(i + 1, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))
#
#    if i % EVALUATE_EVERY == 0 or i == (N_EPOCHS - 1):
#        if i == (N_EPOCHS - 1):
#            # save images at the end for sure
#            SAVE_IMAGES = True
#            ORIG_N_EVALUATIONS = N_EVALUATIONS
#            N_EVALUATIONS = 5
#        if SAVE_IMAGES:
#            img_saver = save_img(i)
#            next(img_saver)
#        evaluation_rewards = []
#        for _ in range(N_EVALUATIONS):
#            #g = episode()
#            S = eval_env.reset()
#            finished = False
#            #S, reward = next(g)
#            reward = 0
#            reward_trace = [reward]
#            if SAVE_IMAGES:
#                img_saver.send((S, reward))
#            policy_net.eval()
#            while not finished:
#                acts = [np.argmax(q.data.numpy(), axis=-1)[0] for q in policy_net(torch.Tensor(S[None]), None)]
#                act_counts = Counter(acts)
#                max_count = max(act_counts.values())
#                top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count]
#                # break action ties with random choice
#                random_state.shuffle(top_actions)
#                act = top_actions[0]
#                #S, reward = g.send(act)
#                S_prime, reward, finished = env.step4(act)
#                reward_trace.append(reward)
#                if SAVE_IMAGES:
#                    img_saver.send((S_prime, reward))
#                S = S_prime
#            #except StopIteration:
#            #    # sum should be either -1 or +1
#            #    evaluation_rewards.append(np.sum(reward_trace))
#            accumulation_rewards.append(np.mean(evaluation_rewards))
#        print("Evaluation reward {}".format(accumulation_rewards[-1]))
#        if SAVE_IMAGES:
#            img_saver.close()
#
#    if i == (N_EPOCHS - 1):
#        plt.figure()
#        trace = np.array(accumulation_rewards)
#        xs = np.array([int(n * EVALUATE_EVERY) for n in range(N_EPOCHS // EVALUATE_EVERY + 1)])
#        plt.plot(xs, trace, label="Reward")
#        plt.legend()
#        plt.ylabel("Average Evaluation Reward ({})".format(ORIG_N_EVALUATIONS))
#
#        model = "Double DQN" if USE_DOUBLE_DQN else "DQN"
#        if N_ENSEMBLE > 1:
#            model = "Bootstrap " + model
#        if PRIOR_SCALE > 0.:
#            model = model + " with randomized prior {}".format(PRIOR_SCALE)
#        footnote_text = "Episodes\n"
#        footnote_text += "\n"
#        footnote_text += "\n"
#        footnote_text += "Settings:\n"
#        footnote_text += "{}\n".format(model)
#        footnote_text += "Number of heads {}\n".format(N_ENSEMBLE)
#        footnote_text += "Epsilon-greedy {}\n".format(EPSILON)
#        if N_ENSEMBLE > 1:
#            footnote_text += "Sharing mask probability {}\n".format(BERNOULLI_P)
#        footnote_text += "Gamma decay {}\n".format(GAMMA)
#        footnote_text += "Grad clip {}\n".format(CLIP_GRAD)
#        footnote_text += "Adam, learning rate {}\n".format(ADAM_LEARNING_RATE)
#        footnote_text += "Batch size {}\n".format(BATCH_SIZE)
#        footnote_text += "Experience replay buffer size {}\n".format(BUFFER_SIZE)
#        footnote_text += "Training time {}\n".format(overall_time)
#        plt.xlabel(footnote_text)
#        plt.tight_layout()
#        plt.savefig("reward_traces.png")

