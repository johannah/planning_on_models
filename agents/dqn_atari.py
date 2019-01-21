# extending on code from
# https://github.com/58402140/Fruit
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import numpy as np
import copy
import time
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Size of game grid e.g. 10 -> 10x10
GRID_SIZE = 10
# How often to check and evaluate
EVALUATE_EVERY_EPOCHS = 100
# Save images of policy at each evaluation if True, otherwise only at the end if False
SAVE_IMAGES = True
LIMIT_NUM_IMAGES = 100
# How often to print statistics
PRINT_EVERY_EPOCHS = 1

# Whether to use double DQN or regular DQN
USE_DOUBLE_DQN = True
# TARGET_UPDATE 0 uses policy network, otherwise use replica target
TARGET_UPDATE = 10
# Number of evaluation episodes to run
N_EVALUATIONS = 100
# Number of heads for ensemble (1 falls back to DQN)
N_ENSEMBLE = 3
# Probability of experience to go to each head
BERNOULLI_P = .5
# Weight for randomized prior, 0. disables
PRIOR_SCALE = 1.
# Number of steps total to run
N_STEPS = 10000
# Batch size to use for learning
BATCH_SIZE = 32
# Buffer size for experience replay
BUFFER_SIZE = 1000
# Epsilon greedy exploration ~prob of random action, 0. disables
EPSILON = .0
# Gamma weight in Q update
GAMMA = .99
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
        if frame < LIMIT_NUM_IMAGES:
            screen, reward = (yield)
            plt.imshow(screen[0], interpolation='none')
            plt.title("reward: {}".format(reward))
            plt.savefig('images_{}/{:0>5d}.png'.format(epoch, frame))
            frame += 1

def episode():
    """
    Coroutine of episode.

    Action has to be explicitly sent to this coroutine.
    actions are 0, 1, 2 for left, don't-move, and right
    """
    x, y, z = (
        random_state.randint(0, GRID_SIZE),  # X of fruit
        0,  # Y of dot
        random_state.randint(1, GRID_SIZE - 1)  # X of basket
    )
    while True:
        X = np.zeros((GRID_SIZE, GRID_SIZE))  # Reset grid
        X = X.astype("float32")
        X[y, x] = 1.  # Draw fruit
        bar = range(z - 1, z + 2)
        X[-1, bar] = 1.  # Draw basket

        # End of game is known when fruit is at penultimate line of grid.
        # End represents either a win or a loss
        end = int(y >= GRID_SIZE - 2)

        reward = 0
        # can add this for dense rewards
        #if x in bar:
        #   reward = 1
        if end and x not in bar:
           reward = -1
        if end and x in bar:
           reward = 1

        action = yield X[None], reward #end
        if end:
            break

        # translate actions
        # 0 is left
        # 1 is same (0)
        # 2 is right
        action = action - 1

        z = min(max(z + action, 1), GRID_SIZE - 2)
        y += 1


def experience_replay(batch_size, max_size):
    """
    Coroutine of experience replay.

    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        inds = np.arange(len(memory))
        experience = yield [memory[i] for i in random_state.choice(inds, size=batch_size, replace=True)] if batch_size <= len(memory) else None
        # send None to just get random experiences, without changing buffer
        if experience is not None:
            memory.append(experience)
            if len(memory) > max_size:
                memory.pop(0)


class CoreNet(nn.Module):
    def __init__(self):
        super(CoreNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(-1, 10 * 10 * 16)
        return x


class HeadNet(nn.Module):
    def __init__(self):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(10 * 10 * 16, 100)
        self.fc2 = nn.Linear(100, 3)

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
                from IPython import embed; embed(); raise ValueError()
                return self.net._heads(core_cache)
        else:
            raise ValueError("Only works with a net_list model")

prior_net = EnsembleNet(N_ENSEMBLE)
policy_net = EnsembleNet(N_ENSEMBLE)
policy_net = NetWithPrior(policy_net, prior_net, PRIOR_SCALE)

target_net = EnsembleNet(N_ENSEMBLE)
target_net = NetWithPrior(target_net, prior_net, PRIOR_SCALE)

opt = optim.Adam(policy_net.parameters(), lr=ADAM_LEARNING_RATE)

# change minibatch setup to use masking...
exp_replays = [experience_replay(BATCH_SIZE, BUFFER_SIZE) for k in range(N_ENSEMBLE)]
[next(e) for e in exp_replays] # Start experience-replay coroutines

# Stores the total rewards from each evaluation, per head over all epochs
accumulated_rewards = []
overall_time = 0.
steps = 0
epochs = 0
last_target_update = 0
while steps < N_STEPS:
    start = time.time()
    ep = episode()
    S, won = next(ep)  # Start coroutine of single episode
    epoch_losses = [0. for k in range(N_ENSEMBLE)]
    epoch_steps = [1. for k in range(N_ENSEMBLE)]
    heads = list(range(N_ENSEMBLE))
    random_state.shuffle(heads)
    active_head = heads[0]
    epochs +=1
    print("Staring EPOCH {}".format(epochs))
    try:
        policy_net.train()
        while True:
            if random_state.rand() < EPSILON:
                action = random_state.randint(0, 3)
            else:
                # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                with torch.no_grad():
                    action = np.argmax(policy_net(torch.Tensor(S[None]),active_head).detach().data.numpy(), axis=-1)[0]

            S_prime, won = ep.send(action)
            steps += 1
            ongoing_flag = 1.
            experience = (S, action, won, S_prime, ongoing_flag)
            S = S_prime
            k_order = [k for k in range(N_ENSEMBLE)]
            random_state.shuffle(k_order)
            for k in k_order:
                exp_replay = exp_replays[k]
                # reset batch to default None, since we are looping
                batch = None
                # .5 probability of adding to each buffer for bootstrap dqn, see paper for details
                if random_state.rand() > BERNOULLI_P or N_ENSEMBLE == 1:
                    batch = exp_replay.send(experience)
                else:
                    continue
                if batch:
                    inputs = []
                    actions = []
                    rewards = []
                    nexts = []
                    ongoing_flags = []
                    for s, a, r, s_prime, ongoing_flag in batch:
                        rewards.append(r)
                        inputs.append(s)
                        actions.append(a)
                        nexts.append(s_prime)
                        ongoing_flags.append(ongoing_flag)

                    if TARGET_UPDATE == 0:
                        next_Qs = policy_net(torch.Tensor(nexts),k).detach()
                    else:
                        next_Qs = target_net(torch.Tensor(nexts),k).detach()

                    if USE_DOUBLE_DQN:
                        # double Q
                        policy_next_Qs = policy_net(torch.Tensor(nexts), k).detach()
                        policy_actions = policy_next_Qs.max(1)[1][:, None]
                        next_max_Qs = next_Qs.gather(1, policy_actions)
                        next_max_Qs = next_max_Qs.squeeze()
                    else:
                        # standard Q
                        next_max_Qs = next_Qs.max(1)[0]
                        next_max_Qs = next_max_Qs.squeeze()

                    # mask based on if it is end of episode or not
                    next_max_Qs = torch.Tensor(ongoing_flags) * next_max_Qs
                    target_Qs = torch.Tensor(np.array(rewards).astype("float32")) + GAMMA * next_max_Qs

                    # get current step predictions
                    Qs = policy_net(torch.Tensor(inputs), k)
                    Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32")))
                    Qs = Qs.squeeze()

                    # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
                    loss = torch.mean((Qs - target_Qs) ** 2)
                    #loss = F.smooth_l1_loss(Qs, target_Qs[:, None])

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(policy_net.parameters(), CLIP_GRAD)
                    opt.step()
                    epoch_losses[k] += loss.detach().cpu().numpy()
                    epoch_steps[k] += 1.

            if TARGET_UPDATE > 0 and (steps-last_target_update) > TARGET_UPDATE:
                last_target_update = steps
                print("Updating target network at {} steps".format(steps))
                target_net.load_state_dict(policy_net.state_dict())

    except StopIteration:
        # add the end of episode experience
        ongoing_flag = 0.
        # just put in S, since it will get masked anyways
        experience = (S, action, won, S, ongoing_flag)
        exp_replay.send(experience)

    stop = time.time()
    overall_time += stop - start
    if epochs % PRINT_EVERY_EPOCHS == 0:
        print("Epoch {} Step {}, head {}, loss: {}".format(epochs, steps, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))

    if epochs % EVALUATE_EVERY_EPOCHS == 0 or steps > (N_STEPS - 1):
        print("EVALUATING at epoch {}".format(epochs))
        if steps > (N_STEPS - 1):
            # save images at the end for sure
            SAVE_IMAGES = True
            ORIG_N_EVALUATIONS = N_EVALUATIONS
            N_EVALUATIONS = 5
        if SAVE_IMAGES:
            img_saver = save_img(epochs)
            next(img_saver)
        ensemble_rewards = []
        for k in range(N_ENSEMBLE):
            avg_rewards = []
            print("Evaluation of head {}".format(k))
            for _ in range(N_EVALUATIONS):
                g = episode()
                S, reward = next(g)
                if SAVE_IMAGES:
                    img_saver.send((S, reward))
                episode_rewards = [reward]
                try:
                    policy_net.eval()
                    while True:
                        act = np.argmax(policy_net(torch.Tensor(S[None]), k).data.numpy(), axis=-1)[0]
                        S, reward = g.send(act)
                        episode_rewards.append(reward)
                        if SAVE_IMAGES:
                            img_saver.send((S, reward))
                except StopIteration:
                    # sum should be either -1 or +1
                    avg_rewards.append(np.sum(episode_rewards))
            ensemble_rewards.append(np.mean(avg_rewards))
        print("Rewards {}".format(ensemble_rewards))
        accumulated_rewards.append(ensemble_rewards)
        if SAVE_IMAGES:
            img_saver.close()

    if steps >= (N_STEPS):
        plt.figure()
        for k in range(N_ENSEMBLE):
            trace = np.array([ar[k] for ar in accumulated_rewards])
            # this is probably wrong
            xs = np.array([int(n * EVALUATE_EVERY_EPOCHS) for n in range(epochs // EVALUATE_EVERY_EPOCHS + 1)])
            plt.plot(xs, trace, label="Head {}".format(k))
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
        footnote_text += "Gamma decay {}\n".format(GAMMA)
        footnote_text += "Grad clip {}\n".format(CLIP_GRAD)
        footnote_text += "Adam, learning rate {}\n".format(ADAM_LEARNING_RATE)
        footnote_text += "Batch size {}\n".format(BATCH_SIZE)
        footnote_text += "Experience replay buffer size {}\n".format(BUFFER_SIZE)
        footnote_text += "Training time {}\n".format(overall_time)
        plt.xlabel(footnote_text)
        plt.tight_layout()
        plt.savefig("reward_traces.png")

