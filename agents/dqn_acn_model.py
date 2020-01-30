
# Model style from Kyle @
# https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed

# from the DQN paper
#The first convolution layer convolves the input with 32 filters of size 8 (stride 4),
#the second layer has 64 layers of size 4
#(stride 2), the final convolution layer has 64 filters of size 3 (stride
#1). This is followed by a fully-connected hidden layer of 512 units.

# init func used by hengyaun
def weights_init(m):
    """custom weights initialization"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        pass
        #print("default init")
        #m.weight.data.normal_(0.0, 0.02)
        #m.bias.data.fill_(0)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' %classtype)


class CoreNet(nn.Module):
    def __init__(self, code_length, num_hidden=84):
        super(CoreNet, self).__init__()
        self.code_length = code_length
        # params from ddqn appendix
        nh = num_hidden
        self.input_layer = nn.Linear(self.code_length, nh)
        self.lin1 = nn.Linear(nh, nh)
        self.lin2 = nn.Linear(nh, nh)
        self.lin3 = nn.Linear(nh, nh)
        self.skipin_to_2 = nn.Linear(nh, nh)
        self.skipin_to_3 = nn.Linear(nh, nh)
        self.skip1_to_out = nn.Linear(nh, nh)
        self.skip2_to_out = nn.Linear(nh, nh)
        self.h1 = nn.Linear(nh, nh)
        self.h2 = nn.Linear(nh, nh)
        self.h3 = nn.Linear(nh, nh)

    def forward(self, x):
        # Skip connections like prior net
        i = F.relu(self.input_layer(x.view(x.shape[0], self.code_length)))
        _h1 = F.relu(self.lin1(i))
        _s2 = F.relu(self.skipin_to_2(_h1))
        _s3 = F.relu(self.skipin_to_3(_h1))

        _h2 = F.relu(self.lin2(_h1+_s2))

        _o1 = F.relu(self.skip1_to_out(_h1))
        _o2 = F.relu(self.skip2_to_out(_h2))
        _o3 = F.relu(self.lin3(_h2+_s3))
        out = _o1+_o2+_o3
        return out

class DuelingHeadNet(nn.Module):
    def __init__(self, input_size, n_actions=4):
        super(DuelingHeadNet, self).__init__()
        self.input_size = input_size
        self.split_size = 36
        self.fc1 = nn.Linear(self.input_size, self.split_size*2)
        self.value = nn.Linear(self.split_size, 1)
        self.advantage = nn.Linear(self.split_size, n_actions)
        self.fc1.apply(weights_init)
        self.value.apply(weights_init)
        self.advantage.apply(weights_init)

    def forward(self, x):
        x1,x2 = torch.split(F.relu(self.fc1(x)), self.split_size, dim=1)
        value = self.value(x1)
        advantage = self.advantage(x2)
        # value is shape [batch_size, 1]
        # advantage is shape [batch_size, n_actions]
        q = value + torch.sub(advantage, torch.mean(advantage, dim=1, keepdim=True))
        return q

class HeadNet(nn.Module):
    def __init__(self, input_size, n_actions):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 36)
        self.fc2 = nn.Linear(36, n_actions)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, code_length, num_hidden, dueling=False):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(code_length, num_hidden)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.net_list = nn.ModuleList([DuelingHeadNet(input_size=num_hidden, n_actions=n_actions) for k in range(n_ensemble)])
        else:
            self.net_list = nn.ModuleList([HeadNet(input_size=num_hidden, n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        embed()
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads

class NetWithPrior(nn.Module):
    def __init__(self, net, prior, prior_scale=1.):
        super(NetWithPrior, self).__init__()
        self.net = net
        # used when scaling core net
        self.core_net = self.net.core_net
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


