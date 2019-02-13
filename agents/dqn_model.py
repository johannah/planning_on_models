
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
#The first convolution layer convolves the input with 32 filters of size 8 (stride 4),
#the second layer has 64 layers of size 4
#(stride 2), the final convolution layer has 64 filters of size 3 (stride
#1). This is followed by a fully-connected hidden layer of 512 units.
class CoreNet(nn.Module):
    def __init__(self, network_output_size=84, num_channels=4):
        super(CoreNet, self).__init__()
        self.network_output_size = network_output_size
        self.num_channels = num_channels
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        # TODO - should we have this init during PRIOR code?
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.conv1.bias.data.fill_(0.)
        self.conv2.bias.data.fill_(0.)
        self.conv3.bias.data.fill_(0.)
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

class DuelingHeadNet(nn.Module):
    def __init__(self, n_actions=4):
        super(DuelingHeadNet, self).__init__()
        mult = 64*7*7
        self.split_size = 512
        self.fc1 = nn.Linear(mult, self.split_size*2)
        self.value = nn.Linear(self.split_size, 1)
        self.advantage = nn.Linear(self.split_size, n_actions)
        self.fc1.bias.data.fill_(0.)
        self.value.bias.data.fill_(0.)
        self.advantage.bias.data.fill_(0.)

    def forward(self, x):
        #x1,x2 = torch.split(F.relu(self.fc1(x)), 2, dim=1)
        x1,x2 = torch.split(F.relu(self.fc1(x)), self.split_size, dim=1)
        value = self.value(x1)
        advantage = self.advantage(x2)
        # value is shape [batch_size, 1]
        # advantage is shape [batch_size, n_actions]
        q = value + torch.sub(advantage, torch.mean(advantage, dim=1, keepdim=True))
        return q

class HeadNet(nn.Module):
    def __init__(self, n_actions=4):
        super(HeadNet, self).__init__()
        mult = 64*7*7
        self.fc1 = nn.Linear(mult, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.fc1.bias.data.fill_(0.)
        self.fc2.bias.data.fill_(0.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, network_output_size, num_channels, dueling=False):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(network_output_size=network_output_size, num_channels=num_channels)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.net_list = nn.ModuleList([DuelingHeadNet(n_actions=n_actions) for k in range(n_ensemble)])
        else:
            self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
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


