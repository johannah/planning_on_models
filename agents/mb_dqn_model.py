
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

class CoreNetEmbedding(nn.Module):
    def __init__(self, reshape_size=16*10*10, num_channels=1, num_clusters=512):
        super(CoreNetEmbedding, self).__init__()
        self.num_clusters=num_clusters
        eo = 64
        self.num_channels = num_channels
        self.embeddings = nn.ModuleList()
        for e in range(self.num_channels):
            self.embeddings.append(nn.Embedding(num_clusters, eo))
        self.reshape_size = reshape_size
        # filter_size is too small at 1 - kk
        #self.conv1 = nn.Conv2d(eo*self.num_channels, eo*2, 1, 1)
        #self.conv2 = nn.Conv2d(eo*2, eo*1, 1, 1)
        #self.conv3 = nn.Conv2d(eo, 16, 1, 1)
        self.conv1 = nn.Conv2d(eo*self.num_channels, eo*2, 3, 1, padding=(1,1))
        self.conv2 = nn.Conv2d(eo*2, eo*1, 3, 1, padding=(1,1))
        self.conv3 = nn.Conv2d(eo, 16, 3, 1, padding=(1,1))
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)

    def forward(self, x):
        for e in range(self.num_channels):
            if not e:
                xo = self.embeddings[e](x[:,e]).permute(0,3,1,2)
            else:
                xo = torch.cat((xo,self.embeddings[e](x[:,e]).permute(0,3,1,2)), dim=1)
        # bs, 16, 10, 10
        xo = F.relu(self.conv1(xo))
        xo = F.relu(self.conv2(xo))
        xo = F.relu(self.conv3(xo))
        # size after conv3
        xo = xo.view(-1, self.reshape_size)
        return xo

class CoreNet(nn.Module):
    def __init__(self, reshape_size=16*10*10, num_channels=1):
        super(CoreNet, self).__init__()
        self.reshape_size = reshape_size
        self.num_channels = num_channels
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(self.num_channels, 64, 3, 1, padding=(1,1))
        self.conv2 = nn.Conv2d(64, 32, 3, 1, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 16, 3, 1, padding=(1,1))
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        x = x.view(-1, self.reshape_size)
        return x

class DuelingHeadNet(nn.Module):
    def __init__(self, n_actions=4, reshape_size=10*10*4):
        super(DuelingHeadNet, self).__init__()
        self.split_size = 1024
        self.fc1 = nn.Linear(reshape_size, self.split_size*2)
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
    def __init__(self, n_actions=4, reshape_size=10*10*4):
        super(HeadNet, self).__init__()
        self.split_size = 1024*2
        self.fc1 = nn.Linear(reshape_size, self.split_size)
        self.fc2 = nn.Linear(self.split_size, n_actions)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, reshape_size, num_channels, dueling=False, num_clusters=0, use_embedding=True):
        super(EnsembleNet, self).__init__()
        if not use_embedding:
            self.core_net = CoreNet(reshape_size=reshape_size, num_channels=num_channels)
        else:
            self.core_net = CoreNetEmbedding(reshape_size=reshape_size, num_channels=num_channels, num_clusters=num_clusters)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.net_list = nn.ModuleList([DuelingHeadNet(n_actions=n_actions, reshape_size=reshape_size) for k in range(n_ensemble)])
        else:
            self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions, reshape_size=reshape_size) for k in range(n_ensemble)])

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


