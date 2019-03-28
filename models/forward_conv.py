import os
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
from IPython import embed
torch.manual_seed(394)

# from pytorch implementation of resnet in torchvision

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ForwardResNet(nn.Module):
    def __init__(self, block, data_width=10, num_channels=4, num_output_channels=512, num_rewards=3, zero_init_residual=False):
        # num output channels will be num clusters
        super(ForwardResNet, self).__init__()

        self.num_rewards = num_rewards

        self.inplanes = 64
        netc = 64
        self.conv1 = nn.Conv2d(num_channels, netc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, netc, 2, stride=1)
        self.layer2 = self._make_layer(block, netc, 2, stride=1)
        self.layer3 = self._make_layer(block, netc, 2, stride=1)
        self.layer_rec = self._make_layer(block, netc, 2, stride=1)
        self.layer_last = nn.Conv2d(netc, num_output_channels, kernel_size=1)

        # divide by two bc of stride length
        #self.layer_action = self._make_layer(block, netc, 1, stride=1)
        #self.layer_out_action = nn.Conv2d(netc, self.num_actions, kernel_size=data_width)

        self.layer_reward = self._make_layer(block, netc, 1, stride=1)
        self.layer_out_reward = nn.Conv2d(netc, self.num_rewards, kernel_size=data_width)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # bs,c,h,w
        nx = F.log_softmax(self.layer_last(self.layer_rec(x)), dim=1)
        # action output should be bs,n_actions,1,1]
        #act = F.log_softmax(self.layer_out_action(self.layer_action(x))[:,:,0,0], dim=1)
        # reward output should be bs,n_rewards,1,1]
        reward = F.log_softmax(self.layer_out_reward(self.layer_reward(x))[:,:,0,0], dim=1)
        return nx, reward

if __name__ == '__main__':
    model = ForwardResNet(BasicBlock, data_width=10, num_channels=1)
    fd = torch.zeros((128,1,10,10))
    ffd,a,r  = model(fd)

