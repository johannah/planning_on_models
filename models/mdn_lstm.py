from copy import deepcopy
import time
import os, sys
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
rdn = np.random.RandomState(33)
# TODO one-hot the action space?
torch.manual_seed(139)
epsilon = 1e-6

class mdnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, number_mixtures=20):
        super(mdnLSTM, self).__init__()
        self.number_mixtures = number_mixtures
        # one extra output for pen up
        # multiple by 3 for each of the two output
        self.output_size = self.number_mixtures*6
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

    def get_mixture_coef(self, output):
        # split of data into pi,sigma,mu for each feature dimension
        z = output
        # z_os is end of stroke signal
        # split into six pieces
        z_pi, out_mu1, out_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z, self.number_mixtures, dim=1)
        # softmax the pis
        max_pi,_ = torch.max(z_pi, dim=1, keepdim=True)
        exp_pi = torch.exp(z_pi-max_pi)
        out_pi = exp_pi/torch.sum(exp_pi, dim=1, keepdim=True)
        # don't allow sigma to get too small
        out_sigma1 = torch.exp(z_sigma1)+1e-3
        out_sigma2 = torch.exp(z_sigma2)+1e-3
        out_corr = torch.tanh(z_corr)
        return out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr

    def pt_2d_normal(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        norm1 = x1-mu1
        norm2 = x2-mu2
        s1s2 = sigma1*sigma2
        z = (norm1/sigma1)**2 + (norm2/sigma2)**2 - 2*((rho*(norm1*norm2))/s1s2)
        negRho = 1-(rho**2) + 0.05
        result = torch.exp(-z/(2*negRho))
        denom = 2*np.pi*(s1s2*torch.sqrt(negRho))
        result = result/(denom)
        return result

    def get_lossfunc(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
        result0 = self.pt_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        result1 = torch.sum(result0*z_pi, dim=1, keepdim=True)
        result = -torch.log(result1+epsilon)
        result = torch.mean(result)
        return result

