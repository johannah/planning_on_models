"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

Strongly referenced ACN implementation and blog post from:
http://jalexvig.github.io/blog/associative-compression-networks/

Base VAE referenced from pytorch examples:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

# TODO conv
# TODO load function
# daydream function
import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
#from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
torch.manual_seed(394)
softplus = torch.nn.Softplus()

"""
\cite{acn} The ACN encoder was a convolutional
network fashioned after a VGG-style classifier (Simonyan
& Zisserman, 2014), and the encoding distribution q(z|x)
was a unit variance Gaussian with mean specified by the
output of the encoder network.
size of z is 16 for mnist, 128 for others
"""
class ConvVAE(nn.Module):
    def __init__(self, code_len, input_size=1, encoder_output_size=5):
        super(ConvVAE, self).__init__()
        self.code_len = code_len
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,
                      out_channels=code_len,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len),
            nn.ReLU(True),
           )
        # found via experimentation - 4 for mnist
        # input_image == 28 -> eo=7
        # encoder_output_size will vary based on input
        self.eo = encoder_output_size
        self.fc21 = nn.Linear(self.eo, code_len)

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        #print(o.shape, ol.shape)
        return self.fc21(ol)

    def reparameterize(self, mu):
        if self.training:
            eps = torch.randn_like(mu)
            o = eps.add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        return z, mu

def gau_kl3(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussians pm,pv to Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    returns KL of each G in pm, pv to all qm, qv
    """
    axis1 = 2
    axis2 = 3
    eps = 1E-3
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod(axis1)
    dqv = qv.prod(axis1)
    # Inverse of diagonal covariance qv
    iqv = 1. / qv
    # Difference between means pm, qm
    diff = qm[:, None] - pm[:, :, None]
    p1 = torch.log(dqv[:, None] / dpv[:, :, None])
    p2 = (iqv[:, None] * pv[:, :, None]).sum(axis2)
    p3 = (diff * iqv[:, None] * diff).sum(axis2)
    p4 = pm.shape[2]
    return 0.5 * (p1 + p2 + p3 - p4)

def log_gau_kl3(pm, lpv, qm, lqv):
    """
    Kullback-Liebler divergence from Gaussians pm,pv to Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    returns KL of each G in pm, pv to all qm, qv
    """
    axis1 = 2
    axis2 = 3
    # Determinants of diagonal covariances pv, qv
    dpv = lpv.sum(axis1)
    dqv = lqv.sum(axis1)

    # Inverse of diagonal covariance qv
    iqv = -lqv
    # Difference between means pm, qm
    diff = qm[:, None] - pm[:, :, None]
    p1 = dqv[:, None] - dpv[:, :, None]
    p2 = torch.exp(torch.logsumexp(iqv[:, None] + lpv[:, :, None], dim=axis2))
    p3 = (diff * torch.exp(iqv[:, None]) * diff).sum(axis2)
    p4 = pm.shape[2]
    return 0.5 * (p1 + p2 + p3 - p4)

def acn_mdn_loss_function(y_hat, y, u_q, pi_ps, u_ps, s_ps):
    ''' compare mdn with k=1 (u_q) to a true mdn

    '''
    # create pi of 1.0 for every sample in minibatch
    pi_q = torch.ones_like(u_q[:,:1])
    # add channel for "1 mixture"
    u_q = u_q[:,None]
    # expect logstd of 1.0 which is 0.0
    s_q = torch.zeros_like(u_q)

    # expects variance
    kl3_num = torch.exp(-log_gau_kl3(u_q, 2*s_q, u_q, 2*s_q))
    kl3_den = torch.exp(-log_gau_kl3(u_q, 2*s_q, u_ps, 2*s_ps))
    # Todo make sure this is the correct direction -
    # are there shortcuts since a is one mixture?
    nums = torch.sum(pi_q[:, None] * kl3_num, dim=2)
    dens = torch.sum(pi_ps[:, None] * kl3_den, dim=2)
    kl = pi_q * torch.log(nums / dens)
    sum_kl = torch.clamp(kl, 1./float(u_ps.shape[0]), 100).sum()
    rec_loss = F.binary_cross_entropy(y_hat, y, reduction='sum')
    if np.isinf(sum_kl.cpu().detach().numpy()) or np.isnan(sum_kl.cpu().detach().numpy()):
        print('sum_kl', sum_kl, 'rec', rec_loss)
        embed()
    if np.isinf(rec_loss.cpu().detach().numpy()) or np.isnan(rec_loss.cpu().detach().numpy()):
        print('sum_kl', sum_kl, 'rec', rec_loss)
        embed()
    # inf before train_cnt cnt 1574400
    return sum_kl+rec_loss


def acn_loss_function(y_hat, y, u_q, u_p, s_p):
    ''' reconstruction loss + coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     Args:
         y_hat: reconstruction output
         y: target
         u_q: mean of model posterior
         s_q: std deviation
         u_p: mean of conditional prior
         s_p: std of conditional prior

     Returns: loss
     '''
    # s_p, s_q used to be logvar, but we did softplus on them - now strictly
    # positive

    #BCE = F.binary_cross_entropy(y_hat, y, reduction='sum')
    #acn_KLD = torch.sum(s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    BCE = F.binary_cross_entropy(y_hat, y, reduction='sum')
    #acn_KLD = torch.sum(s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))

    # our implementation of full loss
    s_q = torch.ones_like(s_p)
    acn_KLD = torch.sum(torch.log(s_p)-torch.log(s_q) + 0.5*((s_q**2)/(s_p**2) + (((u_q-u_p)**2)/(s_p**2)) - 1.0))

    return BCE+acn_KLD


"""
/cite{acn} The prior network was an
MLP with three hidden layers each containing 512 tanh
units, and skip connections from the input to all hidden
layers and all hiddens to the output layer
"""
class PriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, n_mixtures=8, random_seed=4543):
        super(PriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.n_mixtures = n_mixtures
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        # skip connections from input to all hidden layers
        self.s1 = nn.Linear(self.code_length, n_hidden)
        self.s2 = nn.Linear(self.code_length, n_hidden)
        self.s3 = nn.Linear(self.code_length, n_hidden)
        # skip connections from all hidden layers to output layer
        self.sf1 = nn.Linear(n_hidden, n_hidden)
        self.sf2 = nn.Linear(n_hidden, n_hidden)

        # outputs
        # mean linear
        self.fc4_u = nn.Linear(n_hidden, self.n_mixtures*self.code_length)
        # log variance
        self.fc4_s = nn.Linear(n_hidden, self.n_mixtures*self.code_length)
        # m mixture softmax
        self.fc4_mix = nn.Linear(n_hidden, self.n_mixtures)

        self.knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
        codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
        self.fit_knn(codes)

    def encode(self, prev_code):
        h1 = torch.tanh(self.fc1(prev_code)) + self.s1(prev_code)
        h2 = torch.tanh(self.fc2(h1)) + self.s2(prev_code)
        h3 = torch.tanh(self.fc3(h2)) + self.s3(prev_code) + self.sf1(h1) + self.sf2(h2)
        means = self.fc4_u(h3)
        # todo change name - sigma is logvar
        sigmas = self.fc4_s(h3)
        mixes = torch.softmax(self.fc4_mix(h3), dim=1)
        return mixes, means, sigmas

    def fit_knn(self, codes):
        ''' will reset the knn  given an nd array
        '''
        st = time.time()
        self.codes = codes
        assert(len(self.codes)>1)
        y = np.zeros((len(self.codes)))
        self.knn.fit(self.codes, y)

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training example as np
        '''
        # TODO - force used neighbors out of codebook
        neighbor_distances, neighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize)
        else:
            chosen_neighbor_index = np.zeros((bsize), dtype=np.int)
        return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]], neighbor_indexes

    def forward(self, codes):
        st = time.time()
        DEVICE = codes.device
        np_codes = codes.cpu().detach().numpy()
        previous_codes, neighbor_indexes = self.batch_pick_close_neighbor(np_codes)
        previous_codes = torch.FloatTensor(previous_codes).to(DEVICE)
        mixtures, mus, sigmas =  self.encode(previous_codes)
        # output should be of shape (num_k, code_len) embed()
        mus = mus.view(mus.shape[0], self.n_mixtures, self.code_length)
        sigmas = sigmas.view(sigmas.shape[0], self.n_mixtures, self.code_length)
        return mixtures, mus, sigmas


