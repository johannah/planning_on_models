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
softplus_fn = torch.nn.Softplus()
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
    BCE = F.binary_cross_entropy(y_hat, y, reduction='sum')

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
    def __init__(self, size_training_set, code_length, DEVICE, n_hidden=512,
                 k=5, random_seed=4543, require_unique_codes=False):
        super(PriorNetwork, self).__init__()
        self.require_unique_codes = require_unique_codes
        if self.require_unique_codes:
            self.pick_neighbor = self.batch_pick_unique_close_neighbor
        else:
            self.pick_neighbor = self.batch_pick_close_neighbor
        self.DEVICE = DEVICE
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2_u = nn.Linear(n_hidden, self.code_length)
        self.fc2_s = nn.Linear(n_hidden, self.code_length)

        self.knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
        self.codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
        self.fit_knn(self.codes)
        self.new_epoch()

    def fit_knn(self, codes):
        ''' will reset the knn  given an nd array
        '''
        st = time.time()
        #self.codes = codes
        #assert(len(self.codes)>1)
        y = np.zeros((len(codes)))
        self.knn.fit(codes, y)

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training example as np
        '''
        neighbor_distances, neighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize)
        else:
            chosen_neighbor_index = np.zeros((bsize), dtype=np.int)
        return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]], neighbor_indexes


    def new_epoch(self):
        self.code_used = np.zeros((self.codes.shape[0]), dtype=np.bool)
        self.available_indexes = np.arange(self.codes.shape[0], dtype=np.int)

    def add_used(self, used_code_index):
        self.code_used[used_code_index] = True
        w = np.where(self.available_indexes != used_code_index)[0]
        self.available_indexes = self.available_indexes[w]

    def batch_pick_unique_close_neighbor(self, codes):
        '''
        :code latent activation of training example as np
        '''

        if self.training:
            self.fit_knn(self.codes[self.available_indexes])
            uneighbor_distances, uneighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
            used_code_indexes = []
            chosen_codes, chosen_code_indexes, all_neighbor_indexes = [], [], []
            for bi, code in enumerate(codes):
                # randomly choose unique neighbor index from top k
                uneighbor_chosen = self.rdn.choice(uneighbor_indexes[bi])
                # need to take the unique index and figure out which real index it is
                chosen_code_index = self.available_indexes[uneighbor_chosen]
                #print(uneighbor_chosen, chosen_code_index)
                # code has already been used
                if chosen_code_index in used_code_indexes:
                    # add in the code - then remove them from possibilities and
                    # retrain knn
                    for cc in used_code_indexes:
                        self.add_used(cc)
                    print(chosen_code_index, "retrained knn. used-", len(used_code_indexes), 'fit shape', self.available_indexes.shape[0])
                    #print(used_code_indexes)
                    # redo all of our input codes for simplicity
                    if self.available_indexes.shape[0]<=self.k:
                        print('----------------------------------------------')
                        print('new epoch')
                        self.new_epoch()
                    self.fit_knn(self.codes[self.available_indexes])
                    uneighbor_distances, uneighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
                    used_code_indexes = []
                    # redo this code with the new knn
                    uneighbor_chosen = self.rdn.choice(uneighbor_indexes[bi])
                    # need to take the unique index and figure out which real index it is
                    chosen_code_index = self.available_indexes[uneighbor_chosen]
                    #print('reset indexes',self.available_indexes.shape)
                    #print('new',uneighbor_chosen, chosen_code_index)

                used_code_indexes.append(chosen_code_index)
                chosen_code = self.codes[chosen_code_index]
                chosen_codes.append(chosen_code)
                chosen_code_indexes.append(chosen_code_index)
                all_neighbor_indexes.append(self.available_indexes[uneighbor_indexes[bi]])
            for cc in used_code_indexes:
                self.add_used(cc)
            return np.array(chosen_codes), np.array(all_neighbor_indexes)
        else:
            neighbor_distances, neighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
            bsize = neighbor_indexes.shape[0]
            chosen_neighbor_index = np.zeros((bsize), dtype=np.int)
            return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]], neighbor_indexes




    def forward(self, codes):
        st = time.time()
        np_codes = codes.cpu().detach().numpy()
        previous_codes, neighbor_indexes = self.pick_neighbor(np_codes)
        previous_codes = torch.FloatTensor(previous_codes).to(self.DEVICE)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        std = self.fc2_s(h1)
        return mu, softplus_fn(std)+1e-4


