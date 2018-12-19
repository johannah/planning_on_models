"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

Strongly referenced ACN implementation and blog post from:
http://jalexvig.github.io/blog/associative-compression-networks/

Base VAE referenced from pytorch examples:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_losses
torch.manual_seed(394)

class VAE(nn.Module):
    def __init__(self, code_len, input_size=28*28, h=512):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(input_size, h)
        self.fc21 = nn.Linear(h, code_len)
        self.fc22 = nn.Linear(h, code_len)
        self.fc3 = nn.Linear(code_len,h)
        self.fc4 = nn.Linear(h, input_size)

    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(y_hat, y, mu, logvar):
    BCE = F.binary_cross_entropy(y_hat, y.view(-1, 784), reduction='sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

def acn_loss_function(y_hat, y, u_q, s_q, u_p, s_p):
    ''' reconstruction loss + coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     Args:
         y_hat: reconstruction output
         y: target
         u_q: mean of model posterior
         s_q: log std of model posterior
         u_p: mean of conditional prior
         s_p: log std of conditional prior

     Returns: loss
     '''
    BCE = F.binary_cross_entropy(y_hat, y, reduction='sum')
    acn_KLD = torch.sum(s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    return BCE+acn_KLD





class PriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super().__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2_u = nn.Linear(n_hidden, self.code_length)
        self.fc2_s = nn.Linear(n_hidden, self.code_length)

        # TODO why 2*k
        self.knn = KNeighborsClassifier(n_neighbors=2*self.k, n_jobs=-1)
        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
        codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
        self.fit_knn(codes)

    def fit_knn(self, codes):
        ''' will reset the knn  given an nd array
        '''
        self.codes = codes
        self.seen = set()
        y = np.zeros((self.size_training_set))
        self.knn.fit(self.codes, y)

    def pick_close_neighbor(self, code):
        '''
        :code latent activation of training example as np
        '''
        # returns neighbors, n_neighbors
        neighbor_indexes = self.knn.kneighbors([code], return_distance=False)[0]
        # get index for neighbors
        valid_indexes = [n for n in neighbor_indexes if n not in self.seen]
        if len(valid_indexes) < self.k:
            codes_new = [c for i, c in enumerate(self.codes) if i not in self.seen]
            self.fit_knn(codes_new)
            return self.pick_close_neighbors(code)
        neighbor_codes = [self.codes[idx] for idx in valid_indexes]
        # TODO < > seems like there is one missing
        if len(neighbor_codes) > self.k:
            # TODO - use distances from knn neighbors
            neighbor_codes = neighbor_codes = sorted(neighbor_codes, key=lambda n: ((code - n) ** 2).sum())[:self.k]
        rand_neighbor_index = self.rdn.randint(len(neighbor_codes))
        return neighbor_codes[rand_neighbor_index]

    def forward(self, codes):
        previous_codes = [self.pick_close_neighbor(c.detach().numpy()) for c in codes]
        previous_codes = torch.FloatTensor(previous_codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        logstd = self.fc2_s(h1)
        return mu, logstd



def train_acn(train_cnt):
    vae_model.train()
    prior_model.train()
    train_loss = 0
    new_codes = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.shape[0], -1).to(DEVICE)
        print(data.shape)
        opt.zero_grad()
        yhat_batch, u_q, s_q = vae_model(data)
        new_codes.append(u_q)
        u_p, s_p = prior_model(u_q)
        loss = acn_loss_function(yhat_batch, data, u_q, s_q, u_p, s_p)
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        def handle_plot_ckpt(do_plot=False):
            train_loss = loss.data.numpy().mean()/len(data)
            info['train_losses'].append(train_loss)
            info['train_cnts'].append(train_cnt)
            test_loss = test_acn(train_cnt)
            info['test_losses'].append(test_loss)
            info['test_cnts'].append(train_cnt)
            print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                                      info['train_losses'][-1], info['test_losses'][-1]))
            rolling = 3
            # plot
            if do_plot and  len(info['train_losses'])>rolling*3:
                info['last_plot'] = train_cnt
                plot_name = vae_base_filepath + "_%010dloss.png"%train_cnt
                print('plotting: %s with %s points'%(plot_name, len(info['train_cnts'])))
                plot_losses(info['train_cnts'],
                            info['train_losses'],
                            info['test_cnts'],
                            info['test_losses'], name=plot_name, rolling_length=rolling)

        if ((train_cnt-info['last_save'])>=args.save_every):
            info['last_save'] = train_cnt
            info['save_times'].append(time.time())
            handle_plot_ckpt(True)
            filename = vae_base_filepath + "_%010dex.pkl"%train_cnt
            state = {
                     'state_dict':vae_model.state_dict(),
                     'optimizer':opt.state_dict(),
                     'info':info,
                     }
            save_checkpoint(state, filename=filename)
        elif not train_cnt or (train_cnt-info['train_cnts'][-1])>=args.log_every:
            handle_plot_ckpt(False)
        else:
            if (train_cnt-info['last_plot'])>=args.plot_every:
                handle_plot_ckpt(True)

        train_cnt+=len(data)
    new_codes = torch.cat(new_codes).detach().cpu().data
    prior.fit_knn(new_codes)
    return train_cnt

def test_acn(train_cnt):
    vae_model.eval()
    prior_model.eval()
    test_loss = 0
    new_codes = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(data.shape[0], -1).to(DEVICE)
            yhat_batch, u_q, s_q = vae_model(data)
            new_codes.append(u_q)
            u_p, s_p = prior_model(u_q)
            loss = acn_loss_function(yhat_batch, data, u_q, s_q, u_p, s_p)
            test_loss+= loss.item()
            opt.step()

            #test_loss += vae_loss_function(yhat, data, mu, logvar).item()
            #if i == 0:
            #    n = min(data.size(0), 8)
            #    comparison = torch.cat([data[:n],
            #                          yhat.view(args.batch_size, 1, 28, 28)[:n]])
            #img_name = vae_base_filepath + "_%010dvalid_sample.png"%train_cnt
            #save_image(comparison.cpu(), img_name, nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def train_vae(train_cnt):
    vae_model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        opt.zero_grad()
        yhat_batch, mu, logvar = vae_model(data)
        loss = vae_loss_function(yhat_batch, data, mu, logvar)
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        def handle_plot_ckpt(do_plot=False):
            train_loss = loss.data.numpy().mean()/len(data)
            info['train_losses'].append(train_loss)
            info['train_cnts'].append(train_cnt)
            test_loss = test_vae(train_cnt)
            info['test_losses'].append(test_loss)
            info['test_cnts'].append(train_cnt)
            print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                                      info['train_losses'][-1], info['test_losses'][-1]))
            rolling = 3
            # plot
            if do_plot and  len(info['train_losses'])>rolling*3:
                info['last_plot'] = train_cnt
                plot_name = vae_base_filepath + "_%010dloss.png"%train_cnt
                print('plotting: %s with %s points'%(plot_name, len(info['train_cnts'])))
                plot_losses(info['train_cnts'],
                            info['train_losses'],
                            info['test_cnts'],
                            info['test_losses'], name=plot_name, rolling_length=rolling)

        if ((train_cnt-info['last_save'])>=args.save_every):
            info['last_save'] = train_cnt
            info['save_times'].append(time.time())
            handle_plot_ckpt(True)
            filename = vae_base_filepath + "_%010dex.pkl"%train_cnt
            state = {
                     'state_dict':vae_model.state_dict(),
                     'optimizer':vae_opt.state_dict(),
                     'info':info,
                     }
            save_checkpoint(state, filename=filename)
        elif not train_cnt or (train_cnt-info['train_cnts'][-1])>=args.log_every:
            handle_plot_ckpt(False)
        else:
            if (train_cnt-info['last_plot'])>=args.plot_every:
                handle_plot_ckpt(True)

        train_cnt+=len(data)
    return train_cnt

def test_vae(train_cnt):
    vae_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            yhat, mu, logvar = vae_model(data)
            test_loss += vae_loss_function(yhat, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      yhat.view(args.batch_size, 1, 28, 28)[:n]])
            img_name = vae_base_filepath + "_%010dvalid_sample.png"%train_cnt
            save_image(comparison.cpu(), img_name, nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=60000, type=int)
    parser.add_argument('-pe', '--plot_every', default=10000, type=int)
    parser.add_argument('-le', '--log_every', default=10000, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-nc', '--number_condition', default=4, type=int)
    parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=20, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=5000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'gpu'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(config.model_savedir, 'vae')
    prior_base_filepath = os.path.join(config.model_savedir, 'prior')

    train_data = datasets.MNIST(config.base_datadir, train=True, download=True,
                          transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = datasets.MNIST(config.base_datadir, train=False, download=True,
                          transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    size_training_set = len(train_data)

    vae_model = VAE(args.code_length, h=512, input_size=28*28)
    prior_model = PriorNetwork(size_training_set=size_training_set, code_length=args.code_length, k=args.num_k)
    parameters = list(vae_model.parameters()) + list(prior_model.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0
    while train_cnt < args.num_examples_to_train:
        train_cnt = train_acn(train_cnt)

