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
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_losses
torch.manual_seed(394)

class ConvVAE(nn.Module):
    def __init__(self, code_len, input_size=1):
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
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        # found via experimentation - 3 for mnist
        # input_image == 28 -> eo=7
        self.eo=eo=encode_output_size = 7
        self.fc21 = nn.Linear(code_len*2*eo*eo, code_len)
        self.fc22 = nn.Linear(code_len*2*eo*eo, code_len)
        self.fc3 = nn.Linear(code_len, code_len*2*eo*eo)
        self.decoder = nn.Sequential(
               nn.ConvTranspose2d(in_channels=code_len*2,
                      out_channels=32,
                      kernel_size=1,
                      stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=16,
                        out_channels=input_size,
                        kernel_size=4,
                        stride=2, padding=1),
                )

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

    def decode(self, mu, logvar):
        c = self.reparameterize(mu,logvar)
        co = F.relu(self.fc3(c))
        col = co.view(co.shape[0], self.code_len*2, self.eo, self.eo)
        do = torch.sigmoid(self.decoder(col))
        return do

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return  mu, logvar

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
        super(PriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2_u = nn.Linear(n_hidden, self.code_length)
        self.fc2_s = nn.Linear(n_hidden, self.code_length)

        self.knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
        codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
        self.fit_knn(codes)

    def fit_knn(self, codes):
        ''' will reset the knn  given an nd array
        '''
        st = time.time()
        self.codes = codes
        self.seen = set()
        assert(len(self.codes)>1)
        y = np.zeros((len(self.codes)))
        self.knn.fit(self.codes, y)
        #print("FIT KNN!", time.time()-st)


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
        return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]]

    def forward(self, codes):
        st = time.time()
        #print("starting forward")
        np_codes = codes.cpu().detach().numpy()
        #previous_codes = [self.pick_close_neighbor(c) for c in np_codes]
        previous_codes = self.batch_pick_close_neighbor(np_codes)
        #print('finished forward', time.time()-st)
        previous_codes = torch.FloatTensor(previous_codes).to(DEVICE)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        logstd = self.fc2_s(h1)
        return mu, logstd

def handle_plot_ckpt(do_plot, train_cnt, avg_train_loss):
    info['train_losses'].append(avg_train_loss)
    info['train_cnts'].append(train_cnt)
    test_loss = test_acn(train_cnt,do_plot)
    info['test_losses'].append(test_loss)
    info['test_cnts'].append(train_cnt)
    print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                              info['train_losses'][-1], info['test_losses'][-1]))
    # plot
    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_losses'])<rolling*3:
            rolling = 1
        print('adding last loss plot', train_cnt)
        plot_name = vae_base_filepath + "_%010dloss.png"%train_cnt
        print('plotting loss: %s with %s points'%(plot_name, len(info['train_cnts'])))
        plot_losses(info['train_cnts'],
                    info['train_losses'],
                    info['test_cnts'],
                    info['test_losses'], name=plot_name, rolling_length=rolling)

def handle_checkpointing(train_cnt, avg_train_loss):
    if ((train_cnt-info['last_save'])>=args.save_every):
        print("Saving Model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['last_save']))
        info['last_save'] = train_cnt
        info['save_times'].append(time.time())
        handle_plot_ckpt(True, train_cnt, avg_train_loss)
        filename = vae_base_filepath + "_%010dex.pkl"%train_cnt
        state = {
                 'vae_state_dict':vae_model.state_dict(),
                 'prior_state_dict':prior_model.state_dict(),
                 'optimizer':opt.state_dict(),
                 'info':info,
                 }
        save_checkpoint(state, filename=filename)
    elif not len(info['train_cnts']):
        print("Logging model: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(False, train_cnt, avg_train_loss)
    elif (train_cnt-info['last_plot'])>=args.plot_every:
        print("Plotting Model at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, avg_train_loss)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every:
            print("Logging Model at cnt:%s cnt since last logged:%s"%(train_cnt, train_cnt-info['train_cnts'][-1]))
            handle_plot_ckpt(False, train_cnt, avg_train_loss)

def train_acn(train_cnt):
    vae_model.train()
    prior_model.train()
    train_loss = 0
    init_cnt = train_cnt
    st = time.time()
    for batch_idx, (data, _, data_index) in enumerate(train_loader):
        lst = time.time()
        data = data.to(DEVICE)
        opt.zero_grad()
        u_q, s_q = vae_model(data)
        prior_model.codes[data_index] = u_q.detach().cpu().numpy()
        prior_model.fit_knn(prior_model.codes)
        u_p, s_p = prior_model(u_q)
        yhat_batch = vae_model.decode(u_p,s_p)
        loss = acn_loss_function(yhat_batch, data, u_q, s_q, u_p, s_p)
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_loss = train_loss/float((train_cnt+data.shape[0])-init_cnt)
        handle_checkpointing(train_cnt, avg_train_loss)
        train_cnt+=len(data)
    print("finished epoch after %s seconds at cnt %s"%(time.time()-st, train_cnt))
    return train_cnt

def test_acn(train_cnt, do_plot):
    vae_model.eval()
    prior_model.eval()
    test_loss = 0
    print('starting test', train_cnt)
    st = time.time()
    print(len(test_loader))
    with torch.no_grad():
        for i, (data, _, data_index) in enumerate(test_loader):
            lst = time.time()
            #data = data.view(data.shape[0], -1).to(DEVICE)
            data = data.to(DEVICE)
            u_q, s_q = vae_model(data)
            u_p, s_p = prior_model(u_q)
            yhat_batch = vae_model.decode(u_p,s_p)
            loss = acn_loss_function(yhat_batch, data, u_q, s_q, u_p, s_p)
            test_loss+= loss.item()
            if i == 0 and do_plot:
                print('writing img')
                n = min(data.size(0), 8)
                bs = data.shape[0]
                comparison = torch.cat([data.view(bs, 1, 28, 28)[:n],
                                      yhat_batch.view(bs, 1, 28, 28)[:n]])
                img_name = vae_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
                save_image(comparison.cpu(), img_name, nrow=n)
                print('finished writing img', img_name)
            #print('loop test', i, time.time()-lst)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('finished test', time.time()-st)
    return test_loss

class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=60000*10, type=int)
    parser.add_argument('-pe', '--plot_every', default=200000, type=int)
    parser.add_argument('-le', '--log_every', default=200000, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=20, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=5000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(config.model_savedir, 'cacn')
    train_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


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

    vae_model = ConvVAE(args.code_length, input_size=1).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=size_training_set, code_length=args.code_length, k=args.num_k).to(DEVICE)
    parameters = list(vae_model.parameters()) + list(prior_model.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0
    while train_cnt < args.num_examples_to_train:
        train_cnt = train_acn(train_cnt)

