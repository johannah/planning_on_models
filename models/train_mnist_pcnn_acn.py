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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
#from acn import ConvVAE, PriorNetwork
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_losses
from make_tsne_plot import tsne_plot
from pixel_cnn import GatedPixelCNN
torch.manual_seed(394)

"""
\cite{acn} The ACN encoder was a convolutional
network fashioned after a VGG-style classifier (Simonyan
& Zisserman, 2014), and the encoding distribution q(z|x)
was a unit variance Gaussian with mean specified by the
output of the encoder network.
size of z is 16 for mnist, 128 for others
"""
class ConvVAE(nn.Module):
    def __init__(self, code_len, input_size=1, encoder_output_size=1000):
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
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        # found via experimentation - 3 for mnist
        # input_image == 28 -> eo=7
        self.eo = eo = encoder_output_size
        self.fc21 = nn.Linear(eo, code_len)
        self.fc22 = nn.Linear(eo, code_len)

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

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
        return z, mu, logvar

def kl_loss_function(u_q, s_q, u_p, s_p):
    ''' reconstruction loss + coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     Args:
         u_q: mean of model posterior
         s_q: log std of model posterior
         u_p: mean of conditional prior
         s_p: log std of conditional prior

     Returns: loss
     '''
    acn_KLD = torch.sum(s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    return acn_KLD


"""
/cite{acn} The prior network was an
MLP with three hidden layers each containing 512 tanh
units, and skip connections from the input to all hidden
layers and all hiddens to the output layer
"""
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
        assert(len(self.codes)>1)
        y = np.zeros((len(self.codes)))
        self.knn.fit(self.codes, y)

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
        np_codes = codes.cpu().detach().numpy()
        previous_codes = self.batch_pick_close_neighbor(np_codes)
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
        filename = vae_base_filepath + "_%010dex.pt"%train_cnt
        state = {
                 'vae_state_dict':vae_model.state_dict(),
                 'prior_state_dict':prior_model.state_dict(),
                 'pcnn_state_dict':pcnn_decoder.state_dict(),
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
    for batch_idx, (data, label, data_index) in enumerate(train_loader):
        lst = time.time()
        #for xx,i in enumerate(label):
        #    label_size[xx] = i
        target =data = data.to(DEVICE)
        opt.zero_grad()
        z, u_q, s_q = vae_model(data)
        #yhat_batch = vae_model.decode(u_q, s_q, data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=data, float_condition=z))
        prior_model.codes[data_index] = u_q.detach().cpu().numpy()
        prior_model.fit_knn(prior_model.codes)
        u_p, s_p = prior_model(u_q)
        kl = kl_loss_function(u_q, s_q, u_p, s_p)
        rec_loss = F.binary_cross_entropy(yhat_batch, target, reduction='mean')
        loss = kl+rec_loss
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
    print(len(valid_loader))
    with torch.no_grad():
        for i, (data, label, data_index) in enumerate(valid_loader):
            target = data = data.to(DEVICE)
            z, u_q, s_q = vae_model(data)
            #yhat_batch = vae_model.decode(u_q, s_q, data)
            # add the predicted codes to the input
            yhat_batch = torch.sigmoid(pcnn_decoder(x=data, float_condition=z))
            u_p, s_p = prior_model(u_q)
            kl = kl_loss_function(u_q, s_q, u_p, s_p)
            rec_loss = F.binary_cross_entropy(yhat_batch, target, reduction='mean')
            loss = kl+rec_loss
            test_loss+= loss.item()
            if i == 0 and do_plot:
                print('writing img')
                n = min(data.size(0), 8)
                bs = data.shape[0]
                comparison = torch.cat([target.view(bs, 1, 28, 28)[:n],
                                      yhat_batch.view(bs, 1, 28, 28)[:n]])
                img_name = vae_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
                save_image(comparison.cpu(), img_name, nrow=n)
                print('finished writing img', img_name)
            #print('loop test', i, time.time()-lst)

    test_loss /= len(valid_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('finished test', time.time()-st)
    return test_loss


def call_tsne_plot():
    vae_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for batch_idx, (data, label, data_index) in enumerate(data_loader):
                    target = data = data.to(DEVICE)
                    # yhat_batch is bt 0-1
                    z, u_q, s_q = vae_model(data)
                    u_p, s_p = prior_model(u_q)
                    yhat_batch = torch.sigmoid(pcnn_decoder(x=target, float_condition=z))
                    end = args.model_loadpath.split('.')[-1]
                    html_path = args.model_loadpath.replace(end, 'html')
                    X = u_q.cpu().numpy()
                    images = np.round(yhat_batch.cpu().numpy()[:,0], 0).astype(np.int16)
                    #images = next_state[:,0].cpu().numpy()
                    tsne_plot(X=X, images=images, color=batch_idx, perplexity=args.perplexity,
                              html_out_path=html_path)

def sample():
    print('starting sample', train_cnt)
    from skvideo.io import vwrite
    vae_model.eval()
    pcnn_decoder.eval()
    output_savepath = args.model_loadpath.replace('.pt', '')
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for batch_idx, (data, label, data_index) in enumerate(data_loader):
                    data = data.to(DEVICE)
                    bs = data.shape[0]
                    z, u_q, s_q = vae_model(data)
                    # teacher forced version
                    yhat_batch = torch.sigmoid(pcnn_decoder(x=data, float_condition=z))
                    u_p, s_p = prior_model(u_q)
                    canvas = torch.zeros_like(data)
                    building_canvas = []
                    for i in range(canvas.shape[1]):
                        for j in range(canvas.shape[2]):
                            print('sampling row: %s'%j)
                            for k in range(canvas.shape[3]):
                                output = torch.sigmoid(pcnn_decoder(x=canvas, float_condition=z))
                                canvas[:,i,j,k] = output[:,i,j,k].detach()
                                if not k%2:
                                    building_canvas.append(deepcopy(canvas[0].detach().cpu().numpy()))
                    f,ax = plt.subplots(bs, 3, sharex=True, sharey=True, figsize=(2,2*bs))
                    npdata = data.detach().cpu().numpy()
                    npoutput = output.detach().cpu().numpy()
                    npyhat = yhat_batch.detach().cpu().numpy()
                    for idx in range(bs):
                        ax[idx,0].imshow(npdata[idx,0], cmap=plt.cm.gray)
                        ax[idx,0].set_title('true')
                        ax[idx,1].imshow(npyhat[idx,0], cmap=plt.cm.gray)
                        ax[idx,1].set_title('tf')
                        ax[idx,2].imshow(npoutput[idx,0], cmap=plt.cm.gray)
                        ax[idx,2].set_title('sam')
                        ax[idx,0].axis('off')
                        ax[idx,1].axis('off')
                        ax[idx,2].axis('off')
                    iname = output_savepath + '_sample_%s.png'%phase
                    print('plotting %s'%iname)
                    plt.savefig(iname)
                    plt.close()

                    building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
                    print('writing building movie')
                    mname = output_savepath + '_build_%s.mp4'%phase
                    vwrite(mname, building_canvas)
                    print('finished %s'%mname)
                    break
        sys.exit()


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

def save_checkpoint(state, filename='model.pt'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-se', '--save_every', default=60000*10, type=int)
    parser.add_argument('-pe', '--plot_every', default=60000*10, type=int)
    parser.add_argument('--log_every', default=60000*10, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4)
    parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-nc', '--num_classes', default=10)
    parser.add_argument('-eos', '--encoder_output_size', default=2048)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=12)
    parser.add_argument( '--model_savedir', default='../../model_savedir', help='save checkpoints here')
    parser.add_argument( '--base_datadir', default='../../dataset/', help='save datasets here')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-t', '--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=3)
    parser.add_argument('--use_training_set', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-nt', '--num_tsne', default=300, type=int)

    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(args.model_savedir, 'pcnn_acn_mnist_cl64')
    train_data = IndexedDataset(datasets.MNIST, path=args.base_datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data = IndexedDataset(datasets.MNIST, path=args.base_datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    data_dict = {'train':train_loader, 'valid':valid_loader}

    nchans,hsize,wsize = train_loader.dataset[0][0].shape

    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    args.size_training_set = len(train_data)

    vae_model = ConvVAE(args.code_length, input_size=1, encoder_output_size=args.encoder_output_size).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=args.size_training_set,
                               code_length=args.code_length, k=args.num_k).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                      dim=args.possible_values,
                                      n_layers=args.num_pcnn_layers,
                                      n_classes=args.num_classes,
                                      float_condition_size=args.code_length,
                                      last_layer_bias=0.5).to(DEVICE)

    parameters = list(vae_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0

    if args.model_loadpath !='':
        tmlp =  args.model_loadpath+'.tmp'
        os.system('cp %s %s'%(args.model_loadpath, tmlp))
        _dict = torch.load(tmlp, map_location=lambda storage, loc:storage)
        vae_model.load_state_dict(_dict['vae_state_dict'])
        prior_model.load_state_dict(_dict['prior_state_dict'])
        pcnn_decoder.load_state_dict(_dict['pcnn_state_dict'])

        info = _dict['info']
        train_cnt = info['train_cnts'][-1]



    if args.sample:
        sample()
    if args.tsne:
        call_tsne_plot()


    while train_cnt < args.num_examples_to_train:
        train_cnt = train_acn(train_cnt)

