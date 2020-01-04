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
from copy import deepcopy, copy
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from IPython import embed
from pixel_cnn import GatedPixelCNN

class ConvEncoder(nn.Module):
    def __init__(self, code_len, input_size=1, encoder_output_size=1000):
        super(ConvEncoder, self).__init__()
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
        self.codes = codes
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
        device = codes.device
        np_codes = codes.cpu().detach().numpy()
        previous_codes = self.batch_pick_close_neighbor(np_codes)
        previous_codes = torch.FloatTensor(previous_codes).to(device)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        logstd = self.fc2_s(h1)
        return mu, logstd

def set_model_mode(model_dict, phase):
    for name, model in model_dict.items():
        print('setting', name, phase)
        if name != 'opt':
            if phase == 'valid':
                model_dict[name].eval()
            else:
                model_dict[name].train()
    return model_dict

def run_acn(train_cnt, model_dict, data_dict, phase, device):
    st = time.time()
    running_loss = 0.0
    data_loader = data_dict[phase]
    model_dict = set_model_mode(model_dict, phase)
    for batch_idx, (data, label, data_index) in enumerate(data_loader):
        target = data = data.to(device)
        bs = data.shape[0]
        model_dict['opt'].zero_grad()
        z, u_q, s_q = model_dict['encoder_model'](data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=data, float_condition=z))
        model_dict['prior_model'].codes[data_index] = u_q.detach().cpu().numpy()
        model_dict['prior_model'].fit_knn(model_dict['prior_model'].codes)
        u_p, s_p = model_dict['prior_model'](u_q)
        kl = kl_loss_function(u_q, s_q, u_p, s_p)
        rec_loss = F.binary_cross_entropy(yhat_batch, target, reduction='mean')
        loss = kl+rec_loss
        if phase == 'train':
            loss.backward()
            model_dict['opt'].step()
        running_loss+= loss.item()
        # add batch size because it hasn't been added to train cnt yet
        if phase == 'train':
            train_cnt+=bs
    example = {'data':data, 'target':target, 'yhat':yhat_batch}
    avg_loss = running_loss/bs
    print("finished %s after %s secs at cnt %s loss %s"%(phase, time.time()-st, train_cnt, avg_loss))
    return model_dict, data_dict, avg_loss, example

def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info):
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))
        model_dict, data_dict, avg_train_loss, train_example = run_acn(train_cnt, model_dict, data_dict, phase='train', device=info['device'])
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs']:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, avg_valid_loss, valid_example = run_acn(train_cnt, model_dict, data_dict, phase='valid', device=info['device'])
            state_dict = {}
            for key, model in model_dict.items():
                state_dict[key+'_state_dict'] = model.state_dict()
            info['train_losses'].append(avg_train_loss)
            info['train_cnts'].append(train_cnt)
            info['valid_losses'].append(avg_valid_loss)
            info['epoch_cnt'] = epoch_cnt
            state_dict['info'] = info

            ckpt_filepath = os.path.join(base_filepath, "%ss_%010dex.pt"%(base_filename, train_cnt))
            train_img_filepath = os.path.join(base_filepath,"%s_%010d_train_rec.png"%(base_filename, train_cnt))
            valid_img_filepath = os.path.join(base_filepath, "%s_%010d_valid_rec.png"%(base_filename, train_cnt))
            plot_filepath = os.path.join(base_filepath, "%s_%010dloss.png"%(base_filename, train_cnt))

            plot_example(train_img_filepath, train_example, num_plot=5)
            plot_example(valid_img_filepath, valid_example, num_plot=5)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['train_cnts'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)

def call_tsne_plot(model_dict, data_dict, info):
    from make_tsne_plot import tsne_plot
    # always be in eval mode
    model_dict = set_model_mode(model_dict, 'valid')
    with torch.no_grad():
        for phase in ['valid', 'train']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for batch_idx, (data, label, data_index) in enumerate(data_loader):
                    target = data = data.to(info['device'])
                    # yhat_batch is bt 0-1
                    z, u_q, s_q = model_dict['encoder_model'](data)
                    u_p, s_p = model_dict['prior_model'](u_q)
                    yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                    html_path = info['model_loadpath'].replace('.pt', '.html')
                    X = u_q.cpu().numpy()
                    images = np.round(yhat_batch.cpu().numpy()[:,0], 0).astype(np.int32)
                    #images = target[:,0].cpu().numpy()
                    tsne_plot(X=X, images=images, color=batch_idx, perplexity=info['perplexity'],
                              html_out_path=html_path)

def sample(model_dict, data_dict, info):
    from skvideo.io import vwrite
    model_dict = set_model_mode(model_dict, 'valid')
    output_savepath = args.model_loadpath.replace('.pt', '')
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for batch_idx, (data, label, data_index) in enumerate(data_loader):
                    target = data = data.to(info['device'])
                    bs = data.shape[0]
                    z, u_q, s_q = model_dict['encoder_model'](data)
                    # teacher forced version
                    yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                    # create blank canvas for autoregressive sampling
                    canvas = torch.zeros_like(target)
                    building_canvas = []
                    for i in range(canvas.shape[1]):
                        for j in range(canvas.shape[2]):
                            print('sampling row: %s'%j)
                            for k in range(canvas.shape[3]):
                                output = torch.sigmoid(model_dict['pcnn_decoder'](x=canvas, float_condition=z))
                                canvas[:,i,j,k] = output[:,i,j,k].detach()
                                # add frames for video
                                if not k%5:
                                    building_canvas.append(deepcopy(canvas[0].detach().cpu().numpy()))

                    f,ax = plt.subplots(bs, 3, sharex=True, sharey=True, figsize=(3,bs))
                    nptarget = target.detach().cpu().numpy()
                    npoutput = output.detach().cpu().numpy()
                    npyhat = yhat_batch.detach().cpu().numpy()
                    for idx in range(bs):
                        ax[idx,0].imshow(nptarget[idx,0], cmap=plt.cm.viridis)
                        ax[idx,0].set_title('true')
                        ax[idx,1].imshow(npyhat[idx,0], cmap=plt.cm.viridis)
                        ax[idx,1].set_title('tf')
                        ax[idx,2].imshow(npoutput[idx,0], cmap=plt.cm.viridis)
                        ax[idx,2].set_title('sam')
                        ax[idx,0].axis('off')
                        ax[idx,1].axis('off')
                        ax[idx,2].axis('off')
                    iname = output_savepath + '_sample_%s.png'%phase
                    print('plotting %s'%iname)
                    plt.savefig(iname)
                    plt.close()

                    # make movie
                    building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
                    print('writing building movie')
                    mname = output_savepath + '_build_%s.mp4'%phase
                    vwrite(mname, building_canvas)
                    print('finished %s'%mname)
                    # only do one batch
                    break

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

def create_mnist_datasets(dataset_name, base_datadir, batch_size):
    dataset = eval('datasets.'+dataset_name)
    datadir = os.path.join(base_datadir, dataset_name)
    train_data = IndexedDataset(dataset, path=datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = IndexedDataset(dataset, path=datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    data_dict = {'train':train_loader, 'valid':valid_loader}
    nchans,hsize,wsize = data_dict['train'].dataset[0][0].shape
    size_training_set = len(train_data)
    return data_dict, size_training_set, nchans, hsize, wsize

def create_new_info_dict(arg_dict, size_training_set, base_filepath):
    info = {'train_cnts':[],
            'train_losses':[],
            'valid_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
            'epoch_cnt':0,
            'size_training_set':size_training_set,
            'base_filepath':base_filepath,
             }
    for arg,val in arg_dict.items():
        info[arg] = val
    if info['cuda']:
        info['device'] = 'cuda'
    else:
        info['device'] = 'cpu'
    return info

def create_conv_acn_pcnn_models(info, model_loadpath=''):
    '''
    load details of previous model if applicable, otherwise create new models
    '''
    train_cnt = 0
    epoch_cnt = 0

    # use argparse device no matter what info dict is loaded
    preserve_args = ['device', 'batch_size', 'save_every_epochs', 'base_filepath']
    preserve_dict = {}
    for key in preserve_args:
        preserve_dict[key] = info[key]

    if args.model_loadpath !='':
        tmlp =  model_loadpath+'.tmp'
        os.system('cp %s %s'%(args.model_loadpath, tmlp))
        _dict = torch.load(tmlp, map_location=lambda storage, loc:storage)
        info = _dict['info']
        train_cnt = info['train_cnts'][-1]
        epoch_cnt = info['epoch_cnt']

    # use argparse device no matter what device is loaded
    for key in preserve_args:
        info[key] = preserve_dict[key]

    encoder_model = ConvEncoder(info['code_length'], input_size=info['input_channels'],
                            encoder_output_size=info['encoder_output_size']).to(info['device'])
    prior_model = PriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])
    pcnn_decoder = GatedPixelCNN(input_dim=info['target_channels'],
                                  dim=info['possible_values'],
                                  n_layers=info['num_pcnn_layers'],
                                  n_classes=info['num_classes'],
                                  float_condition_size=info['code_length'],
                                  last_layer_bias=info['last_layer_bias']).to(info['device'])

    model_dict = {'encoder_model':encoder_model, 'prior_model':prior_model, 'pcnn_decoder':pcnn_decoder}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, info, train_cnt, epoch_cnt

def seed_everything(seed=394, max_threads=2):
    torch.manual_seed(394)
    torch.set_num_threads(max_threads)

def plot_example(img_filepath, example, plot_on=['target', 'yhat'], num_plot=10):
    '''
    img_filepath: location to write .png file
    example: dict with torch images of the same shape [bs,c,h,w] to write
    plot_on: list of keys of images in example dict to write
    num_plot: limit the number of examples from bs to this int
    '''
    for cnt, pon in enumerate(plot_on):
        bs,c,h,w = example[pon].shape
        num_plot = min([bs, num_plot])
        eimgs = example[pon].view(bs,c,h,w)[:num_plot]
        if not cnt:
            comparison = eimgs
        else:
            comparison = torch.cat([comparison, eimgs])
    save_image(comparison.cpu(), img_filepath, nrow=num_plot)
    print('writing comparison image: %s img_path'%img_filepath)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_losses(train_cnts, train_losses, test_cnts, test_losses, name='loss_example.png', rolling_length=4):
    f,ax=plt.subplots(1,1,figsize=(3,3))
    ax.plot(rolling_average(train_cnts, rolling_length), rolling_average(train_losses, rolling_length), label='train loss', lw=1, c='orangered')
    ax.plot(rolling_average(test_cnts, rolling_length),  rolling_average(test_losses, rolling_length), label='test loss', lw=1, c='cornflowerblue')
    ax.scatter(rolling_average(test_cnts, rolling_length), rolling_average(test_losses, rolling_length), s=4, c='cornflowerblue')
    ax.scatter(rolling_average(train_cnts, rolling_length),rolling_average(train_losses, rolling_length), s=4, c='orangered')
    ax.legend()
    plt.savefig(name)
    plt.close()

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



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # operatation options
    parser.add_argument('-l', '--model_loadpath', default='', help='load model to resume training or sample')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=394)
    parser.add_argument('--num_threads', default=2)
    parser.add_argument('-se', '--save_every_epochs', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('--possible_values', default=1, help='num values that the pcnn output can take')
    parser.add_argument('--last_layer_bias', default=0.5, help='bias for output decoder')
    parser.add_argument('--num_classes', default=10, help='num classes for class condition in pixel cnn')
    parser.add_argument('--encoder_output_size', default=2048, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('--num_pcnn_layers', default=12, help='num layers for pixel cnn')
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='MNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--exp_name', default='pcnn_acn', help='name of experiment')
    parser.add_argument('--model_savedir', default='../../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../../dataset/', help='save datasets here')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    # tsne info
    parser.add_argument('-t', '--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=3)
    args = parser.parse_args()

    seed_everything(args.seed, args.num_threads)
    args.exp_name += '_'+args.dataset_name
    base_filepath = os.path.join(args.model_savedir, args.exp_name)
    if not os.path.exists(base_filepath):
        os.mkdir(base_filepath)

    data_dict, size_training_set, nchans, hsize, wsize = create_mnist_datasets(dataset_name=args.dataset_name, base_datadir=args.base_datadir, batch_size=args.batch_size)
    info = create_new_info_dict(vars(args), size_training_set, base_filepath)
    model_dict, info, train_cnt, epoch_cnt = create_conv_acn_pcnn_models(info, args.model_loadpath)

    if args.sample:
        sample(model_dict, data_dict, info)
    elif args.tsne:
        call_tsne_plot(model_dict, data_dict, info)
    else:
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info)

