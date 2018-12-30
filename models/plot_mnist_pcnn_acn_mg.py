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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import IndexedDataset
import config
from torchvision.utils import save_image
from IPython import embed
from pixel_cnn import GatedPixelCNN
from imageio import imsave
from acn_mg import  PriorNetwork, ConvVAE, acn_loss_function
torch.manual_seed(394)
torch.set_num_threads(1)
from sklearn.decomposition import PCA

def plot_neighbors(data, label, data_idx, nearby_indexes, name):
    for i in range(data.shape[0]):
         f,ax=plt.subplots(1,nearby_indexes.shape[1]+1)
         ax[0].imshow(data[i,-1].detach().numpy())
         ax[0].set_title('true')
         ax[0].axis('off')
         #ax[1,0].axis('off')
         for xx in range(nearby_indexes.shape[1]):
             nearby_x, nearby_y, nearby_index = train_data[nearby_indexes[i,xx]]
             ax[xx+1].imshow(nearby_x[0].detach().numpy())
             ax[xx+1].set_title('%s-%s'%(int(nearby_y), nearby_index))
             ax[xx+1].axis('off')

         ipath = os.path.join(output_savepath, 'nn_%s_%04d.png'%(name,data_idx[i]))
         print('plotting %s' %ipath)
         plt.savefig(ipath)
         plt.close()


def nearest_neighbor_batch():
    seen = 0
    print("getting training neighbors")
    for data, label, data_idx in train_loader:
        z, u_q = encoder_model(data)
        prior_model.codes[data_idx] = u_q.detach().cpu().numpy()
        seen += z.shape[0]
    print("seen", seen)
    prior_model.fit_knn(prior_model.codes)
    # use last
    nearby_codes, nearby_indexes = prior_model.batch_pick_close_neighbor(u_q.cpu().detach().numpy())
    plot_neighbors(data, label, data_idx, nearby_indexes, name='train')
    for test_data, test_label, test_data_idx in test_loader:
        test_z, test_u_q = encoder_model(test_data)
        break
    test_nearby_codes, test_nearby_indexes = prior_model.batch_pick_close_neighbor(test_u_q.cpu().detach().numpy())
    plot_neighbors(test_data, test_label, test_data_idx, test_nearby_indexes, name='test')

def pca_batch(loader, name):
    total = 0
    for data, label, idx in loader:
        z, u_q = encoder_model(data)
        if not total:
            X = z.detach().numpy()
            y = label.detach().numpy()
        else:
            X = np.vstack((X, z.detach().numpy()))
            y = np.hstack((y, label.detach().numpy()))
        total+=data.shape[0]
        if total > args.max_plot:
            break
        print('total', total)

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    plt.figure()
    for i in range(10):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                    alpha=.8,
                    label='%s'%i)
    plt.legend()
    ipath = os.path.join(output_savepath, 'pca_%s.png'%name)
    print('plotting', ipath)
    plt.savefig(ipath)
    plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='sample acn pcnn for freeway')
    parser.add_argument('-l', '--model_loadname', required=True, help='filename of pkl file to load models from')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-nn', '--nearest_neighbor_only', action='store_true', default=False)
    parser.add_argument('-pca', '--pca_only', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-n', '--num_to_sample', default=15, type=int)
    parser.add_argument('-mp', '--max_plot', default=500, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-da', '--data_augmented', default=False, action='store_true')
    parser.add_argument('-daf', '--data_augmented_by_model', default="None")
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    model_loadpath = os.path.abspath(os.path.join(config.model_savedir, args.model_loadname))
    if not os.path.exists(model_loadpath):
        print("Error: given model load path does not exist")
        print(model_loadpath)
        sys.exit()

    output_savepath = model_loadpath.replace('.pkl', '')
    if not os.path.exists(output_savepath):
        os.makedirs(output_savepath)
    model_dict = torch.load(model_loadpath, map_location=lambda storage, loc: storage)
    info = model_dict['info']
    largs = info['args'][-1]

    train_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    nchans,hsize,wsize = test_loader.dataset[0][0].shape

    largs.size_training_set = len(train_data)
    info = model_dict['info']
    largs = info['args'][-1]

    #try:
    #    print(largs.encoder_output_size)
    #except:
    #    largs.encoder_output_size = 1960
    #try:
    #    print(largs.possible_values)
    #except:
    #    largs.possible_values = 1
    #try:
    #    print(largs.num_pcnn_layers)
    #except:
    #    largs.num_pcnn_layers = 12
    #try:
    #    print(largs.num_classes)
    #except:
    #    largs.num_classes = 10


    size_training_set = len(train_data)

    encoder_model = ConvVAE(largs.code_length,
                            input_size=1,
                            encoder_output_size=largs.encoder_output_size)
    encoder_model.load_state_dict(model_dict['vae_state_dict'])


    prior_model = PriorNetwork(size_training_set=largs.size_training_set,
                                code_length=largs.code_length, DEVICE=DEVICE,
                                k=largs.num_k).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=largs.possible_values,
                                 n_layers=largs.num_pcnn_layers,
                                 n_classes=largs.num_classes,
                                 float_condition_size=largs.code_length,
                                 last_layer_bias=0.5, hsize=hsize, wsize=wsize)

    pcnn_decoder.load_state_dict(model_dict['pcnn_state_dict'])

    encoder_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()

    if not args.nearest_neighbor_only:
        print("finding pca")
        pca_batch(test_loader, 'test')
        pca_batch(train_loader, 'train')

    if not args.pca_only:
        print("finding neighbors")
        nearest_neighbor_batch()


