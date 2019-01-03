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
from torchvision import datasets, transforms
from datasets import FreewayForwardDataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
from pixel_cnn import GatedPixelCNN
from imageio import imsave
from acn_mdn import ConvVAE, PriorNetwork, acn_mdn_loss_function
torch.manual_seed(394)
torch.set_num_threads(1)
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot_neighbors(data, label, data_idx, nearby_indexes, name):

    for i in range(data.shape[0]):
         f,ax=plt.subplots(2,nearby_indexes.shape[1]+1, figsize=(4*nearby_indexes.shape[1]+1,4))
         ax[0,0].imshow(data[i,-1].detach().numpy())
         ax[0,0].set_title('tx %s'%data_idx[i])
         ax[1,0].imshow(label[i,0].detach().numpy())
         ax[1,0].set_title('ty %s'%(data_idx[i]+int(largs.steps_ahead)))
         ax[0,0].axis('off')
         ax[1,0].axis('off')
         nearby_x, nearby_y = data_loader.train_loader[nearby_indexes[i]]
         for xx in range(nearby_x.shape[0]):
             ax[0,xx+1].imshow(nearby_x[xx,-1].detach().numpy())
             ax[0,xx+1].set_title('x %s'%(nearby_indexes[i,xx]))
             ax[1,xx+1].imshow(nearby_y[xx,0].detach().numpy())
             ax[1,xx+1].set_title('y %s'%(nearby_indexes[i,xx]+int(largs.steps_ahead)))
             ax[0,xx+1].axis('off')
             ax[1,xx+1].axis('off')

         ipath = os.path.join(output_savepath, 'nn_%s_%04d.png'%(name,data_idx[i]))
         print('plotting %s' %ipath)
         plt.savefig(ipath)
         plt.close()


def nearest_neighbor_batch():
    prior_model.fit_knn(prior_model.codes)

    data, label, data_idx = data_loader.next_batch()
    z, u_q = encoder_model(data)
    nearby_codes, nearby_indexes = prior_model.batch_pick_close_neighbor(u_q.cpu().detach().numpy())
    plot_neighbors(data, label, data_idx, nearby_indexes, name='train')

    test_data, test_label, test_data_idx = data_loader.validation_data()
    test_z, test_u_q = encoder_model(test_data)
    test_nearby_codes, test_nearby_indexes = prior_model.batch_pick_close_neighbor(test_u_q.cpu().detach().numpy())
    plot_neighbors(test_data, test_label, test_data_idx, test_nearby_indexes, name='test')


def pca_batch(loader, name):
    total = 0
    keep_going = True
    while keep_going:
        data, label, data_idx = loader()
        if not data.shape[0]:
            keep_going = False
        elif total > args.max_plot:
            keep_going = False
        else:
            z, u_q = encoder_model(data)
            if not total:
                X = z.detach().numpy()
            else:
                X = np.vstack((X, z.detach().numpy()))
            total+=data.shape[0]

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    plt.figure()
    plt.scatter(X_r[:,0], X_r[:,1], c=np.arange(X.shape[0]))
    plt.colorbar()
    ipath = os.path.join(output_savepath, 'pca_%s.png'%name)
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

    train_data_file = os.path.join(config.base_datadir, 'freeway_train_01000_40x40.npz')
    test_data_file = os.path.join(config.base_datadir, 'freeway_test_00300_40x40.npz')

    if args.data_augmented:
        # model that was used to create the forward predictions dataset
        aug_train_data_file = train_data_file[:-4] + args.data_augmented_by_model + '.npz'
        aug_test_data_file = test_data_file[:-4] + args.data_augmented_by_model + '.npz'
        for i in [aug_train_data_file, aug_test_data_file]:
            if not os.path.exists(i):
                print('augmented data file', i, 'does not exist')
                embed()
    else:
        aug_train_data_file = "None"
        aug_test_data_file = "None"


    train_data_function = FreewayForwardDataset(
                                   train_data_file,
                                   number_condition=largs.number_condition,
                                   steps_ahead=largs.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   augment_file=aug_train_data_file)
    test_data_function = FreewayForwardDataset(
                                   test_data_file,
                                   number_condition=largs.number_condition,
                                   steps_ahead=largs.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   augment_file=aug_test_data_file)

    data_loader = DataLoader(train_data_function, test_data_function,
                                   batch_size=args.num_to_sample)


    args.size_training_set = len(train_data_function)
    hsize = data_loader.train_loader.data.shape[1]
    wsize = data_loader.train_loader.data.shape[2]

    info = model_dict['info']
    largs = info['args'][-1]

    try:
        print(largs.encoder_output_size)
    except:
        largs.encoder_output_size = 1000

    encoder_model = ConvVAE(largs.code_length,
                            input_size=largs.number_condition,
                            encoder_output_size=largs.encoder_output_size)
    encoder_model.load_state_dict(model_dict['vae_state_dict'])

    prior_model = PriorNetwork(size_training_set=largs.size_training_set,
                                code_length=largs.code_length,
                                k=largs.num_k)
    prior_model.codes = model_dict['codes']

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
        pca_batch(data_loader.validation_ordered_batch, 'test')
        pca_batch(data_loader.ordered_batch, 'train')

    if not args.pca_only:
        print("finding neighbors")
        nearest_neighbor_batch()


#    while train_cnt < args.num_examples_to_train:
#        train_cnt = train_acn(train_cnt)
#
