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
from torch.utils.data import Dataset, DataLoader
from datasets import IndexedDataset
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_losses
from pixel_cnn import GatedPixelCNN
from acn_mdn import PriorNetwork, ConvVAE

torch.manual_seed(394)
torch.set_num_threads(1)

def sample_batch(data, label, batch_idx, name):
    z, u_q = encoder_model(data)
    print('generating %s images' %(data.shape[0]))
    print(batch_idx)
    if args.teacher_force:
        canvas = label
        name+='_tf'
    else:
        canvas = 0.0*label
    label = label.detach().numpy()
    np_canvas = np.zeros_like(label)
    for bi in range(canvas.shape[0]):
        # sample one at a time due to memory constraints
        print('sampling image', bi)
        for i in range(canvas.shape[1]):
            for j in range(canvas.shape[2]):
                for k in range(canvas.shape[3]):
                    output = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1], float_condition=z[bi:bi+1]))
                    np_canvas[bi,i,j,k] = output[0,i,j,k].detach().numpy()
        print("starting img")
        f,ax = plt.subplots(1,2)
        iname = os.path.join(output_savepath, '%s_%04d.png'%(name,bi))
        ax[0].imshow(label[bi,0])
        ax[0].set_title('true')
        ax[1].imshow(np_canvas[bi,0])
        ax[1].set_title('est')
        plt.savefig(iname)
        print('saving', iname)



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='sample acn pcnn for freeway')
    parser.add_argument('-l', '--model_loadname', required=True, help='filename of pkl file to load models from')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-bs', '--batch_size', default=10, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-n', '--num_to_sample', default=10, type=int)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
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


    train_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    train_cnt = 0

    info = model_dict['info']
    largs = info['args'][-1]

    ## in future these are in largs
    #try:
    #    size_training_set = largs.size_training_set
    #    possible_values = largs.possible_values
    #    num_classes = largs.num_classes
    #    num_pcnn_layers = largs.num_pcnn_layers
    #except:
    #    size_training_set = len(train_data)
    #    possible_values = 1
    #    num_classes = 10
    #    num_pcnn_layers = 12

    nchans,hsize,wsize = test_loader.dataset[0][0].shape

    encoder_model = ConvVAE(largs.code_length, input_size=1,
                            encoder_output_size=largs.encoder_output_size)
    encoder_model.load_state_dict(model_dict['vae_state_dict'])

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=largs.possible_values,
                                 n_layers=largs.num_pcnn_layers,
                                n_classes=largs.num_classes,
                                float_condition_size=largs.code_length,
                                 last_layer_bias=0.5, hsize=hsize, wsize=wsize)
    pcnn_decoder.load_state_dict(model_dict['pcnn_state_dict'])

    encoder_model.eval()
    pcnn_decoder.eval()
    for test_data, test_label, test_batch_index in test_loader:
        break
    sample_batch(test_data, test_data, test_batch_index, 'test')

    for train_data, train_label, train_batch_index in test_loader:
        break
    sample_batch(train_data, train_data, train_batch_index, 'train')



#    num_samples = 0
#    get_new_sample = True
#    while num_samples <= args.num_to_sample:
#        o = train_data[num_samples]
#        label = o[1]
#        data = o[0][None]
#        z, u_q = encoder_model(data)
#        print('generating sample: %s' %num_samples)
#
#        canvas = 0.0*data
#        for i in range(canvas.shape[1]):
#            for j in range(canvas.shape[2]):
#                for k in range(canvas.shape[3]):
#                    output = torch.sigmoid(pcnn_decoder(x=canvas, float_condition=z))
#                    canvas[:,i,j,k] = output[:,i,j,k]
#
#        f,ax = plt.subplots(1,2)
#        iname = os.path.join(output_savepath, 'train%04d.png'%(num_samples))
#        ax[0].imshow(data[0,0].numpy(), cmap=plt.cm.gray)
#        ax[0].set_title('true')
#        ax[1].imshow(canvas[0,0].detach().numpy(), cmap=plt.cm.gray)
#        ax[1].set_title('est')
#        plt.savefig(iname)
#        num_samples += 1
#
#
#    embed()
#
#    while train_cnt < args.num_examples_to_train:
#        train_cnt = train_acn(train_cnt)
#
