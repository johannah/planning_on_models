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
from acn_mdn import ConvVAE, PriorNetwork
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
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-n', '--num_to_sample', default=5, type=int)
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

    encoder_model = ConvVAE(largs.code_length,
                            input_size=largs.number_condition,
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

    test_data, test_label, test_batch_index = data_loader.validation_ordered_batch()
    sample_batch(test_data, test_label, test_batch_index, 'test')

    train_data, train_label, train_batch_index = data_loader.ordered_batch()
    sample_batch(train_data, train_label, train_batch_index, 'train')


#    while train_cnt < args.num_examples_to_train:
#        train_cnt = train_acn(train_cnt)
#
