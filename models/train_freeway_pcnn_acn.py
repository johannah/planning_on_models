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
#from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_losses
from pixel_cnn import GatedPixelCNN
from datasets import FreewayForwardDataset, DataLoader
from acn import ConvVAE, PriorNetwork, acn_loss_function
torch.manual_seed(394)

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
                 'vae_state_dict':encoder_model.state_dict(),
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
    encoder_model.train()
    prior_model.train()
    train_loss = 0
    init_cnt = train_cnt
    st = time.time()
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    batches = 0
    while train_cnt < args.num_examples_to_train:
        lst = time.time()
        data, label, data_index = data_loader.next_batch()
        #for xx,i in enumerate(label):
        #    label_size[xx] = i
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        opt.zero_grad()
        z, u_q, s_q = encoder_model(data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=label, float_condition=z))
        prior_model.codes[data_index-args.number_condition] = u_q.detach().cpu().numpy()
        prior_model.fit_knn(prior_model.codes)
        u_p, s_p = prior_model(u_q)
        # how was bce working here?
        loss = acn_loss_function(yhat_batch, label, u_q, s_q, u_p, s_p)
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_loss = train_loss/float((train_cnt+data.shape[0])-init_cnt)
        handle_checkpointing(train_cnt, avg_train_loss)
        train_cnt+=len(data)
        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def test_acn(train_cnt, do_plot):
    encoder_model.eval()
    prior_model.eval()
    test_loss = 0
    print('starting test', train_cnt)
    st = time.time()
    test_cnt = 0
    with torch.no_grad():
        i = 0
        #for i, (data, label, data_index) in enumerate(test_loader):
        data, label, data_index = data_loader.validation_data()
        lst = time.time()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        z, u_q, s_q = encoder_model(data)
        #yhat_batch = encoder_model.decode(u_q, s_q, data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=label, float_condition=z))
        u_p, s_p = prior_model(u_q)
        loss = acn_loss_function(yhat_batch, label, u_q, s_q, u_p, s_p)
        test_loss+= loss.item()
        test_cnt += data.shape[0]
        if i == 0 and do_plot:
            print('writing img')
            n = min(data.size(0), 8)
            bs = data.shape[0]
            comparison = torch.cat([label.view(bs, 1, hsize, wsize)[:n],
                                  yhat_batch.view(bs, 1, hsize, wsize)[:n]])
            img_name = vae_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
            save_image(comparison.cpu(), img_name, nrow=n)
            print('finished writing img', img_name)
        #print('loop test', i, time.time()-lst)

    test_loss /= float(test_cnt)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('finished test', time.time()-st)
    return test_loss

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train acn for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--savename', default='fpcnn_acn')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-da', '--data_augmented', default=False, action='store_true')
    parser.add_argument('-daf', '--data_augmented_by_model', default="None")
    parser.add_argument('-se', '--save_every', default=100000*3, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000, type=int)
    parser.add_argument('-le', '--log_every', default=100000, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-eos', '--encoder_output_size', default=1000, type=int)
    parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=20, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-nc', '--num_classes', default=10)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=12)

    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(config.model_savedir, args.savename)

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
                                   number_condition=args.number_condition,
                                   steps_ahead=args.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   augment_file=aug_train_data_file)
    test_data_function = FreewayForwardDataset(
                                   test_data_file,
                                   number_condition=args.number_condition,
                                   steps_ahead=args.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   augment_file=aug_test_data_file)

    data_loader = DataLoader(train_data_function, test_data_function,
                                   batch_size=args.batch_size)


    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    args.size_training_set = len(train_data_function)
    hsize = data_loader.train_loader.data.shape[1]
    wsize = data_loader.train_loader.data.shape[2]

    encoder_model = ConvVAE(args.code_length, input_size=args.number_condition,
                            encoder_output_size=args.encoder_output_size).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=args.size_training_set,
                               code_length=args.code_length, DEVICE=DEVICE,
                               k=args.num_k).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=args.possible_values,
                                 n_layers=args.num_pcnn_layers,
                                 n_classes=args.num_classes,
                                 float_condition_size=args.encoder_output_size,
                                 last_layer_bias=0.5,
                                 hsize=hsize, wsize=wsize).to(DEVICE)

    parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0
    #while train_cnt < args.num_examples_to_train:
    train_cnt = train_acn(train_cnt)

