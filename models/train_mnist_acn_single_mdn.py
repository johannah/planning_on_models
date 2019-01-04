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
from torch.nn.utils.clip_grad import clip_grad_value_
from datasets import IndexedDataset
from acn_mdn_single import ConvVAE, PriorNetwork, acn_loss_function, acn_mdn_loss_function
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_dict_losses
from pixel_cnn import GatedPixelCNN
torch.manual_seed(394)

def handle_plot_ckpt(do_plot, train_cnt, avg_train_kl_loss, avg_train_rec_loss):
    info['train_kl_losses'].append(avg_train_kl_loss)
    info['train_rec_losses'].append(avg_train_rec_loss)
    info['train_losses'].append(avg_train_rec_loss + avg_train_kl_loss)
    info['train_cnts'].append(train_cnt)
    test_kl_loss, test_rec_loss = test_acn(train_cnt,do_plot)
    info['test_kl_losses'].append(test_kl_loss)
    info['test_rec_losses'].append(test_rec_loss)
    info['test_losses'].append(test_rec_loss + test_kl_loss)
    info['test_cnts'].append(train_cnt)
    print('examples %010d train kl loss %03.03f test kl loss %03.03f' %(train_cnt,
                              info['train_kl_losses'][-1], info['test_kl_losses'][-1]))
    print('---------------train rec loss %03.03f test rec loss %03.03f' %(
                              info['train_rec_losses'][-1], info['test_rec_losses'][-1]))
    # plot
    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_kl_losses'])<rolling*3:
            rolling = 1
        print('adding last loss plot', train_cnt)
        plot_name = model_base_filepath + "_%010dloss.png"%train_cnt
        print('plotting loss: %s with %s points'%(plot_name, len(info['train_cnts'])))
        plot_dict = {
                     'test kl':{'index':info['test_cnts'],
                                'val':info['test_kl_losses']},
                     'test rec':{'index':info['test_cnts'],
                                   'val':info['test_rec_losses']},
                     'test loss':{'index':info['test_cnts'],
                                   'val':info['test_losses']},
                     'train kl':{'index':info['train_cnts'],
                                   'val':info['train_kl_losses']},
                     'train rec':{'index':info['train_cnts'],
                                   'val':info['train_rec_losses']},
                     'train loss':{'index':info['train_cnts'],
                                   'val':info['train_losses']},
                    }

        plot_dict_losses(plot_dict, name=plot_name, rolling_length=rolling)

def handle_checkpointing(train_cnt, avg_train_kl_loss, avg_train_rec_loss):
    if ((train_cnt-info['last_save'])>=args.save_every):
        print("Saving Model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['last_save']))
        info['last_save'] = train_cnt
        info['save_times'].append(time.time())
        handle_plot_ckpt(True, train_cnt, avg_train_kl_loss, avg_train_rec_loss)
        filename = model_base_filepath + "_%010dex.pkl"%train_cnt
        state = {
                 'vae_state_dict':encoder_model.state_dict(),
                 'prior_state_dict':prior_model.state_dict(),
                 'pcnn_state_dict':pcnn_decoder.state_dict(),
                 'optimizer':opt.state_dict(),
                 'info':info,
                 'codes':prior_model.codes,
                 }
        save_checkpoint(state, filename=filename)
    elif not len(info['train_cnts']):
        print("Logging model: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(False, train_cnt, avg_train_kl_loss, avg_train_rec_loss)
    elif (train_cnt-info['last_plot'])>=args.plot_every:
        print("Plotting Model at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, avg_train_kl_loss, avg_train_rec_loss)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every:
            print("Logging Model at cnt:%s cnt since last logged:%s"%(train_cnt, train_cnt-info['train_cnts'][-1]))
            handle_plot_ckpt(False, train_cnt, avg_train_kl_loss, avg_train_rec_loss)

def train_acn(train_cnt):
    train_kl_loss = 0.0
    train_rec_loss = 0.0
    init_cnt = train_cnt
    st = time.time()
    for batch_idx, (data, label, data_index) in enumerate(train_loader):
        encoder_model.train()
        prior_model.train()
        pcnn_decoder.train()
        lst = time.time()
        data = data.to(DEVICE)
        opt.zero_grad()
        z, u_q = encoder_model(data)
        #yhat_batch = encoder_model.decode(u_q, s_q, data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=data, float_condition=z))

        np_uq = u_q.detach().cpu().numpy()
        if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
            print('train bad')
            embed()
        prior_model.codes[data_index] = np_uq
        #prior_model.fit_knn(prior_model.codes)
        # output is mdn
        u_ps, s_ps = prior_model(u_q)
        #kl_loss, rec_loss = acn_loss_function(yhat_batch, data, u_q, u_ps, s_ps)
        kl_loss, rec_loss = acn_mdn_loss_function(yhat_batch, data, u_q, u_ps, s_ps)
        loss = kl_loss + rec_loss
        loss.backward()
        parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
        clip_grad_value_(parameters, 10)
        train_kl_loss+= kl_loss.item()
        train_rec_loss+= rec_loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_kl_loss = train_kl_loss/float((train_cnt+data.shape[0])-init_cnt)
        avg_train_rec_loss = train_rec_loss/float((train_cnt+data.shape[0])-init_cnt)
        handle_checkpointing(train_cnt, avg_train_kl_loss, avg_train_rec_loss)
        train_cnt+=len(data)
    print("finished epoch after %s seconds at cnt %s"%(time.time()-st, train_cnt))
    return train_cnt

def test_acn(train_cnt, do_plot):
    encoder_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()
    test_kl_loss = 0.0
    test_rec_loss = 0.0
    print('starting test', train_cnt)
    st = time.time()
    print(len(test_loader))
    #with torch.no_grad():
    for i, (data, label, data_index) in enumerate(test_loader):
        lst = time.time()
        data = data.to(DEVICE)
        z, u_q = encoder_model(data)
        np_uq = u_q.detach().cpu().numpy()
        #yhat_batch = encoder_model.decode(u_q, s_q, data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=data, float_condition=z))
        if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
            print('baad')
            embed()
        u_ps, s_ps = prior_model(u_q)
        #kl_loss,rec_loss = acn_loss_function(yhat_batch, data, u_q, u_ps, s_ps)
        kl_loss,rec_loss = acn_mdn_loss_function(yhat_batch, data, u_q, u_ps, s_ps)
        #loss = acn_loss_function(yhat_batch, data, u_q, u_p, s_p)
        test_kl_loss+= kl_loss.item()
        test_rec_loss+= rec_loss.item()
        if i == 0 and do_plot:
            print('writing img')
            n = min(data.size(0), 8)
            bs = data.shape[0]
            comparison = torch.cat([data.view(bs, 1, 28, 28)[:n],
                                  yhat_batch.view(bs, 1, 28, 28)[:n]])
            img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
            save_image(comparison.cpu(), img_name, nrow=n)
            print('finished writing img', img_name)
        #print('loop test', i, time.time()-lst)

    test_kl_loss /= len(test_loader.dataset)
    test_rec_loss /= len(test_loader.dataset)
    print('====> Test set kl loss: {:.4f}'.format(test_kl_loss))
    print('====> Test set rec loss: {:.4f}'.format(test_rec_loss))
    print('finished test', time.time()-st)
    return test_kl_loss, test_rec_loss

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-sn', '--savename', default='mdn1logkl3')
    parser.add_argument('-se', '--save_every', default=60000*6, type=int)
    parser.add_argument('-pe', '--plot_every', default=60000*2, type=int)
    parser.add_argument('-le', '--log_every', default=60000*2, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-nm', '--num_mixtures', default=1, type=int)
    parser.add_argument('-uniq', '--require_unique_codes', default=False, type=bool)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=48, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-nc', '--num_classes', default=10)
    parser.add_argument('-eos', '--encoder_output_size', default=432)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=12)

    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    run_num = 0
    model_base_filedir = os.path.join(config.model_savedir, args.savename + '%02d'%run_num)
    while os.path.exists(model_base_filedir):
        run_num +=1
        model_base_filedir = os.path.join(config.model_savedir, args.savename + '%02d'%run_num)
    os.makedirs(model_base_filedir)
    model_base_filepath = os.path.join(model_base_filedir, args.savename)

    train_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = IndexedDataset(datasets.MNIST, path=config.base_datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    nchans,hsize,wsize = test_loader.dataset[0][0].shape

    info = {'train_cnts':[],
            'train_kl_losses':[],
            'train_rec_losses':[],
            'train_losses':[],
            'test_cnts':[],
            'test_kl_losses':[],
            'test_rec_losses':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    args.size_training_set = len(train_data)
    nchans,hsize,wsize = test_loader.dataset[0][0].shape
    encoder_model = ConvVAE(args.code_length, input_size=1,
                        encoder_output_size=args.encoder_output_size).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=args.size_training_set,
                               code_length=args.code_length,
                               n_mixtures=args.num_mixtures,
                               k=args.num_k,
                               require_unique_codes=args.require_unique_codes).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                      dim=args.possible_values,
                                      n_layers=args.num_pcnn_layers,
                                      n_classes=args.num_classes,
                                      float_condition_size=args.code_length,
                                      last_layer_bias=0.5, hsize=hsize, wsize=wsize).to(DEVICE)

    parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0
    while train_cnt < args.num_examples_to_train:
        train_cnt = train_acn(train_cnt)

