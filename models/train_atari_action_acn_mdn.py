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
from torch.nn.utils.clip_grad import clip_grad_value_
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_dict_losses
from ae_utils import save_checkpoint
from pixel_cnn import GatedPixelCNN
from datasets import AtariDataset
from acn_mdn import ConvVAE, PriorNetwork, acn_mdn_loss_function
torch.manual_seed(394)

def handle_plot_ckpt(do_plot, train_cnt, avg_train_kl_loss, avg_train_rec_loss):
    info['train_kl_losses'].append(avg_train_kl_loss)
    info['train_rec_losses'].append(avg_train_rec_loss)
    info['train_losses'].append(avg_train_rec_loss + avg_train_kl_loss)
    info['train_cnts'].append(train_cnt)
    valid_kl_loss, valid_rec_loss = valid_acn(train_cnt,do_plot)
    info['valid_kl_losses'].append(valid_kl_loss)
    info['valid_rec_losses'].append(valid_rec_loss)
    info['valid_losses'].append(valid_rec_loss + valid_kl_loss)
    info['valid_cnts'].append(train_cnt)
    print('examples %010d train kl loss %03.03f valid kl loss %03.03f' %(train_cnt,
                              info['train_kl_losses'][-1], info['valid_kl_losses'][-1]))
    print('---------------train rec loss %03.03f valid rec loss %03.03f' %(
                              info['train_rec_losses'][-1], info['valid_rec_losses'][-1]))
    # plot
    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_kl_losses'])<rolling*3:
            rolling = 1
        print('adding last loss plot', train_cnt)
        kl_plot_name = model_base_filepath + "_%010d_kl_loss.png"%train_cnt
        rec_plot_name = model_base_filepath + "_%010d_rec_loss.png"%train_cnt
        tot_plot_name = model_base_filepath + "_%010d_loss.png"%train_cnt
        print('plotting loss: %s with %s points'%(tot_plot_name, len(info['train_cnts'])))
        kl_plot_dict = {
                     'valid kl':{'index':info['valid_cnts'],
                                'val':info['valid_kl_losses']},
                     'train kl':{'index':info['train_cnts'],
                                   'val':info['train_kl_losses']},
                    }


        rec_plot_dict = {
                     'valid rec':{'index':info['valid_cnts'],
                                   'val':info['valid_rec_losses']},
                     'train rec':{'index':info['train_cnts'],
                                   'val':info['train_rec_losses']},
                    }

        tot_plot_dict = {
                     'valid loss':{'index':info['valid_cnts'],
                                   'val':info['valid_losses']},
                     'train loss':{'index':info['train_cnts'],
                                   'val':info['train_losses']},
                    }


        plot_dict_losses(kl_plot_dict, name=kl_plot_name, rolling_length=rolling)
        plot_dict_losses(rec_plot_dict, name=rec_plot_name, rolling_length=rolling)
        plot_dict_losses(tot_plot_dict, name=tot_plot_name, rolling_length=rolling)

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
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    batches = 0
    while train_cnt < args.num_examples_to_train:
        encoder_model.train()
        prior_model.train()
        pcnn_decoder.train()
        opt.zero_grad()
        states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
        states = states.to(DEVICE)
        # 1 channel expected
        next_states = next_states[:,args.number_condition-1:].to(DEVICE)
        actions = actions.to(DEVICE)
        z, u_q = encoder_model(states)
        np_uq = u_q.detach().cpu().numpy()
        if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
            print('train bad')
            embed()

        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(pcnn_decoder(x=next_states, class_condition=actions, float_condition=z))
        #yhat_batch = torch.sigmoid(pcnn_decoder(x=next_states, float_condition=z))
        #print(train_cnt)
        prior_model.codes[relative_indexes-args.number_condition] = u_q.detach().cpu().numpy()
        np_uq = u_q.detach().cpu().numpy()
        if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
            print('train bad')
            embed()
        mix, u_ps, s_ps = prior_model(u_q)
        kl_loss, rec_loss = acn_mdn_loss_function(yhat_batch, next_states, u_q, mix, u_ps, s_ps)
        loss = kl_loss + rec_loss
        loss.backward()
        parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
        clip_grad_value_(parameters, 10)
        train_kl_loss+= kl_loss.item()
        train_rec_loss+= rec_loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_kl_loss = train_kl_loss/float((train_cnt+states.shape[0])-init_cnt)
        avg_train_rec_loss = train_rec_loss/float((train_cnt+states.shape[0])-init_cnt)
        handle_checkpointing(train_cnt, avg_train_kl_loss, avg_train_rec_loss)
        train_cnt+=len(states)

        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def valid_acn(train_cnt, do_plot):
    valid_kl_loss = 0.0
    valid_rec_loss = 0.0
    print('starting valid', train_cnt)
    st = time.time()
    valid_cnt = 0
    encoder_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()
    opt.zero_grad()
    i = 0
    #data, label, data_index = data_loader.validation_data()
    states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = valid_data_loader.get_unique_minibatch()
    states = states.to(DEVICE)
    # 1 channel expected
    next_states = next_states[:,args.number_condition-1:].to(DEVICE)
    actions = actions.to(DEVICE)
    z, u_q = encoder_model(states)

    np_uq = u_q.detach().cpu().numpy()
    if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
        print('baad')
        embed()

    #yhat_batch = encoder_model.decode(u_q, s_q, data)
    # add the predicted codes to the input
    yhat_batch = torch.sigmoid(pcnn_decoder(x=next_states, class_condition=actions, float_condition=z))
    mix, u_ps, s_ps = prior_model(u_q)
    kl_loss,rec_loss = acn_mdn_loss_function(yhat_batch, next_states, u_q, mix, u_ps, s_ps)
    valid_kl_loss+= kl_loss.item()
    valid_rec_loss+= rec_loss.item()
    valid_cnt += states.shape[0]
    if i == 0 and do_plot:
        print('writing img')
        n_imgs = 8
        n = min(states.shape[0], n_imgs)
        #onext_states = torch.Tensor(next_states[:n].data.cpu().numpy()+train_data_loader.frames_mean)#*train_data_loader.frames_diff) + train_data_loader.frames_min)
        #oyhat_batch =  torch.Tensor( yhat_batch[:n].data.cpu().numpy()+train_data_loader.frames_mean)#*train_data_loader.frames_diff) + train_data_loader.frames_min)
        #onext_states = torch.Tensor(((next_states[:n].data.cpu().numpy()*train_data_loader.frames_diff)+train_data_loader.frames_min) + train_data_loader.frames_mean)/255.
        #oyhat_batch =  torch.Tensor((( yhat_batch[:n].data.cpu().numpy()*train_data_loader.frames_diff)+train_data_loader.frames_min) + train_data_loader.frames_mean)/255.
        bs = args.batch_size
        h = train_data_loader.data_h
        w = train_data_loader.data_w
        comparison = torch.cat([next_states.view(bs,1,h,w)[:n],
                                yhat_batch.view(bs,1,h,w)[:n]])
        img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
        save_image(comparison, img_name, nrow=n)
        #embed()
        #ocomparison = torch.cat([onext_states,
        #                        oyhat_batch])
        #img_name = model_base_filepath + "_%010d_valid_reconstructionMINE.png"%train_cnt
        #save_image(ocomparison, img_name, nrow=n)
        #embed()
        print('finished writing img', img_name)
    valid_kl_loss/=float(valid_cnt)
    valid_rec_loss/=float(valid_cnt)
    print('====> valid kl loss: {:.4f}'.format(valid_kl_loss))
    print('====> valid rec loss: {:.4f}'.format(valid_rec_loss))
    print('finished valid', time.time()-st)
    return valid_kl_loss, valid_rec_loss

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_data_file', default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/training_set.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--savename', default='acn')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    parser.add_argument('-se', '--save_every', default=100000*2, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000*2, type=int)
    parser.add_argument('-le', '--log_every',  default=100000*2, type=int)
    #parser.add_argument('-se', '--save_every', default=10, type=int)
    #parser.add_argument('-pe', '--plot_every', default=10, type=int)
    #parser.add_argument('-le', '--log_every',  default=10, type=int)

    parser.add_argument('-pf', '--num_pcnn_filters', default=32, type=int)
    parser.add_argument('-bs', '--batch_size', default=48, type=int)
    parser.add_argument('-eos', '--encoder_output_size', default=4800, type=int)
    parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=48, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4)
    #parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=8)
    parser.add_argument('-nm', '--num_mixtures', default=8, type=int)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    run_num = 0
    train_data_file = args.train_data_file
    data_dir = os.path.split(train_data_file)[0]
    model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
    while os.path.exists(model_base_filedir):
        run_num +=1
        model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
    os.makedirs(model_base_filedir)
    model_base_filepath = os.path.join(model_base_filedir, args.savename)

    # TODO - change loss
    valid_data_file = train_data_file.replace('training', 'valid')

    #if args.data_augmented:
    #    # model that was used to create the forward predictions dataset
    #    aug_train_data_file = train_data_file[:-4] + args.data_augmented_by_model + '.npz'
    #    aug_valid_data_file = valid_data_file[:-4] + args.data_augmented_by_model + '.npz'
    #    for i in [aug_train_data_file, aug_valid_data_file]:
    #        if not os.path.exists(i):
    #            print('augmented data file', i, 'does not exist')
    #            embed()
    #else:
    #    aug_train_data_file = "None"
    #    aug_valid_data_file = "None"


    train_data_loader = AtariDataset(
                                   train_data_file,
                                   number_condition=args.number_condition,
                                   steps_ahead=1,
                                   batch_size=args.batch_size,
                                   norm_by=255.,)
    valid_data_loader = AtariDataset(
                                   valid_data_file,
                                   number_condition=args.number_condition,
                                   steps_ahead=1,
                                   batch_size=args.batch_size,
                                   norm_by=255.0,)


    num_actions = train_data_loader.n_actions

    info = {'train_cnts':[],
            'train_losses':[],
            'train_kl_losses':[],
            'train_rec_losses':[],
            'valid_cnts':[],
            'valid_losses':[],
            'valid_kl_losses':[],
            'valid_rec_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    args.size_training_set = train_data_loader.num_examples
    hsize = train_data_loader.data_h
    wsize = train_data_loader.data_w

    encoder_model = ConvVAE(args.code_length, input_size=args.number_condition,
                            encoder_output_size=args.encoder_output_size,
                             ).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=args.size_training_set,
                               code_length=args.code_length,
                               n_mixtures=args.num_mixtures,
                               k=args.num_k,
                               require_unique_codes=args.require_unique_codes
                               ).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=args.num_pcnn_filters,
                                 n_layers=args.num_pcnn_layers,
                                 n_classes=num_actions,
                                 float_condition_size=args.code_length,
                                 last_layer_bias=0.5,
                                 hsize=hsize, wsize=wsize).to(DEVICE)

    parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = 0
    #while train_cnt < args.num_examples_to_train:
    train_cnt = train_acn(train_cnt)

