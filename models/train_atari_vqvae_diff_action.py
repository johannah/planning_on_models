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
#from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_
from ae_utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_dict_losses
from ae_utils import save_checkpoint
from vqvae import VQVAE
#from vqvae import VQVAE_ENCODER
#from pixel_cnn import GatedPixelCNN
from datasets import AtariDataset
torch.manual_seed(394)

def handle_plot_ckpt(do_plot, train_cnt, avg_train_losses):
    info['train_losses_list'].append(avg_train_losses)
    info['train_cnts'].append(train_cnt)
    avg_valid_losses = valid_vqvae(train_cnt, do_plot)
    info['valid_losses_list'].append(avg_valid_losses)
    info['valid_cnts'].append(train_cnt)
    print('examples %010d loss' %train_cnt, info['train_losses_list'][-1])
    # plot
    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_losses_list'])<rolling*3:
            rolling = 0
        train_losses = np.array(info['train_losses_list'])
        valid_losses = np.array(info['train_losses_list'])
        for i in range(valid_losses.shape[1]):
            plot_name = model_base_filepath + "_%010d_loss%s.png"%(train_cnt, i)
            print("plotting", os.path.split(plot_name)[1])
            plot_dict = {
                         'valid loss %s'%i:{'index':info['valid_cnts'],
                                            'val':valid_losses[:,i]},
                         'train loss %s'%i:{'index':info['train_cnts'],
                                            'val':train_losses[:,i]},
                        }
            plot_dict_losses(plot_dict, name=plot_name, rolling_length=rolling)
        tot_plot_name = model_base_filepath + "_%010d_loss.png"%train_cnt
        tot_plot_dict = {
                         'valid loss':{'index':info['valid_cnts'],
                                            'val':valid_losses.sum(axis=1)},
                         'train loss %s'%i:{'index':info['train_cnts'],
                                            'val':train_losses.sum(axis=1)},
                    }
        plot_dict_losses(tot_plot_dict, name=tot_plot_name, rolling_length=rolling)
        print("plotting", os.path.split(tot_plot_name)[1])

def handle_checkpointing(train_cnt, loss_list):
    if ((train_cnt-info['last_save'])>=args.save_every):
        print("Saving model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['last_save']))
        info['last_save'] = train_cnt
        info['save_times'].append(time.time())
        handle_plot_ckpt(True, train_cnt, loss_list)
        filename = model_base_filepath + "_%010dex.pt"%train_cnt
        print("SAVING MODEL:%s" %filename)
        state = {
                 'vqvae_state_dict':vqvae_model.state_dict(),
                 'optimizer':opt.state_dict(),
                 'embedding':vqvae_model.embedding,
                 'info':info,
                 }
        save_checkpoint(state, filename=filename)
    elif not len(info['train_cnts']):
        print("Logging: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(True, train_cnt, loss_list)
    elif (train_cnt-info['last_plot'])>=args.plot_every:
        print("Calling plot at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, loss_list)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every:
            print("Logging at cnt:%s cnt since last logged:%s"%(train_cnt, train_cnt-info['train_cnts'][-1]))
            handle_plot_ckpt(False, train_cnt, loss_list)

def reshape_input(ss):
    # reshape 84x84 because needs to be divisible by 2 for each of the 4 layers
    return ss[:,:,2:-2,2:-2]

def train_vqvae(train_cnt):
    st = time.time()
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    batches = 0
    while train_cnt < args.num_examples_to_train:
        vqvae_model.train()
        opt.zero_grad()
        #states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
        states, actions, rewards, pred_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_framediff_minibatch()
        # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
        states = (2*reshape_input(states)-1).to(DEVICE)
        rec = (2*reshape_input(pred_states[:,0][:,None])-1).to(DEVICE)
        actions = actions.to(DEVICE)
        x_d, z_e_x, z_q_x, latents, pred_actions = vqvae_model(states)
        # dont normalize diff
        diff = (reshape_input(pred_states[:,1][:,None])).to(DEVICE)
        # (args.nr_logistic_mix/2)*3 is needed for each reconstruction
        z_q_x.retain_grad()
        rec_est =  x_d[:, :nmix]
        diff_est = x_d[:, nmix:]
        loss_rec = discretized_mix_logistic_loss(rec_est, rec, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
        loss_diff = discretized_mix_logistic_loss(diff_est, diff, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
        loss_act = F.nll_loss(pred_actions, actions)
        loss_act.backward(retain_graph=True)
        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_3 = args.beta*F.mse_loss(z_e_x, z_q_x.detach())
        loss_rec.backward(retain_graph=True)
        loss_diff.backward(retain_graph=True)
        vqvae_model.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)
        loss_2.backward(retain_graph=True)
        loss_3.backward()
        parameters = list(vqvae_model.parameters())
        clip_grad_value_(parameters, 10)
        opt.step()
        bs = float(x_d.shape[0])
        loss_list = [loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
        if batches > 100:
            handle_checkpointing(train_cnt, loss_list)
        train_cnt+=len(states)
        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def valid_vqvae(train_cnt, do_plot=False):
    vqvae_model.eval()
    #states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = valid_data_loader.get_unique_minibatch()
    states, actions, rewards, pred_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_framediff_minibatch()
    # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
    states = (2*reshape_input(states)-1).to(DEVICE)
    rec = (2*reshape_input(pred_states[:,0][:,None])-1).to(DEVICE)
    diff = (2*reshape_input(pred_states[:,1][:,None])-1).to(DEVICE)
    actions = actions.to(DEVICE)
    x_d, z_e_x, z_q_x, latents, pred_actions = vqvae_model(states)
    # (args.nr_logistic_mix/2)*3 is needed for each reconstruction
    z_q_x.retain_grad()
    rec_est =  x_d[:, :nmix]
    diff_est = x_d[:, nmix:]
    loss_rec = discretized_mix_logistic_loss(rec_est, rec, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
    loss_diff = discretized_mix_logistic_loss(diff_est, diff, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
    loss_act = F.nll_loss(pred_actions, actions)
    loss_act.backward(retain_graph=True)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = args.beta*F.mse_loss(z_e_x, z_q_x.detach())
    bs,yc,yh,yw = x_d.shape
    yhat = sample_from_discretized_mix_logistic(rec_est, args.nr_logistic_mix)
    if do_plot:
        print('writing img')
        n_imgs = 8
        n = min(states.shape[0], n_imgs)
        gold = (rec.to('cpu')+1)/2.0
        bs,_,h,w = gold.shape
        # sample from discretized should be between 0 and 255
        print("yhat sample", yhat[:,0].min().item(), yhat[:,0].max().item())
        yimg = ((yhat + 1.0)/2.0).to('cpu')
        print("yhat img", yhat.min().item(), yhat.max().item())
        print("gold img", gold.min().item(), gold.max().item())
        comparison = torch.cat([gold.view(bs,1,h,w)[:n],
                                yimg.view(bs,1,h,w)[:n]])
        img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
        save_image(comparison, img_name, nrow=n)
    bs = float(states.shape[0])
    loss_list = [loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
    return loss_list

if __name__ == '__main__':
    from argparse import ArgumentParser

    debug = 0
    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_data_file', default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/training_set.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--savename', default='vqdiffact')
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    if not debug:
        parser.add_argument('-se', '--save_every', default=100000*5, type=int)
        parser.add_argument('-pe', '--plot_every', default=100000*5, type=int)
        parser.add_argument('-le', '--log_every',  default=100000*5, type=int)
    else:
        parser.add_argument('-se', '--save_every', default=10, type=int)
        parser.add_argument('-pe', '--plot_every', default=10, type=int)
        parser.add_argument('-le', '--log_every',  default=10, type=int)
    parser.add_argument('-b', '--beta', default=0.25, type=float, help='scale for loss 3, commitment loss in vqvae')
    parser.add_argument('-z', '--num_z', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=84, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=1000000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1.5e-5)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    train_data_file = args.train_data_file
    data_dir = os.path.split(train_data_file)[0]
    valid_data_file = train_data_file.replace('training', 'valid')


    if args.model_loadpath == '':
         train_cnt = 0
         run_num = 0
         model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
         while os.path.exists(model_base_filedir):
             run_num +=1
             model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
         os.makedirs(model_base_filedir)
         model_base_filepath = os.path.join(model_base_filedir, args.savename)
         print("MODEL BASE FILEPATH", model_base_filepath)

         info = {'train_cnts':[],
                 'train_losses_list':[],
                 'valid_cnts':[],
                 'valid_losses_list':[],
                 'save_times':[],
                 'args':[args],
                 'last_save':0,
                 'last_plot':0,
                 'norm_by':255.0,
                  }


         ## size of latents flattened - dependent on architecture of vqvae
         #info['float_condition_size'] = 100*args.num_z
         ## 3x logistic needed for loss
         ## TODO - change loss
    else:
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info =  model_dict['info']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        train_cnt = info['train_cnts'][-1]
        info['loaded_from'] = args.model_loadpath
    train_data_loader = AtariDataset(
                                   train_data_file,
                                   number_condition=args.number_condition,
                                   steps_ahead=1,
                                   batch_size=args.batch_size,
                                   norm_by=info['norm_by'])
    valid_data_loader = AtariDataset(
                                   valid_data_file,
                                   number_condition=args.number_condition,
                                   steps_ahead=1,
                                   batch_size=args.batch_size,
                                   norm_by=info['norm_by'])
    num_actions = info['num_actions'] = train_data_loader.n_actions
    args.size_training_set = train_data_loader.num_examples
    hsize = train_data_loader.data_h
    wsize = train_data_loader.data_w
    # output mixtures should be 2*nr_logistic_mix + nr_logistic mix for each
    # decorelated channel
    info['num_channels'] = 2
    info['num_output_mixtures']= (2*args.nr_logistic_mix+args.nr_logistic_mix)*info['num_channels']
    nmix = int(info['num_output_mixtures']/2)
    vqvae_model = VQVAE(num_clusters=args.num_k,
                        encoder_output_size=args.num_z,
                        num_output_mixtures=info['num_output_mixtures'],
                        in_channels_size=args.number_condition,
                        n_actions=info['num_actions']).to(DEVICE)

    parameters = list(vqvae_model.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    if args.model_loadpath != '':
        vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        vqvae_model.embedding = model_dict['embedding']

    #args.pred_output_size = 1*80*80
    ## 10 is result of structure of network
    #args.z_input_size = 10*10*args.num_z
    train_cnt = train_vqvae(train_cnt)

