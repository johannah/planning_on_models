
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
from vqvae_module import VQVAE_ENCODER, VQVAE_PCNN_DECODER, VQVAE, get_vqvae_loss
#from vqvae import VQVAE_ENCODER
#from pixel_cnn import GatedPixelCNN
from datasets import AtariDataset
torch.manual_seed(394)

def handle_plot_ckpt(do_plot, train_cnt, avg_train_loss_1, avg_train_loss_2, avg_train_loss_3):
    info['train_losses_1'].append(avg_train_loss_1)
    info['train_losses_2'].append(avg_train_loss_2)
    info['train_losses_3'].append(avg_train_loss_3)
    info['train_losses'].append(avg_train_loss_1 + avg_train_loss_2 + avg_train_loss_3)
    info['train_cnts'].append(train_cnt)
    avg_valid_loss_1, avg_valid_loss_2, avg_valid_loss_3 = valid_vqvae(train_cnt,do_plot)
    info['valid_losses_1'].append(avg_valid_loss_1)
    info['valid_losses_2'].append(avg_valid_loss_2)
    info['valid_losses_3'].append(avg_valid_loss_3)
    info['valid_losses'].append(avg_valid_loss_1 + avg_valid_loss_2 + avg_valid_loss_3)
    info['valid_cnts'].append(train_cnt)
    print('examples %010d tloss1 %03.03f tloss2 %03.03f tloss3 %03.03f' %(train_cnt,
                              info['train_losses_1'][-1],
                              info['train_losses_2'][-1],
                              info['train_losses_2'][-1]))
    # plot

    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_losses_1'])<rolling*3:
            rolling = 1
        print('adding last loss plot', train_cnt)
        l1_plot_name = model_base_filepath + "_%010d_loss1.png"%train_cnt
        l2_plot_name = model_base_filepath + "_%010d_loss2.png"%train_cnt
        l3_plot_name = model_base_filepath + "_%010d_loss3.png"%train_cnt
        tot_plot_name = model_base_filepath + "_%010d_loss.png"%train_cnt
        print('plotting loss: %s with %s points'%(tot_plot_name, len(info['train_cnts'])))
        l1_plot_dict = {
                     'valid loss 1':{'index':info['valid_cnts'],
                                'val':info['valid_losses_1']},
                     'train loss 1':{'index':info['train_cnts'],
                                   'val':info['train_losses_1']},
                    }
        l2_plot_dict = {
                     'valid loss 2':{'index':info['valid_cnts'],
                                'val':info['valid_losses_2']},
                     'train loss 2':{'index':info['train_cnts'],
                                   'val':info['train_losses_2']},
                    }
        l3_plot_dict = {
                     'valid loss 3':{'index':info['valid_cnts'],
                                'val':info['valid_losses_3']},
                     'train loss 3':{'index':info['train_cnts'],
                                   'val':info['train_losses_3']},
                    }

        tot_plot_dict = {
                     'valid loss':{'index':info['valid_cnts'],
                                   'val':info['valid_losses']},
                     'train loss':{'index':info['train_cnts'],
                                   'val':info['train_losses']},
                    }

        plot_dict_losses(l1_plot_dict, name=l1_plot_name, rolling_length=rolling)
        plot_dict_losses(l2_plot_dict, name=l2_plot_name, rolling_length=rolling)
        plot_dict_losses(l3_plot_dict, name=l3_plot_name, rolling_length=rolling)
        plot_dict_losses(tot_plot_dict, name=tot_plot_name, rolling_length=rolling)

def handle_checkpointing(train_cnt, avg_loss_1, avg_loss_2, avg_loss_3):
    if ((train_cnt-info['last_save'])>=args.save_every):
        print("Saving model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['last_save']))
        info['last_save'] = train_cnt
        info['save_times'].append(time.time())
        handle_plot_ckpt(True, train_cnt, avg_loss_1, avg_loss_2, avg_loss_3)
        filename = model_base_filepath + "_%010dex.pt"%train_cnt
        print("SAVING MODEL:%s" %filename)
        state = {
                 'vqvae_state_dict':vqvae_model.state_dict(),
                 'optimizer':opt.state_dict(),
                 'embedding':vqvae_model.encoder.embedding,
                 'info':info,
                 }
        save_checkpoint(state, filename=filename)
    elif not len(info['train_cnts']):
        print("Logging: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(False, train_cnt, avg_loss_1, avg_loss_2, avg_loss_3)
    elif (train_cnt-info['last_plot'])>=args.plot_every:
        print("Calling plot at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, avg_loss_1, avg_loss_2, avg_loss_3)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every:
            print("Logging at cnt:%s cnt since last logged:%s"%(train_cnt, train_cnt-info['train_cnts'][-1]))
            handle_plot_ckpt(False, train_cnt, avg_loss_1, avg_loss_2, avg_loss_3)

def reshape_input(ss):
    # reshape 84x84 because needs to be divisible by 2 for each of the 4 layers
    return ss[:,:,2:-2,2:-2]

#def forward_pass(enc, dec, states, next_states, actions,  nr_logistic_mix, train=True, device='cpu', beta=0.25):
#    states = reshape_input(states)
#    next_states = reshape_input(next_states)
#    states = states.to(device)
#    # 1 channel expected
#    next_states = next_states[:,-1:].to(device)
#    actions = actions.to(device)
#    z_e_x, z_q_x, latents = enc(states)
#    z_q_x.retain_grad()
#    ns = (2*next_states)-1
#    fc = latents.view(latents.shape[0], latents.shape[1]*latents.shape[2]).float()
#    yhat =  pcnn_decoder(x=ns, class_condition=actions, float_condition=fc)
#    x_d =
#    embed()
#    loss_1 = discretized_mix_logistic_loss(x_d,ns,nr_mix=nr_logistic_mix, DEVICE=device)
#    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
#    loss_3 = beta*F.mse_loss(z_e_x, z_q_x.detach())
#    if train:
#        loss_1.backward(retain_graph=True)
#        model.embedding.zero_grad()
#        z_e_x.backward(z_q_x.grad, retain_graph=True)
#        loss_2.backward(retain_graph=True)
#        loss_3.backward()
#        parameters = list(model.parameters())
#        clip_grad_value_(parameters, 10)
#        opt.step()
#    ne = float(next_states.shape[0])
#    return x_d, z_e_x, z_q_x, latents, loss_1.item()/ne, loss_2.item()/ne, loss_3.item()/ne

def train_vqvae(train_cnt):
    st = time.time()
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    batches = 0
    while train_cnt < args.num_examples_to_train:
        vqenc.train()
        pcnn_decoder.train()
        opt.zero_grad()
        states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
        # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
        states = reshape_input(states).to(DEVICE)
        # only predict future observation - normalize
        targets = (2*states[:,-1:]-1).to(DEVICE)
        #actions = actions.to(DEVICE)
        x_d, z_e_x, z_q_x, latents = vqvae_model(states, targets)
        #z_e_x, z_q_x, latents = vqenc(states)
        #float_condition = latents.view(latents.shape[0], latents.shape[1]*latents.shape[2]).float()
        #x_d = pcnn_decoder(targets, class_condition=actions, float_condition=float_condition)
        z_q_x.retain_grad()
        vqvae_model.spatial_condition.retain_grad()
        loss_1 = discretized_mix_logistic_loss(x_d, targets, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_3 = args.beta*F.mse_loss(z_e_x, z_q_x.detach())
        #loss_1, loss_2, loss_3 = get_vqvae_loss(x_d, targets, z_e_x, z_q_x, nr_logistic_mix=args.nr_logistic_mix, beta=args.beta, device=DEVICE)
        loss_1.backward(retain_graph=True)
        #vqvae_model.encoder.embedding.zero_grad()
        #z_e_x.backward(z_q_x.grad, retain_graph=True)
        z_e_x.backward(vqvae_model.spatial_condition.grad, retain_graph=True)
        loss_2.backward(retain_graph=True)
        loss_3.backward()
        parameters = list(vqvae_model.parameters())
        clip_grad_value_(parameters, 10)
        opt.step()

        bs = float(x_d.shape[0])
        handle_checkpointing(train_cnt, loss_1.item()/bs, loss_2.item()/bs, loss_3.item()/bs)
        train_cnt+=len(states)

        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def valid_vqvae(train_cnt, do_plot=False):
    vqvae_model.eval()
    opt.zero_grad()
    states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = valid_data_loader.get_unique_minibatch()
    # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
    states = reshape_input(states).to(DEVICE)
    # only predict future observation - normalize
    targets = (2*states[:,-1:]-1).to(DEVICE)
    #actions = actions.to(DEVICE)
    x_d, z_e_x, z_q_x, latents = vqvae_model(states, targets)
    loss_1 = discretized_mix_logistic_loss(x_d, targets, nr_mix=args.nr_logistic_mix, DEVICE=DEVICE)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = args.beta*F.mse_loss(z_e_x, z_q_x.detach())
    #loss_1, loss_2, loss_3 = get_vqvae_loss(x_d, targets, z_e_x, z_q_x, nr_logistic_mix=args.nr_logistic_mix, beta=args.beta, device=DEVICE)
    bs,yc,yh,yw = x_d.shape
    yhat = sample_from_discretized_mix_logistic(x_d, args.nr_logistic_mix)
    if do_plot:
        print('writing img')
        n_imgs = 8
        n = min(states.shape[0], n_imgs)
        gold = states[:,-1:]
        bs,_,h,w = gold.shape
        comparison = torch.cat([gold.to('cpu').view(bs,1,h,w)[:n],
                                yhat.to('cpu').view(bs,1,h,w)[:n]])
        img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
        save_image(comparison, img_name, nrow=n)
    bs = float(states.shape[0])
    return loss_1.item()/bs, loss_2.item()/bs, loss_3.item()/bs

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train ')
    parser.add_argument('--train_data_file', default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/training_set.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--savename', default='vqpcnnrec')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    parser.add_argument('-se', '--save_every', default=100000*4, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000*4, type=int)
    parser.add_argument('-le', '--log_every',  default=100000*4, type=int)
    #parser.add_argument('-se', '--save_every', default=10, type=int)
    #parser.add_argument('-pe', '--plot_every', default=10, type=int)
    #parser.add_argument('-le', '--log_every',  default=10, type=int)
    parser.add_argument('-b', '--beta', default=0.25, type=float, help='scale for loss 3, commitment loss in vqvae')
    parser.add_argument('-z', '--num_z', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    #parser.add_argument('-nm', '--num_mixtures', default=10, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('-eos', '--encoder_output_size', default=4800, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    #parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-pf', '--num_pcnn_filters', default=32, type=int)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=8)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    train_cnt = 0
    run_num = 0
    train_data_file = args.train_data_file
    data_dir = os.path.split(train_data_file)[0]
    model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
    while os.path.exists(model_base_filedir):
        run_num +=1
        model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
    os.makedirs(model_base_filedir)
    model_base_filepath = os.path.join(model_base_filedir, args.savename)
    print("MODEL BASE FILEPATH", model_base_filepath)

    info = {'train_cnts':[],
            'train_losses':[],
            'train_losses_1':[],
            'train_losses_2':[],
            'train_losses_3':[],
            'valid_cnts':[],
            'valid_losses':[],
            'valid_losses_1':[],
            'valid_losses_2':[],
            'valid_losses_3':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
            'norm_by':255.0,
             }


    ## size of latents flattened - dependent on architecture of vqvae
    #info['float_condition_size'] = 100*args.num_z
    ## 3x logistic needed for loss
    info['decoder_output_channels'] = args.nr_logistic_mix*3
    ## TODO - change loss
    valid_data_file = train_data_file.replace('training', 'valid')

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


    num_actions = train_data_loader.n_actions
    args.size_training_set = train_data_loader.num_examples
    hsize = train_data_loader.data_h
    wsize = train_data_loader.data_w
    vqenc = VQVAE_ENCODER(num_clusters=args.num_k, encoder_output_size=args.num_z,
                   in_channels_size=args.number_condition)

    pcnn_decoder = VQVAE_PCNN_DECODER(n_filters=args.num_pcnn_filters,
                                 n_layers=args.num_pcnn_layers,
                              #   n_classes=num_actions,
                              #   float_condition_size=info['float_condition_size'],
                              #  spatial_condition_size=1,
                                 hsize=hsize, wsize=wsize,
                                 num_output_channels=info['decoder_output_channels'])

    args.pred_output_size = 1*80*80
    # 10 is result of structure of network
    args.z_input_size = 10*10*args.num_z
    vqvae_model = VQVAE(vqenc, pcnn_decoder, z_input_size=args.z_input_size, pred_output_size=args.pred_output_size).to(DEVICE)
    #parameters = list(vqenc.parameters())+list(pcnn_decoder.parameters())
    parameters = list(vqvae_model.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    train_cnt = train_vqvae(train_cnt)

