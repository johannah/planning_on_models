import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import sys
import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torchvision import datasets, transforms
from vqvae import AutoEncoder
#from vqvae_bigger import AutoEncoder
#from vqvae_small import AutoEncoder
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
import os
from imageio import imread, imwrite, mimsave
from PIL import Image
from ae_utils import discretized_mix_logistic_loss
from ae_utils import sample_from_discretized_mix_logistic
from ae_utils import get_cuts, to_scalar
from datasets import FreewayForwardDataset, DataLoader
from lstm_utils import plot_losses
import config
torch.manual_seed(7)

def forward_pass(x,y):
    x = Variable(x, requires_grad=False).to(DEVICE)
    y = Variable(y, requires_grad=False).to(DEVICE)
    x_d, z_e_x, z_q_x, latents = vmodel(x)
    # with bigger model - latents is 64, 6, 6
    z_q_x.retain_grad()
    #loss_1 = F.binary_cross_entropy(x_d, x)
    # going into dml - x should be bt 0 and 1
    loss_1 = discretized_mix_logistic_loss(x_d,2*y-1,DEVICE=DEVICE)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
    return loss_1, loss_2, loss_3, x_d, z_e_x, z_q_x, latents


def train():
    targs = args
    train_cnt = info['last_save']
    while train_cnt <= targs.num_examples_to_train:
        x,y,_ = data_loader.next_batch()
        opt.zero_grad()
        loss_1,loss_2,loss_3,x_d,z_e_x,z_q_x,latents = forward_pass(x,y)
        loss_1.backward(retain_graph=True)
        vmodel.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)
        loss_2.backward(retain_graph=True)
        loss_3.backward()
        opt.step()

        def handle_plot_ckpt(do_plot=False):
            train_loss = np.array(to_scalar([loss_1, loss_2, loss_3])).mean(0)
            info['train_losses'].append(train_loss)
            info['train_cnts'].append(train_cnt)
            vx,vy,_ = data_loader.validation_data()
            vloss_1,vloss_2,vloss_3,vx_d,vz_e_x,vz_q_x,vlatents = forward_pass(vx,vy)
            test_loss = np.array(to_scalar([vloss_1, vloss_2, vloss_3])).mean(0)
            info['test_losses'].append(test_loss)
            info['test_cnts'].append(train_cnt)

            print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                                      info['train_losses'][-1], info['test_losses'][-1]))
            rolling = 3
            if do_plot and  len(info['train_losses'])>rolling*3:
                info['last_plot'] = train_cnt
                plot_name = os.path.join(default_base_savedir, basename + "_%010dloss.png"%train_cnt)
                print('plotting: %s with %s points'%(plot_name, len(info['train_cnts'])))
                plot_losses(info['train_cnts'],
                            info['train_losses'],
                            info['test_cnts'],
                            info['test_losses'], name=plot_name, rolling_length=rolling)

        if ((train_cnt-info['last_save'])>=targs.save_every):
            info['last_save'] = train_cnt
            info['save_times'].append(time.time())
            handle_plot_ckpt(True)
            filename = os.path.join(default_base_savedir , basename + "_%010dex.pkl"%train_cnt)
            state = {
                     'state_dict':vmodel.state_dict(),
                     'optimizer':opt.state_dict(),
                     'info':info,
                     }
            save_checkpoint(state, filename=filename)
        elif not train_cnt or (train_cnt-info['train_cnts'][-1])>=targs.log_every:
            handle_plot_ckpt(False)
        else:
            if (train_cnt-info['last_plot'])>=targs.plot_every:
                handle_plot_ckpt(True)

        train_cnt += data_loader.batch_size

    filename = os.path.join(default_base_savedir , basename + "_%010dex.pkl"%train_cnt)
    info['last_save'] = train_cnt
    info['save_times'].append(time.time())
    handle_plot_ckpt(True)
    state = {
             'state_dict':vmodel.state_dict(),
             'optimizer':opt.state_dict(),
             'info':info,
             }
    save_checkpoint(state, filename=filename)

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

def get_one_step_prediction(vmodel,x,y,batch_idx):
    x_d, z_e_x, z_q_x, latents = vmodel(x.to(DEVICE))
    x_tilde = sample_from_discretized_mix_logistic(x_d, args.nr_logistic_mix)
    # input x is between 0 and 1
    pred_arr =  np.array(x_tilde.cpu().data)[:,0]
    pred = (pred_arr+1.0)/2.0
    return pred

def generate_output_dataset(model_loadname, vmodel, dataloader):
    print("generating output dataset for: %s" %model_loadname)
    oh,ow = 80,80
    fs = 1
    all_test_output = np.zeros((max(dataloader.test_loader.index_array)+fs,oh,ow))
    all_train_output = np.zeros((max(dataloader.train_loader.index_array)+fs,oh,ow))

    while not dataloader.done:
        x,y,batch_idx = dataloader.ordered_batch()
        if x.shape[0]:
            pred = get_one_step_prediction(vmodel,x,y,batch_idx)
            all_train_output[batch_idx+fs] = pred

    while not dataloader.test_done:
        x,y,batch_idx = dataloader.validation_ordered_batch()
        if x.shape[0]:
            pred = get_one_step_prediction(vmodel,x,y,batch_idx)
            # prediction from one step ahead
            all_test_output[batch_idx+fs] = pred

    output_train_data_file = train_data_file.replace('.npz', model_loadname) + '.npz'
    output_test_data_file = test_data_file.replace('.npz', model_loadname) + '.npz'

    np.savez(output_train_data_file, all_train_output)
    np.savez(output_test_data_file, all_test_output)

    mimsave(output_train_data_file.replace('.npz','.gif'), all_train_output[:100])
    mimsave(output_test_data_file.replace('.npz','.gif'), all_test_output)
    print('wrote dataset to: %s'%output_train_data_file)
    return output_train_data_file, output_test_data_file


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--model_savename', default='dag')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=600000, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000, type=int)
    parser.add_argument('-le', '--log_every', default=10000, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-nc', '--number_condition', default=4, type=int)
    parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=500000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    args = parser.parse_args()
    use_cuda = args.cuda
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    if args.log_every > int(args.plot_every/3):
        args.log_every = int(args.plot_every/3)
        print("forcing increase in log every to: %s" %args.log_every)
    # original unaugmented data files
    train_data_file = os.path.join(config.base_datadir, 'freeway_train_01000.npz')
    test_data_file = os.path.join(config.base_datadir, 'freeway_test_00300.npz')
    aug_train_data_file, aug_test_data_file = "None", "None"

    for aug_num in range(10):
        info = {'train_cnts':[],
                'train_losses':[],
                'test_cnts':[],
                'test_losses':[],
                'save_times':[],
                'args':[args],
                'last_save':0,
                'last_plot':0,
                'aug_num':aug_num,
                }
        # create new model
        vmodel = AutoEncoder(nr_logistic_mix=args.nr_logistic_mix,
                                 num_clusters=args.num_k, encoder_output_size=args.num_z,
                                 in_channels_size=args.number_condition, out_channels_size=1).to(DEVICE)
        opt = torch.optim.Adam(vmodel.parameters(), lr=args.learning_rate)

        basename = 'f%s%s_AUG%02d_k%sz%sc%df%02d'%(vmodel.name,
                               args.model_savename,
                               aug_num,
                               args.num_k, args.num_z,
                               args.number_condition,
                               args.steps_ahead,
                               )

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

        train()
        # output next iterations augment file
        aug_train_data_file, aug_test_data_file = generate_output_dataset(basename, vmodel, data_loader)


