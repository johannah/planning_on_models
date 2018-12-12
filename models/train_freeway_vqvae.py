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
from imageio import imread, imwrite
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
    for i in range(targs.num_examples_to_train):
        x,y = data_loader.next_batch()
        opt.zero_grad()
        loss_1,loss_2,loss_3,x_d,z_e_x,z_q_x,latents = forward_pass(x,y)
        loss_1.backward(retain_graph=True)
        vmodel.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        loss_2.backward(retain_graph=True)
        loss_3.backward()
        opt.step()

        def handle_plot_ckpt():
            train_loss = np.array(to_scalar([loss_1, loss_2, loss_3])).mean(0)
            info['train_losses'].append(train_loss)
            info['train_cnts'].append(train_cnt)
            vx,vy = data_loader.validation_data()
            vloss_1,vloss_2,vloss_3,vx_d,vz_e_x,vz_q_x,vlatents = forward_pass(vx,vy)
            test_loss = np.array(to_scalar([vloss_1, vloss_2, vloss_3])).mean(0)
            info['test_losses'].append(test_loss)
            info['test_cnts'].append(train_cnt)

            plot_name = os.path.join(default_base_savedir, basename + "loss%05d.png"%train_cnt)
            n = 10
            plot_losses(rolling_average(info['train_cnts'], n),
                        rolling_average(info['train_losses'], n),
                        rolling_average(info['test_cnts'], n),
                        rolling_average(info['test_losses'], n), name=plot_name)

            print('plotting: %s'%plot_name)
            print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                                  info['train_losses'][-1], info['test_losses'][-1]))

            info['last_plot'] = train_cnt

        if (train_cnt-info['last_save'])>targs.save_every:
            info['last_save'] = train_cnt
            info['save_times'] = time.time()
            handle_plot_ckpt()
            filename = os.path.join(default_base_savedir , basename + "ex%05d.pkl"%train_cnt)
            state = {
                     'state_dict':vmodel.state_dict(),
                     'optimizer':opt.state_dict(),
                     'info':info,
                     }
            save_checkpoint(state, filename=filename)
        else:
            if (train_cnt-info['last_plot'])>targs.plot_every:
                handle_plot_ckpt()

        train_cnt += data_loader.batch_size

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

def rolling_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--model_savename', default='fp')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=10000, type=int)
    parser.add_argument('-pe', '--plot_every', default=5000, type=int)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-k', '--num_k', default=256, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=1000000, type=int)
    args = parser.parse_args()
    use_cuda = args.cuda
    nr_logistic_mix = 10
    num_clusters = args.num_k
    learning_rate = 1e-4
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,
                         num_clusters=num_clusters, encoder_output_size=args.num_z,
                         in_channels_size=4, out_channels_size=1).to(DEVICE)
    opt = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)

    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':{0:args},
            'last_save':0,
            'last_plot':-args.plot_every,
            }

    basename = 'f%s_%s_'%(vmodel.name,
                           args.model_savename,
                           )
    if args.model_loadname is not None:
        model_loadpath = os.path.abspath(os.path.join(default_base_savedir, args.model_loadname))
        if os.path.exists(model_loadpath):
            model_dict = torch.load(model_loadpath)
            vmodel.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            info = model_dict['info']
            info['args'][train_cnt+1] = args
            targs.append(args)
            print('loaded checkpoint from {}'.format(model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(model_loadpath))
            embed()


    train_data_file = os.path.join(config.base_datadir, 'freeway_train_00500.npz')
    test_data_file = os.path.join(config.base_datadir, 'freeway_test_00150.npz')
    train_data_function = FreewayForwardDataset(
                                   train_data_file,
                                   number_condition=4,
                                   steps_ahead=1,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel)
    test_data_function = FreewayForwardDataset(
                                   test_data_file,
                                   number_condition=4,
                                   steps_ahead=1,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel)

    data_loader = DataLoader(train_data_function, test_data_function,
                                   batch_size=32)

    train()


#def test(epoch,test_loader,save_img_path=None):
#    test_loss = []
#    for batch_idx, (data, _) in enumerate(test_loader):
#        start_time = time.time()
#        x = Variable(data, requires_grad=False).to(DEVICE)
#        x_d, z_e_x, z_q_x, latents = vmodel(x)
#        loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,DEVICE=DEVICE)
#        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
#        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
#        test_loss.append(to_scalar([loss_1, loss_2, loss_3]))
#    test_loss_mean = np.asarray(test_loss).mean(0)
#    if save_img_path is not None:
#        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#        idx = 0
#        x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
#        images = x_cat.cpu().data
#        #pred = (((np.array(x_tilde.cpu().data)[0,0]+1.)/2.0)*float(max_pixel-min_pixel)) + min_pixel
#        p =  np.array(x_tilde.cpu().data)[0,0]
#        pred = ((((p+1.0)/2.0)+0.5)*float(config.freeway_max_pixel-config.freeway_min_pixel)) + config.freeway_min_pixel
#
#        # input x is between 0 and 1
#        real = ((np.array(x.cpu().data)[0,0]+0.5)*float(config.freeway_max_pixel-config.freeway_min_pixel))+config.freeway_min_pixel
#        print("real", real.min(), real.max())
#        print("pred", pred.min(), pred.max())
#        f, ax = plt.subplots(1,3, figsize=(10,3))
#        ax[0].imshow(real, vmin=0, vmax=config.freeway_max_pixel, cmap=plt.cm.gray)
#        ax[0].set_title("original")
#        ax[1].imshow(pred, vmin=0, vmax=config.freeway_max_pixel, cmap=plt.cm.gray)
#        ax[1].set_title("pred epoch %s test loss %s" %(epoch,np.mean(test_loss_mean)))
#        ax[2].imshow((pred-real)**2, cmap='gray')
#        ax[2].set_title("error")
#        f.tight_layout()
#        plt.savefig(save_img_path)
#        plt.close()
#        print("saving example image")
#        print("rsync -avhp jhansen@erehwon.cim.mcgill.ca://%s" %os.path.abspath(save_img_path))
#
#    return test_loss_mean
#


#def generate_episodic_npz(data_loader,save_path,make_imgs=False):
#    print("saving to", save_path)
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    for batch_idx, (data, fpaths) in enumerate(data_loader):
#        # batch idx must be exactly one episode
#        #assert np.sum([fpaths[0][:-10] == f[:-10]  for f in fpaths]) == len(fpaths)
#            start_time = time.time()
#            x = Variable(data, requires_grad=False).to(DEVICE)
#            for st in range(skip_frames):
#                nam = os.path.split(fpaths[st])[1].replace('.png', '_seq.npz')
#                episode_path = os.path.join(save_path,nam)
#                frames = range(st, data.shape[0], skip_frames)
#                if len(frames) == batch_size:
#                    if not os.path.exists(episode_path):
#                        print("episode: %s length: %s" %(episode_path, len(frames)))
#                    A_idx = torch.LongTensor(frames).to(DEVICE) # the index vector
#                    XX = x.index_select(0, A_idx)
#                    # make batch
#                    x_d, z_e_x, z_q_x, latents = vmodel(XX)
#                    xds = x_d.cpu().data.numpy()
#                    zes = z_e_x.cpu().data.numpy()
#                    zqs = z_q_x.cpu().data.numpy()
#                    ls = latents.cpu().data.numpy()
#
#                    # split episode into chunks that are reasonable
#                    np.savez(episode_path, z_e_x=zes.astype(np.float32),
#                              z_q_x=zqs.astype(np.float32), latents=ls.astype(np.int))
#


#        if args.model_loadname is None:
#            print("must give valid model!")
#            sys.exit()
#
#        skip_frames = 4
#        batch_size = 64
#        bs = batch_size*skip_frames
#        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
#                                       transform=transforms.ToTensor(),
#                                       limit=args.num_train_limit,
#                                       max_pixel_used=max_pixel, min_pixel_used=min_pixel),
#                                       batch_size=bs, shuffle=False)
#        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
#                                      transform=transforms.ToTensor(),
#                                      max_pixel_used=max_pixel, min_pixel_used=min_pixel),
#                                      batch_size=bs, shuffle=False)
#
#        basedatadir = '../../dataset'
#        test_gen_dir =  os.path.join(basedatadir, 'test_' + basename+'_e%05d'%epoch)
#        train_gen_dir = os.path.join(basedatadir, 'train_'+ basename+'_e%05d'%epoch)
#
#        generate_episodic_npz(data_test_loader,test_gen_dir)
#        generate_episodic_npz(data_train_loader,train_gen_dir)
#
