import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import sys
import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
from utils import get_cuts, to_scalar
from datasets import FroggerDataset
import config
torch.manual_seed(7)

def train(epoch,train_loader):
    print("starting epoch {}".format(epoch))
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).to(DEVICE)
        opt.zero_grad()
        x_d, z_e_x, z_q_x, latents = vmodel(x)
        # with bigger model - latents is 64, 6, 6
        z_q_x.retain_grad()
        #loss_1 = F.binary_cross_entropy(x_d, x)
        # going into dml - x should be bt 0 and 1
        loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,DEVICE=DEVICE)
        loss_1.backward(retain_graph=True)
        vmodel.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_2.backward(retain_graph=True)
        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
        loss_3.backward()
        opt.step()
        train_loss.append(to_scalar([loss_1, loss_2, loss_3]))
        if not batch_idx%100:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )

    return np.asarray(train_loss).mean(0)

def test(epoch,test_loader,save_img_path=None):
    test_loss = []
    for batch_idx, (data, _) in enumerate(test_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).to(DEVICE)
        x_d, z_e_x, z_q_x, latents = vmodel(x)
        loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,DEVICE=DEVICE)
        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
        test_loss.append(to_scalar([loss_1, loss_2, loss_3]))
    test_loss_mean = np.asarray(test_loss).mean(0)
    if save_img_path is not None:
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        idx = 0
        x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
        images = x_cat.cpu().data
        #pred = (((np.array(x_tilde.cpu().data)[0,0]+1.)/2.0)*float(max_pixel-min_pixel)) + min_pixel
        p =  np.array(x_tilde.cpu().data)[0,0]
        pred = ((((p+1.0)/2.0)+0.5)*float(max_pixel-min_pixel)) + min_pixel

        # input x is between 0 and 1
        real = ((np.array(x.cpu().data)[0,0]+0.5)*float(max_pixel-min_pixel))+min_pixel
        print("real", real.min(), real.max())
        print("pred", pred.min(), pred.max())
        f, ax = plt.subplots(1,3, figsize=(10,3))
        ax[0].imshow(real, vmin=0, vmax=max_pixel, cmap=plt.cm.gray)
        ax[0].set_title("original")
        ax[1].imshow(pred, vmin=0, vmax=max_pixel, cmap=plt.cm.gray)
        ax[1].set_title("pred epoch %s test loss %s" %(epoch,np.mean(test_loss_mean)))
        ax[2].imshow((pred-real)**2, cmap='gray')
        ax[2].set_title("error")
        f.tight_layout()
        plt.savefig(save_img_path)
        plt.close()
        print("saving example image")
        print("rsync -avhp jhansen@erehwon.cim.mcgill.ca://%s" %os.path.abspath(save_img_path))

    return test_loss_mean


def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def generate_episodic_npz(data_loader,save_path,make_imgs=False):
    print("saving to", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for batch_idx, (data, fpaths) in enumerate(data_loader):
        # batch idx must be exactly one episode
        #assert np.sum([fpaths[0][:-10] == f[:-10]  for f in fpaths]) == len(fpaths)
            start_time = time.time()
            x = Variable(data, requires_grad=False).to(DEVICE)
            for st in range(skip_frames):
                nam = os.path.split(fpaths[st])[1].replace('.png', '_seq.npz')
                episode_path = os.path.join(save_path,nam)
                frames = range(st, data.shape[0], skip_frames)
                if len(frames) == batch_size:
                    if not os.path.exists(episode_path):
                        print("episode: %s length: %s" %(episode_path, len(frames)))
                    A_idx = torch.LongTensor(frames).to(DEVICE) # the index vector
                    XX = x.index_select(0, A_idx)
                    # make batch
                    x_d, z_e_x, z_q_x, latents = vmodel(XX)
                    xds = x_d.cpu().data.numpy()
                    zes = z_e_x.cpu().data.numpy()
                    zqs = z_q_x.cpu().data.numpy()
                    ls = latents.cpu().data.numpy()

                    # split episode into chunks that are reasonable
                    np.savez(episode_path, z_e_x=zes.astype(np.float32),
                              z_q_x=zqs.astype(np.float32), latents=ls.astype(np.int))


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='train vq-vae for freeway')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--model_savename', default='nl')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=5, type=int)
    parser.add_argument('-z', '--num_z', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', default=250, type=int)
    parser.add_argument('-p', '--port', default=8097, type=int, help='8097 for erehwon 8096 for numenor by default')
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')
    parser.add_argument('-d', '--datatype', type=str, default='freeway')
    args = parser.parse_args()
    if args.datatype == 'freeway':
        max_pixel = 254.0
        min_pixel = 0.0
        train_data_dir = config.freeway_train_frames_dir
        test_data_dir =  config.freeway_test_frames_dir
        num_channels = 1

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
                         data_channels_size=num_channels).to(DEVICE)
    opt = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)
    train_loss_list = []
    test_loss_list = []
    epochs = []
    epoch = 1

    basename = '%s_%s_%s_k%s_z%s'%(args.datatype, vmodel.name,
                                        args.model_savename,
                                        args.num_k, args.num_z,
                                        )
    port = args.port
    train_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s Train Loss'%basename})

    test_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s Test Loss'%basename})

    if args.model_loadname is not None:
        model_loadpath = os.path.abspath(os.path.join(default_base_savedir, args.model_loadname))
        if os.path.exists(model_loadpath):
            model_dict = torch.load(model_loadpath)
            vmodel.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epochs.extend(model_dict['epochs'])
            train_loss_list.extend(model_dict['train_losses'])
            test_loss_list.extend(model_dict['test_losses'])
            for e, tr, te in zip(epochs, train_loss_list, test_loss_list):
                train_loss_logger.log(e, np.sum(tr))
                test_loss_logger.log(e, np.sum(te))
            epoch = epochs[-1]+1
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(model_loadpath))
            embed()
    else:
        print('created new model')

    if not args.generate_results:
        print("starting data loader")
        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(), limit=args.num_train_limit, max_pixel_used=max_pixel, min_pixel_used=min_pixel),
                                       batch_size=64, shuffle=True)
        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor(), max_pixel_used=max_pixel, min_pixel_used=min_pixel),
                                      batch_size=64, shuffle=True)


        #    test_img = args.model_savepath.replace('.pkl', '_test.png')
        for e in xrange(epoch,epoch+args.num_epochs):
            train_loss = train(e,data_train_loader)
            test_img_name = os.path.join(default_base_savedir , basename + "e%05d.png"%e)
            test_loss = test(e,data_test_loader,save_img_path=test_img_name)
            epochs.append(e)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            print('send data to plotter')
            train_loss_logger.log(e, np.sum(train_loss_list[-1]))
            test_loss_logger.log(e,  np.sum(test_loss_list[-1]))
            print('epoch {} train_loss {} test_loss {}'.format(e, train_loss,test_loss))
            print('epoch {} train_loss sum {} test_loss sum {}'.format(e, np.sum(train_loss),np.sum(test_loss)))

            if (not e%args.save_every) or (e==epoch+args.num_epochs):
                print('------------------------------------------------------------')
                print('----------------------saving--------------------------------')
                filename = os.path.join(default_base_savedir , basename + "e%05d.pkl"%e)

                state = {'epoch':e,
                         'epochs':epochs,
                         'state_dict':vmodel.state_dict(),
                         'train_losses':train_loss_list,
                         'test_losses':test_loss_list,
                         'optimizer':opt.state_dict(),
                         }

                save_checkpoint(state, filename=filename)
    else:
        if args.model_loadname is None:
            print("must give valid model!")
            sys.exit()

        skip_frames = 4
        batch_size = 64
        bs = batch_size*skip_frames
        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(),
                                       limit=args.num_train_limit,
                                       max_pixel_used=max_pixel, min_pixel_used=min_pixel),
                                       batch_size=bs, shuffle=False)
        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor(),
                                      max_pixel_used=max_pixel, min_pixel_used=min_pixel),
                                      batch_size=bs, shuffle=False)

        basedatadir = '../../dataset'
        test_gen_dir =  os.path.join(basedatadir, 'test_' + basename+'_e%05d'%epoch)
        train_gen_dir = os.path.join(basedatadir, 'train_'+ basename+'_e%05d'%epoch)

        generate_episodic_npz(data_test_loader,test_gen_dir)
        generate_episodic_npz(data_train_loader,train_gen_dir)

