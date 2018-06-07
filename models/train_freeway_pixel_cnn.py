import matplotlib
matplotlib.use('Agg')
import config
import matplotlib.pyplot as plt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from vqvae import AutoEncoder
import sys
import shutil
import torch
from IPython import embed
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
#from vqvae import AutoEncoder, to_scalar
#from vqvae_small import AutoEncoder, to_scalar
from pixel_cnn import GatedPixelCNN
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
from datasets import EpisodicVqVaeFroggerDataset, FroggerDataset
max_pixel = 255.0
min_pixel = 0.0

def train(epoch,train_loader,DEVICE,history_size):
    print("starting epoch {}".format(epoch))
    train_loss = []
    for batch_idx, (data, dname) in enumerate(train_loader):
        # loaded data is latent codes from a vqvae
        x = Variable(data, requires_grad=False).to(DEVICE)
        # episodic dta looks like (batch_size, frames, 6, 6)
        episode_length = x.shape[1]
        for i in range(history_size, episode_length):
            # NoneiNone because we have only 1 channel
            start_time = time.time()
            opt.zero_grad()
            i_x = x[:,i].contiguous()
            cond_x = x[:,(i-history_size):i]
            o = pcnn_model(i_x, spatial_cond=cond_x)
            logits = o.permute(0,2,3,1).contiguous()
            loss = criterion(logits.view(-1,num_clusters), i_x.view(-1))
            loss.backward()
            opt.step()
            train_loss.append(float(loss.cpu().data.numpy()))
            if not i%50:
                print 'Train Epoch: {} frame: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    epoch, i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / float(len(train_loader)),
                    np.mean(train_loss),
                    time.time() - start_time
                     )
    return np.mean(train_loss)


def test(epoch,test_loader,DEVICE,history_size,save_img_path=None):
    test_loss = []
    for batch_idx, (data, dname) in enumerate(test_loader):
        dshape = tuple(data.shape[2:])
        start_time = time.time()
        # loaded data is latent codes from a vqvae
        x = Variable(data, requires_grad=False).to(DEVICE)
        # episodic data looks like (batch_size, frames, 6, 6)
        episode_length = x.shape[1]
        for i in range(history_size, episode_length):
            # NoneiNone because we have only 1 channel
            i_x = x[:,i].contiguous()
            cond_x = x[:,(i-history_size):i]
            o = pcnn_model(i_x, spatial_cond=cond_x)
            logits = o.permute(0,2,3,1).contiguous()
            loss = criterion(logits.view(-1,num_clusters), i_x.view(-1))
            test_loss.append(float(loss.cpu().data.numpy()))
        if not batch_idx%10:
            print 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / float(len(test_loader)),
                np.mean(test_loss),
                time.time() - start_time
            )
    ## make image from the last example
    ep = 0
    ep_name = os.path.split(dname[ep])[1].replace('_seq.npz', '.png')
    # need to get exact name
    orig_img_path = os.path.join(base_orig_name, ep_name)
    frame_save_path = os.path.abspath(os.path.join(config.model_savedir, 'vqpcnn_e%04d.png'%epoch))
    # dshape was (6,6) for dyanmic env
    gen_latents = pcnn_model.generate(spatial_cond=cond_x[ep][None],
                                   shape=dshape,batch_size=1)
    generate(i, gen_latents, orig_img_path=orig_img_path, save_img_path=frame_save_path, dshape=dshape)
    return np.mean(test_loss)


def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def generate(frame_num, gen_latents, orig_img_path, save_img_path, dshape):
    z_q_x = vmodel.embedding(gen_latents.view(gen_latents.size(0),-1))
    z_q_x = z_q_x.view(gen_latents.shape[0],dshape[0], dshape[1],-1).permute(0,3,1,2)
    x_d = vmodel.decoder(z_q_x)
    if save_img_path is not None:
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        pred = (((np.array(x_tilde.cpu().data)[0,0]+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
        # input x is between 0 and 1
        real = imread(orig_img_path)
        f, ax = plt.subplots(1,3, figsize=(10,3))
        ax[0].imshow(real,cmap='gray',  vmin=0, vmax=max_pixel)
        ax[0].set_title("original frame %s"%frame_num)
        ax[1].imshow(pred, cmap='gray', vmin=0, vmax=max_pixel)
        ax[1].set_title("pred")
        ax[2].imshow((pred-real)**2, cmap='gray',  vmin=0, vmax=max_pixel)
        ax[2].set_title("error")
        f.tight_layout()
        plt.savefig(save_img_path)
        plt.close()
        print("saving example image")
        print("rsync -avhp jhansen@erehwon.cim.mcgill.ca://%s" %os.path.abspath(save_img_path))

#
if __name__ == '__main__':
    import argparse
    base_savedir = '../../../trajectories_frames/saved/vqvae'
    vqvae_model_loadname = 'freeway_vqvae4layer_nl_k512_z64e00181.pkl'
    vqvae_dataset = 'freeway_vqvae4layer_nl_k512_z64_e00182'
    num_clusters = 512
    num_z = 64
    parser = argparse.ArgumentParser(description='train pixel-cnn on vqvae latents')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--model_savename', default='_')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=1, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-p', '--port', default=8097, type=int, help='8097 for erehwon 8096 for numenor by default')
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    train_data_dir = os.path.join(config.base_datadir, 'train_'+vqvae_dataset)
    test_data_dir =  os.path.join(config.base_datadir, 'test_'+ vqvae_dataset)
    use_cuda = args.cuda
    #opt = torch.optim.Adam(vmodel.parameters(), lr=1e-3)
    #train_loss_list = []
    #test_loss_list = []
    #epochs = []
    #epoch = 1

    N_LAYERS = 15
    DIM = 256
    history_size = 4
    cond_size = history_size*DIM
    if use_cuda:
        print("using cuda")
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    pcnn_model = GatedPixelCNN(num_clusters, DIM, N_LAYERS, history_size, spatial_cond_size=cond_size).to(DEVICE)
    criterion = CrossEntropyLoss().to(DEVICE)

    opt = torch.optim.Adam(pcnn_model.parameters(), lr=3e-4, amsgrad=True)
    train_loss_list = []
    test_loss_list = []
    epochs = []
    epoch = 1

    basename = 'm%s_%s_k%s_z%s'%(pcnn_model.name, args.model_savename,
                                        num_clusters, num_z)
    port = args.port
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': """%s
Train Loss"""%basename})

    test_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': """%s
Test Loss"""%basename})


    if args.model_loadname is not None:
        model_loadpath = os.path.abspath(os.path.join(config.model_savedir, args.model_loadname))
        if os.path.exists(model_loadpath):
            model_dict = torch.load(model_loadpath)
            pcnn_model.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epochs.extend(model_dict['epochs'])
            train_loss_list.extend(model_dict['train_losses'])
            test_loss_list.extend(model_dict['test_losses'])
            for e, tr, te in zip(epochs, train_loss_list, test_loss_list):
                train_loss_logger.log(e, np.sum(tr))
                test_loss_logger.log(e, np.sum(te))
            epoch = epochs[-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, model_loadpath))
            epoch = epochs[-1]+1
        else:
            print('could not find checkpoint at {}'.format(model_loadpath))
            embed()
    else:
        print('created new model')

    nr_logistic_mix = 10
    num_channels = 1
    vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, num_clusters=num_clusters,
                         encoder_output_size=num_z, data_channels_size=num_channels).to(DEVICE)

    vqvae_model_loadpath = os.path.join(config.model_savedir, vqvae_model_loadname)

    if not os.path.exists(vqvae_model_loadpath):
        print("must give valid model! %s does not exist!"%vqvae_model_loadpath)
        embed()
    else:
        vqvae_model_dict = torch.load(vqvae_model_loadpath)
        vmodel.load_state_dict(vqvae_model_dict['state_dict'])
        vqvae_last_epoch = vqvae_model_dict['epochs'][-1]
        print('loaded vqvae model from epoch %s'%vqvae_last_epoch)


    base_orig_name = os.path.join(config.freeway_test_frames_dir)
    base_save_name = os.path.join(config.base_datadir, 'test_'+basename+'e%05d'%epoch)

    if not args.generate_results:
        data_test_loader = DataLoader(EpisodicVqVaeFroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor(), search='freeway*.npz'),
                                      batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, num_workers=4)
        data_train_loader = DataLoader(EpisodicVqVaeFroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(), limit=args.num_train_limit, search='freeway*.npz'),
                                       batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=4)

        for e in xrange(epoch,epoch+args.num_epochs):
            train_loss = train(e,data_train_loader,DEVICE,history_size=history_size)

            test_img_name = os.path.join(config.model_savedir , basename + "e%05d.png"%e)
            filename = os.path.join(config.model_savedir , basename + "e%05d.pkl"%e)
            epochs.append(e)
            test_loss = test(e,data_test_loader,DEVICE,history_size=history_size,save_img_path=None)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            print('send data to plotter')
            train_loss_logger.log(e, np.sum(train_loss_list[-1]))
            test_loss_logger.log(e,  np.sum(test_loss_list[-1]))
            print('train_loss {} test_loss {}'.format(train_loss,test_loss))
            print('train_loss sum {} test_loss sum {}'.format(np.sum(train_loss),np.sum(test_loss)))

            if (not e%args.save_every) or (e==epoch+args.num_epochs):
                print('------------------------------------------------------------')
                state = {'epoch':e,
                         'epochs':epochs,
                         'state_dict':pcnn_model.state_dict(),
                         'train_losses':train_loss_list,
                         'test_losses':test_loss_list,
                         'optimizer':opt.state_dict(),
                         }

                save_checkpoint(state, filename=filename)
    else:
        #data_train_loader = DataLoader(FroggerDataset(train_data_dir,
        #                               transform=transforms.ToTensor(), limit=args.num_train_limit),
        #                               batch_size=episode_length, shuffle=False)
        #data_test_loader = DataLoader(FroggerDataset(test_data_dir,
        #                              transform=transforms.ToTensor()),
        #                              batch_size=episode_length, shuffle=False)

        atype = 'test'
        data_test_loader = DataLoader(EpisodicVqVaeFroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor(), max_pixel=255.0, min_pixel=0.0),
                                      batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
        if not os.path.exists(base_save_name):
            os.makedirs(base_save_name)
        for batch_idx, (data, dname) in enumerate(data_test_loader):
            x = Variable(data, requires_grad=False).to(DEVICE)
            batch_size, episode_length, h, w = x.shape
            for i in range(history_size, episode_length):
                ep_name = os.path.split(dname[0])[1].replace('.npz', '_frame_%05d.png'%i)
                orig_img_path = os.path.abspath(os.path.join(base_orig_name, ep_name))
                frame_save_path = os.path.abspath(os.path.join(base_save_name, ep_name.replace('.png', '_pcnn_vqvae.png')))
                cond_x = x[:,(i-history_size):i]
                latent_shape=(6,6)
                gen_latents = pcnn_model.generate(spatial_cond=cond_x,
                                                  shape=latent_shape,batch_size=batch_size)
                generate(i, gen_latents, orig_img_path=orig_img_path,
                                              save_img_path=frame_save_path)
            break




        #test_gen_dir = os.path.join(args.datadir, 'test_'  + basename+'_e%05d'%epoch)
        #train_gen_dir = os.path.join(args.datadir, 'train_'+ basename+'_e%05d'%epoch)

        #generate_episodic_npz(data_test_loader,use_cuda,train_gen_dir)
        #generate_episodic_npz(data_train_loader,use_cuda,train_gen_dir)



