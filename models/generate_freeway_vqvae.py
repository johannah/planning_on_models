import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from imageio import imread, imwrite, mimwrite, mimsave
from PIL import Image
from ae_utils import discretized_mix_logistic_loss
from ae_utils import sample_from_discretized_mix_logistic
from ae_utils import get_cuts, to_scalar
from datasets import FreewayForwardDataset, DataLoader
from lstm_utils import plot_losses
import config
fdiff = float(config.freeway_max_pixel-config.freeway_min_pixel)
torch.manual_seed(7)


#def generate_episodic_npz(dataloader,save_path,make_imgs=False):
#    print("saving to", save_path)
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    for batch_idx, (data, fpaths) in enumerate(dataloader):
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
def generate_imgs(x,y,idxs,basepath):
    x = Variable(x, requires_grad=False).to(DEVICE)
    y = Variable(y, requires_grad=False).to(DEVICE)
    x_d, z_e_x, z_q_x, latents = vmodel(x)
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    for idx, cnt in enumerate(idxs):
        x_cat = torch.cat([y[idx], x_tilde[idx]], 0)
        images = x_cat.cpu().data
        obs_arr = np.array(x[idx,largs.number_condition-1].cpu().data)
        real_arr = np.array(y.cpu().data)[idx,0] # only 1 channel
        pred_arr =  np.array(x_tilde.cpu().data)[idx,0]
        # input x is between 0 and 1
        pred = ((((pred_arr+1.0)/2.0))*fdiff) + config.freeway_min_pixel
        obs = ((obs_arr+0.5)*fdiff)+config.freeway_min_pixel
        real = ((real_arr+0.5)*fdiff)+config.freeway_min_pixel
        f, ax = plt.subplots(1,4, figsize=(10,3))
        ax[0].imshow(obs, vmin=0, vmax=config.freeway_max_pixel)
        ax[0].set_title("obs")
        ax[1].imshow(real, vmin=0, vmax=config.freeway_max_pixel)
        ax[1].set_title("true %d steps" %largs.steps_ahead)
        ax[2].imshow(pred, vmin=0, vmax=config.freeway_max_pixel)
        ax[2].set_title("pred %d steps" %largs.steps_ahead)
        ax[3].imshow((pred-real)**2)
        ax[3].set_title("error")
        f.tight_layout()
        save_img_path = os.path.join(basepath, 'cond%02d_pred%02d_idx%06d.png'%(largs.number_condition,
                                                                              largs.steps_ahead,
                                                                              cnt))
        plt.savefig(save_img_path)
        plt.close()

def generate_rollout(x,y,batch_idx,basepath,datal,idx=4):
    # todo - change indexing
    x = Variable(x, requires_grad=False).to(DEVICE)
    y = Variable(y, requires_grad=False).to(DEVICE)
    bs,_,oh,ow = y.shape
    all_pred = np.zeros((bs,args.rollout_length,oh,ow))
    all_real = np.zeros((bs,args.rollout_length,oh,ow))
    all_obs = np.zeros((bs,oh,ow))

    last_timestep_img = x
    obs_f = largs.number_condition-1

    save_img_paths = []
    for i, idx in enumerate(batch_idx):
        # dont rollout past true frames
        max_idx = min(max(datal.index_array)-1, idx+args.rollout_length)
        # get ys from our index to the length of the rollout
        _x,_y=datal[np.arange(idx, max_idx)]
        # entire trace of true future observations
        reals = _y[:,0]#(_y[:,0]*fdiff)+config.freeway_min_pixel
        # what would have been observed at this timestep by the agent
        all_real[i,:_y.shape[0]] = reals.cpu().data

    for rs in range(args.rollout_length):
        print('rs', rs)
        x_d, z_e_x, z_q_x, latents = vmodel(last_timestep_img)
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        # input x is between 0 and 1
        pred_arr =  np.array(x_tilde.cpu().data)[:,0]
        pred = (pred_arr+1.0)/2.0
        # put in real one for sanity check
        # unscale
        all_pred[:,rs] = pred
        #all_pred[:,rs] = fake_pred
        # zero out oldest timestep
        last_timestep_img[:,0] *=0.0
        # force correct new obs
        # somewhere right here is where it is messed up
        # if i sample from the data set - it works, if i use below two lines
        # with what I think is the correct answer, it failes
        my_step = Variable(torch.FloatTensor(pred), requires_grad=False).to(DEVICE)[:,None]
        #this_x, this_y = datal[batch_idx+rs+1]
        #real_step = this_x[:,3][:,None]
        #real_rollout = all_real[:,rs][:,None]
        this_step = Variable(torch.FloatTensor(my_step), requires_grad=False).to(DEVICE)
        last_timestep_img = torch.cat((last_timestep_img[:,1:],this_step), dim=1)
        #print('truevsmine')
        #print(x.max(), pred.max(), real_step.max(), real_rollout.max())
        #print(x.min(), pred.min(), real_step.min(), real_rollout.min())


    for i, idx in enumerate(batch_idx):
        for rs in range(args.rollout_length):
            f, ax = plt.subplots(1,4, figsize=(12,3))
            pred = all_pred[i,rs]
            real = all_real[i,rs]
            error = (pred-real)**2
            # observation is last frame of x for this index
            ax[0].imshow(x[i,-1])# vmin=0, vmax=config.freeway_max_pixel)
            ax[0].set_title("obs")
            ax[1].imshow(real,)# vmin=0, vmax=config.freeway_max_pixel)
            ax[1].set_title("true t+%d steps" %(rs+1))
            ax[2].imshow(pred,)# vmin=0, vmax=config.freeway_max_pixel)
            ax[2].set_title("rollout t+%d steps" %(rs+1))
            ax[3].imshow(error)
            ax[3].set_title("error")
            f.tight_layout()
            save_img_path = os.path.join(basepath,
                              'cond%02d_pred%02d_bidx%d_r%02d.png'%(largs.number_condition,
                                                         largs.steps_ahead,
                                                         idx, rs))
            plt.savefig(save_img_path)
            plt.close()
        save_img_paths.append(save_img_path.replace('.png','')[:-4])
    return save_img_paths

def get_one_step_prediction(x,y,batch_idx):
    print(x.shape)
    x_d, z_e_x, z_q_x, latents = vmodel(x.to(DEVICE))
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    # input x is between 0 and 1
    pred_arr =  np.array(x_tilde.cpu().data)[:,0]
    pred = (pred_arr+1.0)/2.0
    return pred

def generate_output_dataset():
    oh,ow = 80,80
    fs = 1
    all_test_output = np.zeros((max(dataloader.test_loader.index_array)+fs,oh,ow))
    all_train_output = np.zeros((max(dataloader.train_loader.index_array)+fs,oh,ow))

    while not dataloader.done:
        x,y,batch_idx = dataloader.ordered_batch()
        if x.shape[0]:
            pred = get_one_step_prediction(x,y,batch_idx)
            all_train_output[batch_idx+fs] = pred

    while not dataloader.test_done:
        x,y,batch_idx = dataloader.validation_ordered_batch()
        if x.shape[0]:
            pred = get_one_step_prediction(x,y,batch_idx)
            # prediction from one step ahead
            all_test_output[batch_idx+fs] = pred

    mname = os.path.split(args.model_loadname)[1]
    output_train_data_file = train_data_file.replace('.pkl', mname)
    output_test_data_file = test_data_file.replace('.pkl', mname)
    np.savez(output_train_data_file+'.npz', all_train_output)
    np.savez(output_test_data_file+'.npz', all_test_output)

    mimsave(output_train_data_file+'.gif', all_train_output[:100])
    mimsave(output_test_data_file+'.gif', all_test_output)

def create_gif_script(name, search='*.png', run=False):
    rname = name+'.sh'
    gname = name+'.gif'

    gg = open(rname, 'w')
    gg.write('convert %s %s\n' %(search, gname))
    gg.close()
    print('creating script', rname)
    if run:
        print('creating gif', gname)
        os.system('sh %s'%rname)

def teacher_force():
    for d in [base_img_path_train, base_img_path_test]:
        if os.path.exists(d):
           shutil.rmtree(d)
        os.makedirs(d)

    if not args.train_only:
        vdone = False
        cnt = 0
        while not vdone:
            vx,vy,vbatch_idx = dataloader.validation_ordered_batch()
            if dataloader.test_done:
                done = True
            elif vbatch_idx.max() > args.max_generations:
                 vdone = True
            else:
               search = generate_imgs(vx,vy,vbatch_idx,base_img_path_test)
               cnt+=vx.shape[0]
        name = '00_gengif_cond%02d_pred%02d_cnt%03d'%(
                                                         largs.number_condition,
                                                         largs.steps_ahead,
                                                         cnt
                                                         )

        create_gif_script(base_img_path_train, name, search, run=args.generate_gif)

    if not args.test_only:
        done = False
        cnt = 0
        while not done:
            x,y,batch_idx = dataloader.ordered_batch()
            if dataloader.done:
                done = True
            elif batch_idx.max() > args.max_generations:
                done = True
            else:
               generate_imgs(x,y,batch_idx,base_img_path_train)
               cnt+=vx.shape[0]
        name = '0cond%02d_pred%02d_cnt%03d'%(
                                                         largs.number_condition,
                                                         largs.steps_ahead,
                                                         cnt
                                                         )

        create_gif_script(base_img_path_test, name, search, run=args.generate_gif)

def manage_rollout():
    for d in [rollout_base_img_path_train, rollout_base_img_path_test]:
        if os.path.exists(d):
           shutil.rmtree(d)
        os.makedirs(d)

    # test
    if not args.train_only:
       vx,vy,vbatch_idx = dataloader.validation_data()
       spaths = generate_rollout(vx,vy,vbatch_idx,rollout_base_img_path_test,dataloader.test_loader)
       for spath in spaths:
           f0,f1 = os.path.split(spath)
           name = os.path.join(f0,'0c'+f1+'rs'+ str(args.rollout_length))
           create_gif_script(name, spath+'*.png', run=args.generate_gif)


    # train
    if not args.test_only:
        tx,ty,tbatch_idx = dataloader.next_batch()
        spaths = generate_rollout(tx,ty,tbatch_idx,rollout_base_img_path_train,dataloader.train_loader)
        for spath in spaths:
            f0,f1 = os.path.split(spath)
            name = os.path.join(f0,'0c'+f1+'rs'+ str(args.rollout_length))
            create_gif_script(name, spath+'*.png', run=args.generate_gif)


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae for freeway')
    parser.add_argument('-l', '--model_loadname')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--generate_savename', default='g')
    parser.add_argument('-bs', '--batch_size', default=5, type=int)
    parser.add_argument('-n', '--max_generations', default=70, type=int)
    parser.add_argument('-gg', '--generate_gif', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('-r', '--rollout_length', default=0, type=int)
    args = parser.parse_args()
    use_cuda = args.cuda
    nr_logistic_mix = 10
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    model_loadpath = os.path.abspath(os.path.join(default_base_savedir, args.model_loadname))
    if os.path.exists(model_loadpath):
        # load model and state info
        model_dict = torch.load(model_loadpath, map_location=lambda storage, loc: storage)
        # load training arguments
        info = model_dict['info']
        largs = info['args'][-1]
        vmodel = AutoEncoder(nr_logistic_mix=largs.nr_logistic_mix,
                         num_clusters=largs.num_k, encoder_output_size=largs.num_z,
                         in_channels_size=largs.number_condition, out_channels_size=1).to(DEVICE)
        vmodel.load_state_dict(model_dict['state_dict'])
        info = model_dict['info']
        base_img_path = model_loadpath.replace('.pkl', '')
        print('loaded checkpoint from {}'.format(model_loadpath))
    else:
        print('could not find checkpoint at {}'.format(model_loadpath))
        embed()

    base_img_path_train = os.path.join(base_img_path, 'generate_train')
    base_img_path_test = os.path.join(base_img_path, 'generate_test')

    rollout_base_img_path_train = os.path.join(base_img_path, 'rollout_train_%02d'%args.rollout_length)
    rollout_base_img_path_test = os.path.join(base_img_path, 'rollout_test_%02d' %args.rollout_length)

    train_data_file = os.path.join(config.base_datadir, 'freeway_train_00500.npz')
    test_data_file = os.path.join(config.base_datadir, 'freeway_test_00150.npz')
    train_data_function = FreewayForwardDataset(
                                   train_data_file,
                                   number_condition=largs.number_condition,
                                   steps_ahead=largs.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   )
    test_data_function = FreewayForwardDataset(
                                   test_data_file,
                                   number_condition=largs.number_condition,
                                   steps_ahead=largs.steps_ahead,
                                   max_pixel_used=config.freeway_max_pixel,
                                   min_pixel_used=config.freeway_min_pixel,
                                   )

    dataloader = DataLoader(train_data_function, test_data_function,
                             batch_size=args.batch_size,
                             )

    if args.rollout_length > 0:
        kind='idx'
    else:
        kind='r%d'%args.rollout_length

    #teacher_force()
    manage_rollout()
    #generate_output_dataset()

