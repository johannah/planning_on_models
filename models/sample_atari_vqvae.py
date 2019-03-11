import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import torch
from IPython import embed
from vqvae import VQVAE
import numpy as np
from imageio import imread, imwrite, mimwrite, mimsave
from ae_utils import sample_from_discretized_mix_logistic
from datasets import AtariDataset
from train_atari_action_vqvae import forward_pass
import config
torch.manual_seed(394)

def sample_batch(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        for i in range(states.shape[0]):
            fpr = forward_pass(vmodel, states[i:i+1], next_states[i:i+1], actions[i:i+1], nr_logistic_mix=largs.nr_logistic_mix, train=False, device=DEVICE, beta=largs.beta)
            x_d, z_e_x, z_q_x, latents, avg_loss_1, avg_loss_2, avg_loss_3 = fpr
            yhat = sample_from_discretized_mix_logistic(x_d, largs.nr_logistic_mix)
            true = (2*states[i:i+1,-1:])-1
            f,ax = plt.subplots(1,2)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
            title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i], rewards[i])
            ax[0].imshow(true[0,0])
            ax[0].set_title('true')
            ax[1].imshow(yhat[0,0])
            ax[1].set_title('est')
            plt.suptitle(title)
            plt.savefig(iname)
            print('saving', iname)
        search_path = iname[:-10:] + '*.png'
        gif_path = iname[:-10:] + '.gif'
        cmd = 'convert %s %s' %(search_path, gif_path)
        print('creating gif', gif_path)
        os.system(cmd)



#
#
#
#def generate_imgs(x,y,idxs,basepath):
#    x = Variable(x, requires_grad=False).to(DEVICE)
#    y = Variable(y, requires_grad=False).to(DEVICE)
#    x_d, z_e_x, z_q_x, latents = vmodel(x)
#    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#    for idx, cnt in enumerate(idxs):
#        x_cat = torch.cat([y[idx], x_tilde[idx]], 0)
#        images = x_cat.cpu().data
#        obs_arr = np.array(x[idx,largs.number_condition-1].cpu().data)
#        real_arr = np.array(y.cpu().data)[idx,0] # only 1 channel
#        pred_arr =  np.array(x_tilde.cpu().data)[idx,0]
#        # input x is between 0 and 1
#        pred = ((((pred_arr+1.0)/2.0))*fdiff) + config.freeway_min_pixel
#        obs = ((obs_arr+0.5)*fdiff)+config.freeway_min_pixel
#        real = ((real_arr+0.5)*fdiff)+config.freeway_min_pixel
#        f, ax = plt.subplots(1,4, figsize=(10,3))
#        ax[0].imshow(obs, vmin=0, vmax=config.freeway_max_pixel)
#        ax[0].set_title("obs")
#        ax[1].imshow(real, vmin=0, vmax=config.freeway_max_pixel)
#        ax[1].set_title("true %d steps" %largs.steps_ahead)
#        ax[2].imshow(pred, vmin=0, vmax=config.freeway_max_pixel)
#        ax[2].set_title("pred %d steps" %largs.steps_ahead)
#        ax[3].imshow((pred-real)**2)
#        ax[3].set_title("error")
#        f.tight_layout()
#        save_img_path = os.path.join(basepath, 'cond%02d_pred%02d_idx%06d.png'%(largs.number_condition,
#                                                                              largs.steps_ahead,
#                                                                              cnt))
#        plt.savefig(save_img_path)
#        plt.close()
#
#def generate_rollout(x,y,batch_idx,basepath,datal,idx=4):
#    # todo - change indexing
#    x = Variable(x, requires_grad=False).to(DEVICE)
#    y = Variable(y, requires_grad=False).to(DEVICE)
#    bs,_,oh,ow = y.shape
#    all_pred = np.zeros((bs,args.rollout_length,oh,ow))
#    all_real = np.zeros((bs,args.rollout_length,oh,ow))
#    all_obs = np.zeros((bs,oh,ow))
#
#    last_timestep_img = x
#    obs_f = largs.number_condition-1
#
#    save_img_paths = []
#    for i, idx in enumerate(batch_idx):
#        # dont rollout past true frames
#        max_idx = min(max(datal.index_array)-1, idx+args.rollout_length)
#        # get ys from our index to the length of the rollout
#        _x,_y=datal[np.arange(idx, max_idx)]
#        # entire trace of true future observations
#        reals = _y[:,0]#(_y[:,0]*fdiff)+config.freeway_min_pixel
#        # what would have been observed at this timestep by the agent
#        all_real[i,:_y.shape[0]] = reals.cpu().data
#
#    for rs in range(args.rollout_length):
#        print('rs', rs)
#        x_d, z_e_x, z_q_x, latents = vmodel(last_timestep_img)
#        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#        # input x is between 0 and 1
#        pred_arr =  np.array(x_tilde.cpu().data)[:,0]
#        pred = (pred_arr+1.0)/2.0
#        # put in real one for sanity check
#        # unscale
#        all_pred[:,rs] = pred
#        #all_pred[:,rs] = fake_pred
#        # zero out oldest timestep
#        last_timestep_img[:,0] *=0.0
#        # force correct new obs
#        # somewhere right here is where it is messed up
#        # if i sample from the data set - it works, if i use below two lines
#        # with what I think is the correct answer, it failes
#        my_step = Variable(torch.FloatTensor(pred), requires_grad=False).to(DEVICE)[:,None]
#        #this_x, this_y = datal[batch_idx+rs+1]
#        #real_step = this_x[:,3][:,None]
#        #real_rollout = all_real[:,rs][:,None]
#        this_step = Variable(torch.FloatTensor(my_step), requires_grad=False).to(DEVICE)
#        last_timestep_img = torch.cat((last_timestep_img[:,1:],this_step), dim=1)
#        #print('truevsmine')
#        #print(x.max(), pred.max(), real_step.max(), real_rollout.max())
#        #print(x.min(), pred.min(), real_step.min(), real_rollout.min())
#
#
#    for i, idx in enumerate(batch_idx):
#        for rs in range(args.rollout_length):
#            f, ax = plt.subplots(1,4, figsize=(12,3))
#            pred = all_pred[i,rs]
#            real = all_real[i,rs]
#            error = (pred-real)**2
#            # observation is last frame of x for this index
#            ax[0].imshow(x[i,-1])# vmin=0, vmax=config.freeway_max_pixel)
#            ax[0].set_title("obs")
#            ax[1].imshow(real,)# vmin=0, vmax=config.freeway_max_pixel)
#            ax[1].set_title("true t+%d steps" %(rs+1))
#            ax[2].imshow(pred,)# vmin=0, vmax=config.freeway_max_pixel)
#            ax[2].set_title("rollout t+%d steps" %(rs+1))
#            ax[3].imshow(error)
#            ax[3].set_title("error")
#            f.tight_layout()
#            save_img_path = os.path.join(basepath,
#                              'cond%02d_pred%02d_bidx%d_r%02d.png'%(largs.number_condition,
#                                                         largs.steps_ahead,
#                                                         idx, rs))
#            plt.savefig(save_img_path)
#            plt.close()
#        save_img_paths.append(save_img_path.replace('.png','')[:-4])
#    return save_img_paths
#
#def get_one_step_prediction(x,y,batch_idx):
#    print(x.shape)
#    x_d, z_e_x, z_q_x, latents = vmodel(x.to(DEVICE))
#    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#    # input x is between 0 and 1
#    pred_arr =  np.array(x_tilde.cpu().data)[:,0]
#    pred = (pred_arr+1.0)/2.0
#    return pred
#
#def generate_output_dataset():
#    oh,ow = 80,80
#    fs = 1
#    all_test_output = np.zeros((max(dataloader.test_loader.index_array)+fs,oh,ow))
#    all_train_output = np.zeros((max(dataloader.train_loader.index_array)+fs,oh,ow))
#
#    while not dataloader.done:
#        x,y,batch_idx = dataloader.ordered_batch()
#        if x.shape[0]:
#            pred = get_one_step_prediction(x,y,batch_idx)
#            all_train_output[batch_idx+fs] = pred
#
#    while not dataloader.test_done:
#        x,y,batch_idx = dataloader.validation_ordered_batch()
#        if x.shape[0]:
#            pred = get_one_step_prediction(x,y,batch_idx)
#            # prediction from one step ahead
#            all_test_output[batch_idx+fs] = pred
#
#    mname = os.path.split(args.model_loadname)[1]
#    output_train_data_file = train_data_file.replace('.pkl', mname)
#    output_test_data_file = test_data_file.replace('.pkl', mname)
#    np.savez(output_train_data_file+'.npz', all_train_output)
#    np.savez(output_test_data_file+'.npz', all_test_output)
#
#    mimsave(output_train_data_file+'.gif', all_train_output[:100])
#    mimsave(output_test_data_file+'.gif', all_test_output)


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--generate_savename', default='g')
    parser.add_argument('-bs', '--batch_size', default=5, type=int)
    parser.add_argument('-n', '--max_generations', default=70, type=int)
    parser.add_argument('-gg', '--generate_gif', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('-r', '--rollout_length', default=0, type=int)



    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    model_loadpath = os.path.abspath(args.model_loadname)
    if not os.path.exists(model_loadpath):
        print("Error: given model load path does not exist")
        print(model_loadpath)
        sys.exit()

    output_savepath = model_loadpath.replace('.pt', '_samples')
    if not os.path.exists(output_savepath):
        os.makedirs(output_savepath)
    model_dict = torch.load(model_loadpath, map_location=lambda storage, loc: storage)
    info = model_dict['info']
    largs = info['args'][-1]

    run_num = 0
    train_data_file = largs.train_data_file
    valid_data_file = largs.train_data_file.replace('training', 'valid')

    #train_data_loader = AtariDataset(
    #                               train_data_file,
    #                               number_condition=4,
    #                               steps_ahead=1,
    #                               batch_size=args.batch_size,
    #                               norm_by=255.,)
    valid_data_loader = AtariDataset(
                                   valid_data_file,
                                   number_condition=4,
                                   steps_ahead=1,
                                   batch_size=largs.batch_size,
                                   norm_by=255.0,)

    num_actions = valid_data_loader.n_actions
    args.size_training_set = valid_data_loader.num_examples
    hsize = valid_data_loader.data_h
    wsize = valid_data_loader.data_w

    vmodel = VQVAE(nr_logistic_mix=largs.nr_logistic_mix,
                   num_clusters=largs.num_k, encoder_output_size=largs.num_z,
                   in_channels_size=largs.number_condition, out_channels_size=1).to(DEVICE)

    vmodel.load_state_dict(model_dict['vmodel_state_dict'])
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode()
    sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')

