import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from copy import deepcopy
import torch
from IPython import embed
from vqvae import VQVAE
from forward_conv import BasicBlock, ForwardResNet
import numpy as np
from ae_utils import sample_from_discretized_mix_logistic, discretized_mix_logistic_loss
from datasets import AtariDataset
from train_atari_action_vqvae import reshape_input
import config
torch.manual_seed(394)

import torch.nn.functional as F
from torchvision import utils
import cv2

def grow_latent(latent, out_size=(80,80)):
    latent = latent.astype(np.float)
    lm = latent.max()
    out_latent = (cv2.resize(latent/lm, out_size)*lm).astype(np.int)
    return out_latent

def sample_episode(data, episode_number, episode_reward, name):
    params = (episode_number, episode_reward, name)
     # rollout for number of steps and decode with vqvae decoder
    states, actions, rewards, values, next_states, terminals, reset, relative_indexes = data
    snp = reshape_input(deepcopy(states))
    s = (2*reshape_input(torch.FloatTensor(states))-1)
    nsnp = reshape_input(next_states)
    embed()
    # make channels for actions which is the size of the latents
    actions = torch.LongTensor(actions).to(DEVICE)
    elen = actions.shape[0]
    channel_actions = torch.zeros((elen, forward_info['num_actions'], forward_info['hsize'], forward_info['hsize']))
    for a in range(forward_info['num_actions']):
        channel_actions[actions==a,a] = 1.0
    all_real_latents = []
    # first pred index is zeros - since we cant predict it
    all_pred_latents = []
    all_pred_rewards = []
    used_prev_latents = []
    assert args.lead_in >= 2
    for i in range(args.rollout_length):
        x_d, z_e_x, z_q_x, real_latents, pred_actions, pred_signals = vqvae_model(s[i:i+1])
        # for the ith index
        all_real_latents.append(real_latents[0].cpu().numpy())
        if i < args.lead_in:
            used_prev_latents.append(real_latents)
        else:
            # use last predicted latents
            used_prev_latents.append(pred_next_latents)
        if len(used_prev_latents) >=2:
            state_input = torch.cat((channel_actions[i][None,:], used_prev_latents[-2][:,None].float(), used_prev_latents[-1][:,None].float()), dim=1)
            out_pred_next_latents, pred_prev_actions, pred_rewards = conv_forward_model(state_input)
            # take argmax over channels axis
            pred_next_latents = torch.argmax(out_pred_next_latents, dim=1)
            # prediction for the i + 1 index
            all_pred_latents.append(pred_next_latents[0].cpu().numpy())
            all_pred_rewards.append(pred_rewards)
    used_prev_latents = [ul[0].cpu().numpy() for ul in used_prev_latents]
    #plot_latents(all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params)
    plot_reconstructions(snp, nsnp, all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params)

def plot_latents(all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params):
    episode_number, episode_reward, name = params
    for i in range(args.rollout_length-1):
        f,ax = plt.subplots(3,3)
        # true latent at time i
        ax[0,0].imshow(all_real_latents[i], interpolation="None")
        ax[0,0].set_title('s-%02d true'% i)

        # was this teacher forced or rolled out?
        if i <  args.lead_in:
            ax[1,0].set_title('s-%02d given'%i)
        else:
            ax[1,0].set_title('s-%02d self '%i)
        ax[1,0].imshow(used_prev_latents[i], interpolation="None")
        ax[2,0].set_title('s-%02d error'%i)
        s_error = np.square(used_prev_latents[i] - all_real_latents[i])
        ax[2,0].imshow(s_error, interpolation="None")

        ax[0,1].set_title('s1-%02d true latent'%(i+1))
        ax[0,1].imshow(all_real_latents[i+1], interpolation="None")
        ax[1,1].set_title('s1-%02d pred'%(i+1))
        ax[1,1].imshow(all_pred_latents[i], interpolation="None")

        s1_error = np.square(all_real_latents[i+1] - all_pred_latents[i])
        ax[2,1].set_title('s1-%02d error'%(i+1))
        ax[2,1].imshow(s1_error, interpolation="None")

        ts_diff = np.square(all_real_latents[i]-all_real_latents[i+1])
        ax[0,2].imshow(ts_diff, interpolation="None")
        ax[0,2].set_title('s-s1-true diff')

        ts_pred_diff = np.square(all_real_latents[i+1]-all_pred_latents[i])
        ax[1,2].imshow(ts_pred_diff, interpolation="None")
        ax[1,2].set_title('s-s1-pred diff')

        error_ts_diff = np.square(ts_pred_diff, ts_diff)
        ax[2,2].imshow(error_ts_diff, interpolation="None")
        ax[2,2].set_title('s-s1 diff error')

        iname = os.path.join(output_savepath, '%s_latent_forward_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
        for a in range(len(ax[0])):
            for b in range(len(ax[1])):
                ax[a,b].set_yticklabels([])
                ax[a,b].set_xticklabels([])
                ax[a,b].set_yticks([])
                ax[a,b].set_xticks([])
        plt.tight_layout()
        #title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i].item(), rewards[i].item())
        #plt.suptitle(title)
        plt.savefig(iname)
        plt.close()
    gif_path = iname[:-10:] + '.gif'
    search_path = iname[:-10:] + '*.png'
    cmd = 'convert %s %s' %(search_path, gif_path)
    print('creating gif', gif_path)
    os.system(cmd)


def plot_reconstructions(true_states, true_next_states, all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params):
    # TODO - feed in true actions
    episode_number, episode_reward, name = params
    for i in range(args.rollout_length-1):
        # todo - this needs to go in main function so we have the option to
        # rollout actions

        all_latents = np.concatenate((all_real_latents[i][None,None],  # true s
                                      used_prev_latents[i][None,None],  # pred s
                                      all_real_latents[i+1][None,None], # true s+1
                                      all_pred_latents[i][None,None], # pred s+1
                                      ), axis=0).astype(np.int)
        all_latents = torch.LongTensor(all_latents)
        N,H,W,C = all_latents.shape[0],10,10,vq_largs.num_z
        z_q_x, x_d = vqvae_model.decode_clusters(all_latents,N,H,W,C)


        # TODO
        nmix = 30
        rec_mest = x_d[:,:nmix].detach()
        if args.num_samples:
            rec_sams = np.zeros((4, args.num_samples, 1, 80, 80))
            for n in range(args.num_samples):
                sam = sample_from_discretized_mix_logistic(rec_mest, vq_largs.nr_logistic_mix, only_mean=False)
                rec_sams[:,n] = (((sam+1)/2.0)).cpu().numpy()
            rec_est = np.mean(rec_sams, axis=1)
        rec_mean = sample_from_discretized_mix_logistic(rec_mest, vq_largs.nr_logistic_mix, only_mean=True)
        rec_mean = (((rec_mean+1)/2.0)).cpu().numpy()


        ##########################################
        f,ax = plt.subplots(3,3)
        ax[0,0].imshow(true_states[i,-1], interpolation="None")
        ax[0,0].set_title('%02d true s '%i)
        ax[0,1].imshow(true_next_states[i,-1], interpolation="None")
        ax[0,1].set_title('%02d true s1'%(i+1))
        ax[0,2].imshow(true_next_states[i,-1], interpolation="None")
        ax[0,2].set_title('%02d true s1'%(i+1))

        #ax[1,0].imshow(rec_est[0,0], interpolation="None")
        #ax[1,0].set_title('s rec true sam')

        ## was this teacher forced or rolled out?
        if i <  args.lead_in:
            ax[1,0].set_title('s rec given')
        else:
            ax[1,0].set_title('s rec self sam')
        ax[1,0].imshow(rec_est[1,0], interpolation="None")

        ax[1,2].set_title('s1 rec tf sam')
        ax[1,2].imshow(rec_est[2,0], interpolation="None")

        ax[1,1].set_title('s1 rec est sam')
        ax[1,1].imshow(rec_est[3,0], interpolation="None")

        ax[2,0].set_title('error s')
        serror = np.square(true_states[i,-1]-rec_est[1,0])
        ax[2,0].imshow(serror, interpolation="None")

        ax[2,1].set_title('error s1')
        s1error = np.square(true_next_states[i,-1]-rec_est[3,0])
        ax[2,1].imshow(s1error, interpolation="None")

        ax[2,2].set_title('error tf s1')
        s1error = np.square(true_next_states[i,-1]-rec_est[2,0])
        ax[2,2].imshow(s1error, interpolation="None")

        iname = os.path.join(output_savepath, '%s_rec_forward_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
        for a in range(len(ax[0])):
            for b in range(len(ax[1])):
                ax[a,b].set_yticklabels([])
                ax[a,b].set_xticklabels([])
                ax[a,b].set_yticks([])
                ax[a,b].set_xticks([])
        plt.tight_layout()
        #title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i].item(), rewards[i].item())
        #plt.suptitle(title)
        plt.savefig(iname)
        plt.close()
    gif_path = iname[:-10:] + '.gif'
    search_path = iname[:-10:] + '*.png'
    cmd = 'convert %s %s' %(search_path, gif_path)
    print('creating gif', gif_path)
    os.system(cmd)

if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('-l', '--forward_model_loadname', help='full path to model', default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward00/convnrpa00/convnrpa_0052009984ex.pt')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-as', '--action_saliency', action='store_true', default=True)
    parser.add_argument('-rs', '--reward_saliency', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=False)
    parser.add_argument('-s', '--generate_savename', default='g')
    parser.add_argument('-bs', '--batch_size', default=5, type=int)
    parser.add_argument('-li', '--lead_in', default=5, type=int)
    parser.add_argument('-rl', '--rollout_length', default=20, type=int)
    parser.add_argument('-ns', '--num_samples', default=40, type=int)
    parser.add_argument('-mr', '--min_reward', default=-999, type=int)
    parser.add_argument('-lim', '--limit', default=1000, type=int)

    args = parser.parse_args()
    if args.action_saliency:
        saliency_name = 'A'
    if args.reward_saliency:
        saliency_name = 'R'
        args.action_saliency = False

    if args.cuda:
        DEVICE = 'cuda'
        args.use_cuda = True
    else:
        DEVICE = 'cpu'
        args.use_cuda = False

    forward_model_loadpath = os.path.abspath(args.forward_model_loadname)
    if not os.path.exists(forward_model_loadpath):
        print("Error: given forwrad model load path does not exist")
        print(forward_model_loadpath)
        sys.exit()

    output_savepath = forward_model_loadpath.replace('.pt', '_samples')
    if not os.path.exists(output_savepath):
        os.makedirs(output_savepath)
    forward_model_dict = torch.load(forward_model_loadpath, map_location=lambda storage, loc: storage)
    forward_info = forward_model_dict['info']
    forward_largs = forward_info['args'][-1]

    vq_model_loadpath = forward_largs.train_data_file.replace('_train_forward.npz', '.pt')
    vq_model_dict = torch.load(vq_model_loadpath, map_location=lambda storage, loc: storage)
    vq_info = vq_model_dict['info']
    vq_largs = vq_info['args'][-1]

    run_num = 0
    train_data_file = vq_largs.train_data_file
    valid_data_file = vq_largs.train_data_file.replace('training', 'valid')

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
                                   batch_size=args.batch_size,
                                   norm_by=255.0,)

    args.size_training_set = valid_data_loader.num_examples
    hsize = valid_data_loader.data_h
    wsize = valid_data_loader.data_w

    num_k = vq_largs.num_k
    if args.reward_int:
        int_reward = vq_info['num_rewards']
        vqvae_model = VQVAE(num_clusters=num_k,
                            encoder_output_size=vq_largs.num_z,
                            num_output_mixtures=vq_info['num_output_mixtures'],
                            in_channels_size=vq_largs.number_condition,
                            n_actions=vq_info['num_actions'],
                            int_reward=vq_info['num_rewards']).to(DEVICE)
    elif 'num_rewards' in vq_info.keys():
        print("CREATING model with est future reward")
        vqvae_model = VQVAE(num_clusters=num_k,
                            encoder_output_size=vq_largs.num_z,
                            num_output_mixtures=vq_info['num_output_mixtures'],
                            in_channels_size=vq_largs.number_condition,
                            n_actions=vq_info['num_actions'],
                            int_reward=False,
                            reward_value=True).to(DEVICE)
    else:
        vqvae_model = VQVAE(num_clusters=num_k,
                            encoder_output_size=vq_largs.num_z,
                            num_output_mixtures=vq_info['num_output_mixtures'],
                            in_channels_size=vq_largs.number_condition,
                            n_actions=vq_info['num_actions'],
                            ).to(DEVICE)

    vqvae_model.load_state_dict(vq_model_dict['vqvae_state_dict'])
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode(diff=False, limit=args.limit, min_reward=args.min_reward)
    #train_episode_batch, episode_index, episode_reward = train_data_loader.get_entire_episode(diff=False, limit=args.limit, min_reward=args.min_reward)

    conv_forward_model = ForwardResNet(BasicBlock, data_width=forward_info['hsize'],
                                       num_channels=forward_info['num_channels'],
                                       num_actions=forward_info['num_actions'],
                                       num_output_channels=num_k,
                                       num_rewards=forward_info['num_rewards'])
    conv_forward_model.load_state_dict(forward_model_dict['conv_forward_model'])
    conv_forward_model = conv_forward_model.to(DEVICE)

    sample_episode(valid_episode_batch, episode_index, episode_reward, 'valid')
