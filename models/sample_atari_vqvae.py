import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import torch
from IPython import embed
from vqvae_module import VQVAE, VQVAE_PCNN_DECODER, VQVAE_ENCODER
import numpy as np
from copy import deepcopy
from ae_utils import sample_from_discretized_mix_logistic
from datasets import AtariDataset
from train_atari_action_vqvae import reshape_input
import config
torch.manual_seed(394)

def sample_batch(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        for i in range(states.shape[0]):
            fpr = forward_pass(vmodel, states[i:i+1], next_states[i:i+1], actions[i:i+1], nr_logistic_mix=largs.nr_logistic_mix, train=False, device=DEVICE, beta=largs.beta)
            x_d, z_e_x, z_q_x, latents, avg_loss_1, avg_loss_2, avg_loss_3 = fpr
            yhat = sample_from_discretized_mix_logistic(x_d, largs.nr_logistic_mix)
            yhat = (yhat+1)/2.0
            true = next_states[i:i+1,-1:]
            f,ax = plt.subplots(1,2)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
            title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i].item(), rewards[i].item())
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

def sample_autoregressive_batch(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        states = reshape_input(states).to(DEVICE)
        targets = (2*states[:,-1:]-1).to(DEVICE)
        if args.teacher_force:
            name+='_tf'
        bs = states.shape[0]
        #vqvae_model.scl
        print('generating %s images' %(bs))
        np_targets = deepcopy(targets.cpu().numpy())
        #output = np.zeros((targets.shape[2], targets.shape[3]))
        total_reward = 0
        for bi in range(bs):
            # sample one at a time due to memory constraints
            total_reward += rewards[bi].item()
            y = targets[bi:bi+1]
            if not args.teacher_force:
                y *=0.0
            title = 'step:%05d action:%d reward:%s %s/%s' %(bi, actions[bi].item(), int(rewards[bi]), total_reward, int(episode_reward))
            print("making", title)
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    for k in range(y.shape[3]):
                        x_d, z_e_x, z_q_x, latents = vqvae_model(states[bi:bi+1], y)
                        yhat = sample_from_discretized_mix_logistic(x_d, largs.nr_logistic_mix)
                        y[0,0,j,k] = 2*(yhat[0,0,j,k]/255.0)-1

            np_canvas = yhat[0,0].cpu().numpy()
            f,ax = plt.subplots(1,2)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), bi))
            ax[0].imshow(np_targets[bi,0])
            ax[0].set_title('true')
            ax[1].imshow(np_canvas)
            ax[1].set_title('est')
            plt.suptitle(title)
            plt.savefig(iname)
            print('saving', iname)
        search_path = iname[:-10:] + '*.png'
        gif_path = iname[:-10:] + '.gif'
        cmd = 'convert %s %s' %(search_path, gif_path)
        print('creating gif', gif_path)
        os.system(cmd)

def sample_autoregressive_batch_last_state(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        states = reshape_input(states).to(DEVICE)
        targets = (2*reshape_input(next_states[:,-1:])-1).to(DEVICE)
        actions = actions.to(DEVICE)
#
        if args.teacher_force:
            name+='_tf'
        bs = states.shape[0]
        #vqvae_model.scl
        print('generating %s images' %(bs))
        np_targets = deepcopy(targets.cpu().numpy())
        #output = np.zeros((targets.shape[2], targets.shape[3]))
        total_reward = 0
        for bi in range(bs):
            # sample one at a time due to memory constraints
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), bi))
            if not os.path.exists(iname):
                total_reward += rewards[bi].item()
                y = targets[bi:bi+1]*0.0
                y[0,0,0,0] = targets[bi,0,0,0]
                title = 'step:%05d action:%d reward:%s %s/%s' %(bi, actions[bi].item(), int(rewards[bi]), total_reward, int(episode_reward))
                print("making", title)
                for i in range(y.shape[1]):
                    for j in range(y.shape[2]):
                        for k in range(y.shape[3]):
                            x_d, z_e_x, z_q_x, latents = vqvae_model(states[bi:bi+1], y=y)
                            yhat = sample_from_discretized_mix_logistic(x_d, largs.nr_logistic_mix)
                            if not args.teacher_force:
                                y[0,0,j,k] = 2*(yhat[0,0,j,k]/255.0)-1
                            else:
                                y[0,0,j,k] = targets[bi,0,j,k]

                np_canvas = yhat[0,0].cpu().numpy()
                f,ax = plt.subplots(1,2)
                ax[0].imshow(np_targets[bi,0])
                ax[0].set_title('true')
                ax[1].imshow(np_canvas)
                ax[1].set_title('est')
                plt.suptitle(title)
                plt.savefig(iname)
                print('saving', iname)
        search_path = iname[:-10:] + '*.png'
        gif_path = iname[:-10:] + '.gif'
        cmd = 'convert %s %s' %(search_path, gif_path)
        print('creating gif', gif_path)
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
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

    vqenc = VQVAE_ENCODER(num_clusters=largs.num_k, encoder_output_size=largs.num_z,
                          in_channels_size=largs.number_condition)
    print("FIX ARGUMENTS")
    #pcnn_decoder = VQVAE_PCNN_DECODER(n_filters=largs.num_pcnn_filters,
    #                                  n_layers=largs.num_pcnn_layers,
    #                                  n_classes=num_actions,
    #                                  spatial_condition_size=1,
    #                                  float_condition_size=100*largs.num_z,
    #                                  hsize=hsize, wsize=wsize,
    #                                  num_output_channels=largs.nr_logistic_mix*3)
    pcnn_decoder = VQVAE_PCNN_DECODER(n_filters=largs.num_pcnn_filters,
                                      n_layers=largs.num_pcnn_layers,
                                      #n_classes=num_actions,
                                      #spatial_condition_size=1,
                                      #float_condition_size=100*largs.num_z,
                                      hsize=hsize, wsize=wsize,
                                      num_output_channels=info['decoder_output_channels'])

    #vqvae_model=VQVAE(vqenc,pcnn_decoder).to(DEVICE)
    vqvae_model=VQVAE(vqenc,pcnn_decoder, z_input_size=largs.z_input_size, pred_output_size=largs.pred_output_size).to(DEVICE)

    vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode()
    #sample_autoregressive_batch(valid_episode_batch, episode_index, episode_reward, 'valid')
    sample_autoregressive_batch_last_state(valid_episode_batch, episode_index, episode_reward, 'valid')

