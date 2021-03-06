import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import torch
from IPython import embed
from vqvae import VQVAE
import numpy as np
from copy import deepcopy
from ae_utils import sample_from_discretized_mix_logistic, discretized_mix_logistic_loss
from datasets import AtariDataset
from train_atari_action_vqvae import reshape_input
import config
torch.manual_seed(394)

def sample_batch(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        x = (2*reshape_input(states[:,-1:])-1).to(DEVICE)
        for i in range(states.shape[0]):
            x_d, z_e_x, z_q_x, latents = vqvae_model(x[i:i+1])
            loss_1 = discretized_mix_logistic_loss(x_d, x[i:i+1], nr_mix=largs.nr_logistic_mix, DEVICE=DEVICE)
            yhat = sample_from_discretized_mix_logistic(x_d, largs.nr_logistic_mix)
            yhat = (((yhat+1)/2.0)*255.0).cpu().numpy().astype(np.int)
            true = (states[i:i+1,-1:]*255.0).cpu().numpy().astype(np.int)
            f,ax = plt.subplots(1,2)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
            print("writing", os.path.split(iname)[1])
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

    train_data_loader = AtariDataset(
                                   train_data_file,
                                   number_condition=4,
                                   steps_ahead=1,
                                   batch_size=args.batch_size,
                                   norm_by=255.,)
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

    vqvae_model = VQVAE(num_clusters=largs.num_k,
                        encoder_output_size=largs.num_z,
                        in_channels_size=largs.number_condition).to(DEVICE)

    vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    #valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode()
    #sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')
    train_episode_batch, episode_index, episode_reward = train_data_loader.get_entire_episode()
    sample_batch(train_episode_batch, episode_index, episode_reward, 'train')

