import matplotlib
matplotlib.use('Agg')
import sys
import os
import torch
from IPython import embed
from vqvae import VQVAE
import numpy as np
from datasets import AtariDataset
from train_atari_action_vqvae import reshape_input
import config
torch.manual_seed(394)

def generate_forward_datasets():
    with torch.no_grad():
        for dname in ['valid', 'train']:
            reset = False
            ecnt = 0
            while not reset:
                data = eval('%s_data_loader.get_unique_minibatch()'%dname)
                states, actions, rewards, values, next_states, terminals, reset, relative_indexes = data
                s = (2*reshape_input(states)-1).to(DEVICE)
                ns = (2*reshape_input(next_states)-1).to(DEVICE)
                x_d, z_e_x, z_q_x, latents, _, _ = vqvae_model(s)
                nx_d, nz_e_x, nz_q_x, next_latents, _, _ = vqvae_model(ns)
                if not ecnt:
                    all_latents = latents.cpu()
                    all_next_latents = next_latents.cpu()
                    all_rewards = rewards
                    all_values = values
                    all_actions = actions
                else:
                    all_latents = torch.cat((all_latents, latents.cpu()), dim=0)
                    all_next_latents = torch.cat((all_next_latents, next_latents.cpu()), dim=0)
                    all_rewards = torch.cat((all_rewards, rewards))
                    all_values = torch.cat((all_values, values))
                    all_actions = torch.cat((all_actions, actions))
                ecnt +=1

            forward_filename = args.model_loadname.replace('.pt', '_%s_forward.npz'%dname)
            np.savez(forward_filename,
                                latents=all_latents.numpy(),
                                next_latents=all_next_latents.numpy(),
                                rewards=all_rewards.numpy(),
                                values=all_values.numpy(),
                                actions=all_actions.numpy(),
                                num_k=largs.num_k)


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=False)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
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

    args.size_training_set = valid_data_loader.num_examples
    hsize = valid_data_loader.data_h
    wsize = valid_data_loader.data_w

    if args.reward_int:
        int_reward = info['num_rewards']
        vqvae_model = VQVAE(num_clusters=largs.num_k,
                            encoder_output_size=largs.num_z,
                            num_output_mixtures=info['num_output_mixtures'],
                            in_channels_size=largs.number_condition,
                            n_actions=info['num_actions'],
                            int_reward=info['num_rewards']).to(DEVICE)
    elif 'num_rewards' in info.keys():
        print("CREATING model with est future reward")
        vqvae_model = VQVAE(num_clusters=largs.num_k,
                            encoder_output_size=largs.num_z,
                            num_output_mixtures=info['num_output_mixtures'],
                            in_channels_size=largs.number_condition,
                            n_actions=info['num_actions'],
                            int_reward=False,
                            reward_value=True).to(DEVICE)
    else:
        vqvae_model = VQVAE(num_clusters=largs.num_k,
                            encoder_output_size=largs.num_z,
                            num_output_mixtures=info['num_output_mixtures'],
                            in_channels_size=largs.number_condition,
                            n_actions=info['num_actions'],
                            ).to(DEVICE)

    vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
    generate_forward_datasets()
