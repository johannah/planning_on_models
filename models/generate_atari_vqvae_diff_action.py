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
        for dname, data_loader in {'valid':valid_data_loader, 'train':train_data_loader}.items():
            rmax = data_loader.relative_indexes.max()
            new = True
            st = 1
            en = 0
            while en < rmax-1:
                en = min(st+args.batch_size, rmax-1)
                fdata = data_loader.get_data(np.arange(st, en, dtype=np.int))
                fterminals = list(fdata[5])
                # end at end of episode
                if 1 in fterminals:
                    en = st+list(fterminals).index(1)+1
                    data = data_loader.get_data(np.arange(st, en, dtype=np.int))
                else:
                    data = fdata
                print('generating from %s to %s of %s' %(st,en,rmax))
                states, actions, rewards, values, next_states, terminals, reset, relative_indexes = data
                assert np.sum(terminals[:-1]) == 0
                prev_relative_indexes = relative_indexes-1
                prev_data = data_loader.get_data(prev_relative_indexes)
                pstates, pactions, prewards, pvalues, pnext_states, pterminals, preset, prelative_indexes = prev_data
                ps = (2*reshape_input(torch.FloatTensor(pstates))-1).to(DEVICE)
                s = (2*reshape_input(torch.FloatTensor(states))-1).to(DEVICE)
                ns = (2*reshape_input(torch.FloatTensor(next_states))-1).to(DEVICE)
                for xx in range(s.shape[0]):
                    try:
                        assert ps[xx,-1].sum() == s[xx,-2].sum() ==  ns[xx,-3].sum()
                    except:
                        print("assert broke", xx)
                        embed()
                px_d, zp_e_x, pz_q_x, platents, _, _ = vqvae_model(ps)
                x_d, z_e_x, z_q_x, latents, _, _ = vqvae_model(s)
                nx_d, nz_e_x, nz_q_x, nlatents, _, _ = vqvae_model(ns)
                if new:
                    all_prev_latents = platents.cpu()
                    all_latents = latents.cpu()
                    all_next_latents = nlatents.cpu()

                    all_prev_actions = pactions
                    all_prev_rewards = prewards
                    all_prev_values = pvalues

                    all_rewards = rewards
                    all_values = values
                    all_actions = actions
                    all_rel_inds = relative_indexes
                    new = False
                else:
                    all_prev_latents = np.concatenate((all_prev_latents, platents.cpu().numpy()), axis=0)
                    all_latents = np.concatenate((all_latents, latents.cpu().numpy()), axis=0)
                    all_next_latents = np.concatenate((all_next_latents, nlatents.cpu().numpy()), axis=0)

                    all_prev_rewards = np.concatenate((all_prev_rewards, prewards))
                    all_prev_values =  np.concatenate((all_prev_values, pvalues))
                    all_prev_actions = np.concatenate((all_prev_actions, pactions))

                    all_rewards = np.concatenate((all_rewards, rewards))
                    all_values =  np.concatenate((all_values, values))
                    all_actions = np.concatenate((all_actions, actions))
                    all_rel_inds = np.concatenate((all_rel_inds, relative_indexes))
                if 1 in fterminals:
                    # skip ahead one so that prev state is correct
                    st = en+1
                else:
                    st = en

            forward_filename = args.model_loadname.replace('.pt', '_%s_forward.npz'%dname)
            np.savez(forward_filename,
                     relative_indexes=all_rel_inds,
                     prev_latents=all_prev_latents,
                     latents=all_latents,
                     next_latents=all_next_latents,
                     rewards=all_rewards,
                     values=all_values,
                     actions=all_actions,
                     prev_rewards=all_prev_rewards,
                     prev_values= all_prev_values,
                     prev_actions=all_prev_actions,
                     num_k=largs.num_k)


if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('-l', '--model_loadname', help='full path to vq model',
                        #default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward00/vqdiffactintreward_0118012272ex.pt')
                        default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0071507436ex.pt')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=True)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
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
                                   batch_size=args.batch_size,
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
