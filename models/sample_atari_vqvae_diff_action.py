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
    nmix = int(info['num_output_mixtures']/2)
    with torch.no_grad():
        states, actions, rewards, pred_states, terminals, reset, relative_indexes = data
        actions = actions.to(DEVICE)
        rec = (2*reshape_input(pred_states[:,0][:,None])-1).to(DEVICE)
        x = (2*reshape_input(states)-1).to(DEVICE)
        diff = (reshape_input(pred_states[:,1][:,None])).to(DEVICE)
        prev_true = (reshape_input(states[:,-2:-1])*255).cpu().numpy().astype(np.int)
        action_preds = []
        action_preds_lsm = []
        action_preds_wrong = []
        action_steps = []
        action_trues = []
        # (args.nr_logistic_mix/2)*3 is needed for each reconstruction
        for i in range(states.shape[0]):
            x_d, z_e_x, z_q_x, latents, pred_actions = vqvae_model(x[i:i+1])
            rec_est = x_d[:,:nmix]
            diff_est = x_d[:,nmix:]
            rec_est = sample_from_discretized_mix_logistic(rec_est, largs.nr_logistic_mix)
            rec_true= (((rec[i,0]+1)/2.0)*255.0).cpu().numpy().astype(np.int)
            rec_est = (((rec_est[0,0]+1)/2.0)*255.0).cpu().numpy().astype(np.int)
            diff_est = sample_from_discretized_mix_logistic(diff_est, largs.nr_logistic_mix)[0,0]
            diff_true = diff[i,0]
            f,ax = plt.subplots(2,3)
            title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i].item(), rewards[i].item())
            pred_action = torch.argmax(pred_actions).item()
            action = int(actions[i].item())
            action_preds.append(pred_action)
            action_preds_lsm.append(pred_actions.cpu().numpy())
            if pred_action != action:
                action_preds_wrong.append(pred_action)
                action_trues.append(action)
                action_steps.append(i)

            print(action_preds_lsm[-1], pred_action, action)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
            if not os.path.exists(iname):
                ax[0,0].imshow(prev_true[i,0])
                ax[0,0].set_title('prev A:%s'%action)
                ax[1,0].set_title('est  A:%s'%pred_action)
                ax[0,1].imshow(rec_true, vmin=0, vmax=255)
                ax[0,1].set_title('rec true')
                ax[1,1].imshow(rec_est, vmin=0, vmax=255)
                ax[1,1].set_title('rec est')
                ax[0,2].imshow(diff_true, vmin=-1, vmax=1)
                ax[0,2].set_title('diff true')
                ax[1,2].imshow(diff_est, vmin=-1, vmax=1)
                ax[1,2].set_title('diff est')
                plt.suptitle(title)
                plt.savefig(iname)
                if not i%10:
                    print("saving", os.path.split(iname)[1])

        gif_path = iname[:-10:] + '.gif'
        if not os.path.exists(gif_path):
            search_path = iname[:-10:] + '*.png'
            cmd = 'convert %s %s' %(search_path, gif_path)
            print('creating gif', gif_path)
            os.system(cmd)
        # plot actions
        aname = os.path.join(output_savepath, '%s_E%05d_action.png'%(name, int(episode_number)))
        plt.figure()
        plt.scatter(action_steps, action_preds_wrong, alpha=.5, label='predict')
        plt.scatter(action_steps, action_trues, alpha=.1, label='actual')
        plt.legend()
        plt.savefig(aname)

        actions = actions.cpu().data.numpy()
        action_preds = np.array(action_preds)
        actions_correct = []
        actions_incorrect = []
        actions_error = []

        arname = os.path.join(output_savepath, '%s_E%05d_action.txt'%(name, int(episode_number)))
        af = open(arname, 'w')
        for a in sorted(list(set(actions))):
            actcor = np.sum(action_preds[actions==a] == actions[actions==a])
            acticor = np.sum(action_preds[actions==a] != actions[actions==a])
            error = acticor/float(np.sum(actcor+acticor))
            actions_correct.append(actcor)
            actions_incorrect.append(acticor)
            actions_error.append(error)
            v = 'action {} correct {} incorrect {} error {}'.format(a,actcor,acticor,error)
            print(v)
            af.write(v+'\n')
        af.close()
        embed()

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

    args.size_training_set = valid_data_loader.num_examples
    hsize = valid_data_loader.data_h
    wsize = valid_data_loader.data_w

    vqvae_model = VQVAE(num_clusters=largs.num_k,
                        encoder_output_size=largs.num_z,
                        num_output_mixtures=info['num_output_mixtures'],
                        in_channels_size=largs.number_condition,
                        n_actions=info['num_actions']).to(DEVICE)

    vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    #valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode(diff=True)
    #sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')
    train_episode_batch, episode_index, episode_reward = train_data_loader.get_entire_episode(diff=True)
    sample_batch(train_episode_batch, episode_index, episode_reward, 'train')

