import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
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
    s = (2*reshape_input(states)-1)
    ns = (2*reshape_input(next_states)-1).cpu().numpy()
    # make channels for actions which is the size of the latents
    elen = actions.shape[0]
    channel_actions = torch.zeros((elen, forward_info['num_actions'], forward_info['hsize'], forward_info['hsize']))
    for a in range(forward_info['num_actions']):
        channel_actions[actions==a,a] = 1.0
    all_real_latents = []
    # first pred index is zeros - since we cant predict it
    all_pred_latents = []
    all_pred_rewards = []
    used_prev_latents = []
    assert args.lead_in > 0
    for i in range(args.rollout_length):
        x_d, z_e_x, z_q_x, real_latents, pred_actions, pred_signals = vqvae_model(s[i:i+1])
        # for the ith index
        all_real_latents.append(real_latents[0].cpu().numpy())
        if i < args.lead_in:
            latents = real_latents
        else:
            # use last predicted latents
            latents = pred_next_latents
        # predict next latent step given action
        used_prev_latents.append(latents[0].cpu().numpy())
        state_input = torch.cat((channel_actions[i][None,:], latents[:,None].float()), dim=1)
        out_pred_next_latents, pred_rewards = conv_forward_model(state_input)
        # take argmax over channels axis
        pred_next_latents = torch.argmax(out_pred_next_latents, dim=1)
        # prediction for the i + 1 index
        all_pred_latents.append(pred_next_latents[0].cpu().numpy())
        all_pred_rewards.append(pred_rewards)
    plot_latents(all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params)

def plot_latents(all_real_latents, all_pred_latents, all_pred_rewards, used_prev_latents, params):
    episode_number, episode_reward, name = params
    for i in range(args.rollout_length-1):
        # todo - im plotting one short

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

        iname = os.path.join(output_savepath, '%sforward_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
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





#    # true data as numpy for plotting
#    rec = (2*reshape_input(pred_states[:,0][:,None])-1).to(DEVICE)
#    rec_true = (((rec+1)/2.0)).cpu().numpy()
#    diff = (reshape_input(pred_states[:,1][:,None])).to(DEVICE)
#    prev_true = (reshape_input(states[:,-2:-1])).cpu().numpy()
#
#    if args.reward_int:
#        true_signals = rewards.cpu().numpy()
#    else:
#        true_signals = values.cpu().numpy()
#
#    action_preds = []
#    action_preds_lsm = []
#    action_preds_wrong = []
#    action_steps = []
#    action_trues = []
#    signal_preds = []
#    # (args.nr_logistic_mix/2)*3 is needed for each reconstruction
#    raw_masks = []
#    print("getting gradcam masks")
#    raw_masks = np.array(raw_masks)
#    mask_max = raw_masks.max()
#    mask_min = raw_masks.min()
#    raw_masks = (raw_masks-mask_min)/mask_max
#    # flip grads for more visually appealing opencv JET colorplot
#    raw_masks = 1-raw_masks
#    cams = []
#    for i in range(states.shape[0]):
#        #heatmap = cv2.applyColorMap(np.uint8(255*raw_masks[i]), cv2.COLORMAP_AUTUMN)
#        heatmap = cv2.applyColorMap(np.uint8(255*raw_masks[i]), cv2.COLORMAP_JET)
#        cams.append(np.float32(heatmap)/255.0)
#    cams = np.array(cams).astype(np.float)
#    cams = 20 * np.log10((cams-cams.min()) + 1.)
#    cams = (cams/cams.max())
#    #cams = np.array([c-c.min() for c in cams])
#    #cams = np.array([c/c.max() for c in cams])
#
#    print("starting vqvae")
#    rec_sams = np.zeros((args.num_samples, 80, 80), np.float32)
#    rec_est = np.zeros((80,80))
#    for i in range(states.shape[0]):
#        with torch.no_grad():
#            cimg = cv2.cvtColor(rec_true[i,0],cv2.COLOR_GRAY2RGB).astype(np.float32)
#            # both are between 0 and 1
#            cam = cams[i]*.4 + cimg*.6
#            x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals = vqvae_model(x[i:i+1])
#            rec_mest = x_d[:,:nmix].detach()
#            diff_est = x_d[:,nmix:].detach()
#            if args.num_samples:
#                for n in range(args.num_samples):
#                    sam = sample_from_discretized_mix_logistic(rec_mest, largs.nr_logistic_mix, only_mean=False)
#                    rec_sams[n] = (((sam[0,0]+1)/2.0)).cpu().numpy()
#                rec_est = np.mean(rec_sams, axis=0)
#            rec_mean = sample_from_discretized_mix_logistic(rec_mest, largs.nr_logistic_mix, only_mean=True)
#            rec_mean = (((rec_mean[0,0]+1)/2.0)).cpu().numpy()
#
#            # just take the mean from diff
#            diff_est = sample_from_discretized_mix_logistic(diff_est, largs.nr_logistic_mix)[0,0]
#            diff_true = diff[i,0]
#
#            if args.reward_int:
#                print('using int reward')
#                pred_signal = torch.argmax(pred_signals).item()
#            elif 'num_rewards' in info.keys():
#                pred_signal = (pred_signals[0].cpu().numpy())
#                print('using val reward',pred_signal)
#            else:
#                print('using no reward')
#                pred_signal = -99
#
#            signal_preds.append(pred_signal)
#            f,ax = plt.subplots(2,4)
#            title = 'step %s/%s action %s reward %s' %(i, states.shape[0], actions[i].item(), rewards[i].item())
#            pred_action = torch.argmax(pred_actions).item()
#            action = int(actions[i].item())
#            action_preds.append(pred_action)
#            action_preds_lsm.append(pred_actions.cpu().numpy())
#            if pred_action != action:
#                action_preds_wrong.append(pred_action)
#                action_trues.append(action)
#                action_steps.append(i)
#
#            print("A",action_preds_lsm[-1], pred_action, action)
#            action_correct = pred_action == action
#            print("R",true_signals[i], pred_signal)
#            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
#            ax[0,0].imshow(prev_true[i,0])
#            ax[0,0].set_title('prev TA:%s PA:%s'%(action,pred_action))
#            ax[1,0].imshow(cam, vmin=0, vmax=1)
#
#            # plot action saliency map
#            if args.action_saliency:
#                if action_correct:
#                    ax[1,0].set_title('gc%s PA:%s COR  '%(saliency_name,pred_action))
#                else:
#                    ax[1,0].set_title('gc%s PA:%s WRG'%(saliency_name,pred_action))
#            # plot reward saliency map
#            if args.reward_saliency:
#                reward_correct = true_signals[i]  == pred_signal
#                if reward_correct:
#                    ax[1,0].set_title('gcam-%s PR:%s COR  '%(saliency_name,pred_signal))
#                else:
#                    ax[1,0].set_title('gcam-%s PR:%s WRG'%(saliency_name,pred_signal))
#
#            if args.reward_int:
#                reward_correct = true_signals[i]  == pred_signal
#                ax[0,1].set_title('rec true')#%(true_signals[i], pred_signal))
#                ax[0,2].set_title('TR:%s PR:%s'%(true_signals[i], pred_signal))
#                if reward_correct:
#                    ax[1,1].set_title('avgest COR')
#                    ax[1,2].set_title('samest COR')
#                else:
#                    ax[1,1].set_title('avgest WRG')
#                    ax[1,2].set_title('samest WRG')
#            elif 'num_rewards' in info.keys():
#                ax[0,1].set_title('rec true')#%(np.round(true_signals[i],2), np.round(pred_signal,2)))
#                ax[0,2].set_title('TR:%s PR:%s'%(np.round(true_signals[i],2), np.round(pred_signal,2)))
#                ax[1,1].set_title('avgrec est PR:%s'%np.round(pred_signal,2))
#                ax[1,2].set_title('samrec est PR:%s'%np.round(pred_signal,2))
#            else:
#                ax[0,1].set_title('rec true')
#                ax[0,2].set_title('rec true')
#                ax[1,1].set_title('avgrec est')
#                ax[1,2].set_title('samrec est')
#
#            ax[0,1].imshow(rec_true[i,0], vmin=0, vmax=1)
#            ax[0,2].imshow(rec_true[i,0], vmin=0, vmax=1)
#            ax[1,1].imshow(rec_mean, vmin=0, vmax=1)
#            ax[1,2].imshow(rec_est, vmin=0, vmax=1)
#
#            ax[0,3].set_title('diff true')
#            ax[1,3].set_title('diff est')
#            ax[0,3].imshow(diff_true, vmin=-1, vmax=1)
#            ax[1,3].imshow(diff_est, vmin=-1, vmax=1)
#            for a in range(2):
#                for b in range(4):
#                    ax[a,b].set_yticklabels([])
#                    ax[a,b].set_xticklabels([])
#                    ax[a,b].set_yticks([])
#                    ax[a,b].set_xticks([])
#            plt.tight_layout()
#            plt.suptitle(title)
#            plt.savefig(iname)
#            plt.close()
#            if not i%10:
#                print("saving", os.path.split(iname)[1])
#    # plot actions
#    aname = os.path.join(output_savepath, '%s_E%05d_action.png'%(name, int(episode_number)))
#    plt.figure()
#    plt.scatter(action_steps, action_preds_wrong, alpha=.5, label='predict')
#    plt.scatter(action_steps, action_trues, alpha=.1, label='actual')
#    plt.legend()
#    plt.savefig(aname)
#
#    actions = actions.cpu().data.numpy()
#    action_preds = np.array(action_preds)
#    actions_correct = []
#    actions_incorrect = []
#    actions_error = []
#
#    arname = os.path.join(output_savepath, '%s_E%05d_action.txt'%(name, int(episode_number)))
#    af = open(arname, 'w')
#    for a in sorted(list(set(actions))):
#        actcor = np.sum(action_preds[actions==a] == actions[actions==a])
#        acticor = np.sum(action_preds[actions==a] != actions[actions==a])
#        error = acticor/float(np.sum(actcor+acticor))
#        actions_correct.append(actcor)
#        actions_incorrect.append(acticor)
#        actions_error.append(error)
#        v = 'action {} correct {} incorrect {} error {}'.format(a,actcor,acticor,error)
#        print(v)
#        af.write(v+'\n')
#    af.close()
#
#    srname = os.path.join(output_savepath, '%s_E%05d_signal.txt'%(name, int(episode_number)))
#    sf = open(srname, 'w')
#    if args.reward_int:
#        signal_preds = np.array(signal_preds).astype(np.int)
#        signal_correct = []
#        signal_incorrect = []
#        signal_error = []
#
#        for s in sorted(list(set(true_signals))):
#            sigcor = np.sum(signal_preds[true_signals==s] ==  true_signals[true_signals==s])
#            sigicor = np.sum(signal_preds[true_signals==s] != true_signals[true_signals==s])
#            error = sigicor/float(np.sum(sigcor+sigicor))
#            signal_correct.append(sigcor)
#            signal_incorrect.append(sigicor)
#            signal_error.append(error)
#            v = 'reward signal {} correct {} incorrect {} error {}'.format(s,sigcor,sigicor,error)
#            print(v)
#            sf.write(v+'\n')
#    else:
#        mse = np.square(signal_preds-true_signals).mean()
#        sf.write('mse: %s'%mse)
#    sf.close()
#    gif_path = iname[:-10:] + '.gif'
#    search_path = iname[:-10:] + '*.png'
#    cmd = 'convert %s %s' %(search_path, gif_path)
#    print('creating gif', gif_path)
#    os.system(cmd)

if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('forward_model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-as', '--action_saliency', action='store_true', default=True)
    parser.add_argument('-rs', '--reward_saliency', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-s', '--generate_savename', default='g')
    parser.add_argument('-bs', '--batch_size', default=5, type=int)
    parser.add_argument('-li', '--lead_in', default=5, type=int)
    parser.add_argument('-rl', '--rollout_length', default=10, type=int)
    parser.add_argument('-ns', '--num_samples', default=40, type=int)
    parser.add_argument('-mr', '--min_reward', default=-999, type=int)
    parser.add_argument('-l', '--limit', default=200, type=int)
    parser.add_argument('-n', '--max_generations', default=70, type=int)
    parser.add_argument('-gg', '--generate_gif', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)

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

    vq_model_loadpath = forward_largs.train_data_file.replace('train_forward.npz', '.pt')
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
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode(diff=True, limit=args.limit, min_reward=args.min_reward)
    #train_episode_batch, episode_index, episode_reward = train_data_loader.get_entire_episode(diff=True, limit=args.limit, min_reward=args.min_reward)

    conv_forward_model = ForwardResNet(BasicBlock, data_width=forward_info['hsize'],
                                       num_channels=forward_info['num_channels'],
                                       num_output_channels=num_k,
                                       num_rewards=forward_info['num_rewards'])
    conv_forward_model.load_state_dict(forward_model_dict['conv_forward_model'])
    conv_forward_model = conv_forward_model.to(DEVICE)

    sample_episode(valid_episode_batch, episode_index, episode_reward, 'valid')
