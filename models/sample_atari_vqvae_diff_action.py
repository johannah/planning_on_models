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

from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import utils
import cv2

# GRAD cam from https://github.com/jacobgil/pytorch-grad-cam
# started 845

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        #for name, module in self.model._modules.items():
        for name, module in self.model.encoder._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        #for name, module in self.model.action_conv._modules.items():
        #    x = module(x)
        #    if name in self.target_layers:
        #        x.register_hook(self.save_gradient)
        #        outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        #self.feature_extractor = FeatureExtractor(self.model.action_conv, ['4'])

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        #output = output.view(output.size(0), -1)
        if args.action_saliency:
            output = self.model.action_conv(output)[:,:,0,0]
        if args.reward_saliency:
            output = self.model.int_reward_conv(output)[:,:,0,0]
        return target_activations, output

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals = self.model(input)
        return pred_actions


    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.encoder.zero_grad()
        self.model.action_conv.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        dsh = input.shape[2]
        dsw = input.shape[3]
        cam = cv2.resize(cam, (dsh, dsw))
        #cam = cam - np.min(cam)
        #cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output
#
#def show_cam_on_image(img, mask):
#    # mask should be between 0 and 1
#    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_AUTUMN)
#    heatmap = np.float32(heatmap) / 255.0
#    #cam = heatmap + np.float32(img)
#    #cam = (cam - cam.min())/cam.max()
#    #cam = (cam - cam.min())/cam.max()
#    #cam = cam / np.max(cam)
#    #combine = np.hstack((img,cam))
#    #cv2.imwrite("cam.jpg", np.uint8(255 * combine))
#    return heatmap



def sample_batch(data, episode_number, episode_reward, name):
    nmix = int(info['num_output_mixtures']/2)
    grad_cam = GradCam(model=vqvae_model, target_layer_names=['10'], use_cuda=args.use_cuda)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    states, actions, rewards, values, pred_states, terminals, reset, relative_indexes = data
    actions = actions.to(DEVICE)
    rec = (2*reshape_input(pred_states[:,0][:,None])-1).to(DEVICE)
    # true data as numpy for plotting
    rec_true = (((rec+1)/2.0)).cpu().numpy()
    x = (2*reshape_input(states)-1).to(DEVICE)
    diff = (reshape_input(pred_states[:,1][:,None])).to(DEVICE)
    prev_true = (reshape_input(states[:,-2:-1])).cpu().numpy()

    if args.reward_int:
        true_signals = rewards.cpu().numpy()
    else:
        true_signals = values.cpu().numpy()

    action_preds = []
    action_preds_lsm = []
    action_preds_wrong = []
    action_steps = []
    action_trues = []
    signal_preds = []
    # (args.nr_logistic_mix/2)*3 is needed for each reconstruction
    raw_masks = []
    print("getting gradcam masks")
    for i in range(states.shape[0]):
        mask = grad_cam(x[i:i+1], target_index)
        raw_masks.append(mask)
        vqvae_model.zero_grad()
    raw_masks = np.array(raw_masks)
    mask_max = raw_masks.max()
    mask_min = raw_masks.min()
    raw_masks = (raw_masks-mask_min)/mask_max
    # flip grads for more visually appealing opencv JET colorplot
    raw_masks = 1-raw_masks
    cams = []
    for i in range(states.shape[0]):
        #heatmap = cv2.applyColorMap(np.uint8(255*raw_masks[i]), cv2.COLORMAP_AUTUMN)
        heatmap = cv2.applyColorMap(np.uint8(255*raw_masks[i]), cv2.COLORMAP_JET)
        cams.append(np.float32(heatmap)/255.0)
    cams = np.array(cams).astype(np.float)
    cams = 20 * np.log10((cams-cams.min()) + 1.)
    cams = (cams/cams.max())
    #cams = np.array([c-c.min() for c in cams])
    #cams = np.array([c/c.max() for c in cams])

    print("starting vqvae")
    rec_sams = np.zeros((args.num_samples, 80, 80), np.float32)
    for i in range(states.shape[0]):
        with torch.no_grad():
            cimg = cv2.cvtColor(rec_true[i,0],cv2.COLOR_GRAY2RGB).astype(np.float32)
            # both are between 0 and 1
            cam = cams[i]*.4 + cimg*.6
            x_d, z_e_x, z_q_x, latents, pred_actions, pred_signals = vqvae_model(x[i:i+1])
            rec_mest = x_d[:,:nmix].detach()
            diff_est = x_d[:,nmix:].detach()
            for n in range(args.num_samples):
                sam = sample_from_discretized_mix_logistic(rec_mest, largs.nr_logistic_mix, only_mean=False)
                rec_sams[n] = (((sam[0,0]+1)/2.0)).cpu().numpy()
            rec_est = np.mean(rec_sams, axis=0)

            # just take the mean from diff
            diff_est = sample_from_discretized_mix_logistic(diff_est, largs.nr_logistic_mix)[0,0]
            diff_true = diff[i,0]

            if args.reward_int:
                print('using int reward')
                pred_signal = torch.argmax(pred_signals).item()
            elif 'num_rewards' in info.keys():
                pred_signal = (pred_signals[0].cpu().numpy())
                print('using val reward',pred_signal)
            else:
                print('using no reward')
                pred_signal = -99

            signal_preds.append(pred_signal)
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

            print("A",action_preds_lsm[-1], pred_action, action)
            action_correct = pred_action == action
            print("R",true_signals[i], pred_signal)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), i))
            ax[0,0].imshow(prev_true[i,0])
            ax[0,0].set_title('prev TA:%s PA:%s'%(action,pred_action))
            ax[1,0].imshow(cam, vmin=0, vmax=1)

            # plot action saliency map
            if args.action_saliency:
                if action_correct:
                    ax[1,0].set_title('gcam-%s PA:%s COR  '%(saliency_name,pred_action))
                else:
                    ax[1,0].set_title('gcam-%s PA:%s WRG'%(saliency_name,pred_action))
            # plot reward saliency map
            if args.reward_saliency:
                reward_correct = true_signals[i]  == pred_signal
                if reward_correct:
                    ax[1,0].set_title('gcam-%s PR:%s COR  '%(saliency_name,pred_signal))
                else:
                    ax[1,0].set_title('gcam-%s PR:%s WRG'%(saliency_name,pred_signal))

            ax[0,1].imshow(rec_true[i,0], vmin=0, vmax=1)
            if args.reward_int:
                reward_correct = true_signals[i]  == pred_signal
                ax[0,1].set_title('rec true TR:%s PR:%s'%(true_signals[i], pred_signal))
                if reward_correct:
                    ax[1,1].set_title('rec est  PR:%s COR'%pred_signal)
                else:
                    ax[1,1].set_title('rec est  PR:%s WRG'%pred_signal)
            elif 'num_rewards' in info.keys():
                ax[0,1].set_title('rec true TR:%s PR:%s'%(np.round(true_signals[i],2), np.round(pred_signal,2)))
                ax[1,1].set_title('rec est PR:%s'%np.round(pred_signal,2))
            else:
                ax[0,1].set_title('rec true')
                ax[1,1].set_title('rec est')
            ax[1,1].imshow(rec_est, vmin=0, vmax=1)
            ax[0,2].imshow(diff_true, vmin=-1, vmax=1)
            ax[0,2].set_title('diff true')
            ax[1,2].imshow(diff_est, vmin=-1, vmax=1)
            ax[1,2].set_title('diff est')
            for a in range(2):
                for b in range(3):
                    ax[a,b].set_yticklabels([])
                    ax[a,b].set_xticklabels([])
                    ax[a,b].set_yticks([])
                    ax[a,b].set_xticks([])
            plt.suptitle(title)
            plt.savefig(iname)
            plt.close()
            if not i%10:
                print("saving", os.path.split(iname)[1])
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

    srname = os.path.join(output_savepath, '%s_E%05d_signal.txt'%(name, int(episode_number)))
    sf = open(srname, 'w')
    if args.reward_int:
        signal_preds = np.array(signal_preds).astype(np.int)
        signal_correct = []
        signal_incorrect = []
        signal_error = []

        for s in sorted(list(set(true_signals))):
            sigcor = np.sum(signal_preds[true_signals==s] ==  true_signals[true_signals==s])
            sigicor = np.sum(signal_preds[true_signals==s] != true_signals[true_signals==s])
            error = sigicor/float(np.sum(sigcor+sigicor))
            signal_correct.append(sigcor)
            signal_incorrect.append(sigicor)
            signal_error.append(error)
            v = 'reward signal {} correct {} incorrect {} error {}'.format(s,sigcor,sigicor,error)
            print(v)
            sf.write(v+'\n')
    else:
        mse = np.square(signal_preds-true_signals).mean()
        sf.write('mse: %s'%mse)
    sf.close()
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
    parser.add_argument('model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-as', '--action_saliency', action='store_true', default=True)
    parser.add_argument('-rs', '--reward_saliency', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-s', '--generate_savename', default='g')
    parser.add_argument('-bs', '--batch_size', default=5, type=int)
    parser.add_argument('-ns', '--num_samples', default=40, type=int)
    parser.add_argument('-mr', '--min_reward', default=-999, type=int)
    parser.add_argument('-l', '--limit', default=200, type=int)
    parser.add_argument('-n', '--max_generations', default=70, type=int)
    parser.add_argument('-gg', '--generate_gif', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()
    if args.action_saliency:
        saliency_name = 'act'
    if args.reward_saliency:
        saliency_name = 'rew'
        args.action_saliency = False

    if args.cuda:
        DEVICE = 'cuda'
        args.use_cuda = True
    else:
        DEVICE = 'cpu'
        args.use_cuda = False

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
    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode(diff=True, limit=args.limit, min_reward=args.min_reward)
    train_episode_batch, episode_index, episode_reward = train_data_loader.get_entire_episode(diff=True, limit=args.limit, min_reward=args.min_reward)


    sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')

    sample_batch(train_episode_batch, episode_index, episode_reward, 'train')

