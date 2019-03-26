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

import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

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

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    combine = np.hstack((img,cam))
    cv2.imwrite("cam.jpg", np.uint8(255 * combine))

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
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
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

if __name__ == '__main__':
    import argparse
    default_base_savedir = config.model_savedir
    if not os.path.exists(default_base_savedir):
        os.makedirs(default_base_savedir)
    parser = argparse.ArgumentParser(description='generate vq-vae')
    parser.add_argument('model_loadname', help='full path to model')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-ri', '--reward_int', action='store_true', default=False)
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
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode(diff=True)
    #sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    grad_cam = GradCam(model = vqvae_model,
                    target_layer_names =['10'], use_cuda=args.use_cuda)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    states, actions, rewards, values, pred_states, terminals, reset, relative_indexes = valid_episode_batch

    input_data = (2*reshape_input(states[:1])-1).to(DEVICE)

    mask = grad_cam(input_data, target_index)
    img = input_data[0,-1].cpu().numpy()
    img = (img+1)/2.0
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    show_cam_on_image(cimg, mask)
    #utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')

    #gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    #gb = gb_model(input, index=target_index)
    #utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    #cam_mask = np.zeros(gb.shape)
    #for i in range(0, gb.shape[0]):
    #    cam_mask[i, :, :] = mask

    #cam_gb = np.multiply(cam_mask, gb)
    #utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')


