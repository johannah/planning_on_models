"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

Strongly referenced ACN implementation and blog post from:
http://jalexvig.github.io/blog/associative-compression-networks/

Base VAE referenced from pytorch examples:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

# TODO conv
# TODO load function
# daydream function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2

from torch import nn, optim
import torch
from torch.nn import functional as F
import config
from lstm_utils import plot_losses
torch.manual_seed(394)
torch.set_num_threads(4)
from ae_utils import save_checkpoint
#from acn_gmp import ConvVAE, PriorNetwork, acn_gmp_loss_function
sys.path.append('../agents')
from replay import ReplayMemory
from IPython import embed
random_state = np.random.RandomState(3)
#

def make_subset_buffer(buffer_path, max_examples=100000, frame_height=40, frame_width=40):
    # keep max_examples < 100000 to enable knn search
    # states [top of image:bottom of image,:]
    # in breakout - can safely reduce size to be 80x80 of the given image
    # try to get an even number of each type of reward

    small_path = buffer_path.replace('.npz', '_%06d.npz' %max_examples)
    if os.path.exists(small_path):
        print('loading small buffer path')
        print(small_path)
        load_buffer = ReplayMemory(load_file=small_path)
    else:
        load_buffer = ReplayMemory(load_file=buffer_path)
        print('loading prescribed buffer path')
        print(buffer_path)

    # TODO if frame size is wrong - we arent handling
    if load_buffer.count > max_examples:
        print('creating small buffer')
        # actions for breakout:
        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        sbuffer = ReplayMemory(max_examples, frame_height=frame_height, frame_width=frame_width,
                               agent_history_length=load_buffer.agent_history_length)

        # remove ends because they are scary
        ends = np.where(load_buffer.terminal_flags==1)[0][1:-1]
        random_state.shuffle(ends)
        for tidx in ends:
            if sbuffer.count >= max_examples:
                continue
            else:
                # start after the last terminal
                i = tidx+1
                # while there isnt a new terminal flag
                while not load_buffer.terminal_flags[i+1]:
                    frame=cv2.resize(load_buffer.frames[i][:,:,None], (frame_height, frame_width))
                    sbuffer.add_experience(action=load_buffer.actions[i],
                                           frame=frame,
                                           reward=load_buffer.rewards[i],
                                           terminal=load_buffer.terminal_flags[i])
                    i+=1

        sbuffer.save_buffer(small_path)
        load_buffer = sbuffer
    assert load_buffer.count > 10
    return load_buffer

def prepare_next_state(st, DEVICE, NORM_BY):
    # states come in at uint8
    # Can use BCE if we convert to be between 0 and one - but this means we miss
    # color detail
    ####################################3
    #output = (torch.FloatTensor(st)/255.0).to(DEVICE)
    #output[output>0] = 1.0
    #output[output==0] = -1
    #return o
    #
    ####################################3
    # should be converted to float between -1 and 1
    output = (2*torch.FloatTensor(st)/NORM_BY-1).to(DEVICE)
    assert output.max() < 1.01
    assert output.min() > -1.01
    ####################################3
    # should be converted to float between 0 and 1
    #output = (torch.FloatTensor(st)/NORM_BY).to(DEVICE)
    #assert output.max() < 1.01
    #assert output.min() > -.01
    return output

def prepare_state(st, DEVICE, NORM_BY):
    # states come in at uint8 - should be converted to float between -1 and 1
    # st.shape is bs,4,40,40
    output = (2*torch.FloatTensor(st)/NORM_BY-1).to(DEVICE)
    assert output.max() < 1.01
    assert output.min() > -1.01
    # convert to 0 and 1
    return output

def make_state(batch, DEVICE, NORM_BY):
    # states are [ts0, ts1, ts2, ts3]
    # actions are   [a0,  a1,  a2,  a3]
    # next_states     [ts1, ts2, ts3, ts4]
    # rewards are    [r0,  r1,  r2,  a3]
    states, actions, rewards, next_states, terminal_flags, masks = batch
    states = prepare_state(states, DEVICE, NORM_BY)
    next_states = prepare_next_state(next_states, DEVICE, NORM_BY)
    # next state is the corresponding action
    actions = torch.LongTensor(actions).to(DEVICE)
    rewards = torch.LongTensor(rewards).to(DEVICE)
    bs, _, h, w = states.shape
    return states, actions, rewards, next_states

def save_model(info, model_dict):
    train_cnt = info['model_train_cnts'][-1]
    info['model_last_save'] = train_cnt
    info['model_save_times'].append(time.time())
    #avg_valid_losses = valid_vqvae(train_cnt, model, info, valid_batch)
    #handle_plot_ckpt(train_cnt, info, avg_train_losses, avg_valid_losses)
    # TODO - replace w valid
    #handle_plot_ckpt(train_cnt, info, avg_train_losses, avg_valid_losses)
    filename = os.path.join(info['MODEL_BASE_FILEDIR'], "_%010dex.pt"%train_cnt)
    print("Saving model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['model_last_save']))
    print(filename)
    state = {
             'model_info':info,
             }
    for (model_name, model) in model_dict.items():
        state[model_name+'_state_dict'] = model.state_dict()
    save_checkpoint(state, filename=filename)
    return info


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def run_module(self, model, x, outputs, register=[]):
        for name, module in model._modules.items():
            x = module(x)
            #print(name, x.shape)
            #if name in self.target_layers:
            if name in register:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return x, outputs

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # the way i wrote the action model means that we run through conv first, then linear
        x, outputs = self.run_module(self.model.conv_network, x, outputs, ['4'])
        x = torch.flatten(x, 1)
        x, outputs = self.run_module(self.model.linear_network, x, outputs)
        x = F.log_softmax(x, dim=1)
        return outputs, x

class ModelOutputs():
    """
    Class formaking a forward pass and getting
    1. network output
    2. activation from intermediate targeted layers
    3. gradients from intermediate targeted layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        # maybe need to do stuff here
        return target_activations, output

class GradCam:
    def __init__(self, model, target_layer_names, grad_on_forward_index):
        """
        grad_on_forward_index - which index of the model's forward() function to use for grad cam.
                                if None, return result forward(x)  (suitable if only one return from forward is expected)
        """
        self.model = model
        self.model.eval()
        self.grad_on_forward_index = grad_on_forward_index
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, x):
        if self.grad_on_forward_index == None:
            return self.model(x)
        else:
            return self.model(x)[self.grad_on_forward_index]

    def __call__(self, x, index=None):
        """
        index is prediction class to use for cam. If None, use predicted class
        """
        # features are output from specified layers
        features, output = self.extractor(x)
        if index == None:
            index = torch.argmax(output, 1)
        output = output.to('cpu')
        one_hot = torch.zeros((1, output.shape[-1]))
        # one_hot should be float
        one_hot[0,index] = 1
        one_hot = torch.sum(one_hot*output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # grads_val should be bs,c,h,w - > 1,16,20,20
        grads_val = self.extractor.get_gradients()[-1].cpu().numpy()
        # get last layer used as feature - should be 2d
        # target should be c,h,w - > 16,20,20
        bs = x.shape[0]
        _,_,dsh,dsw = x.shape
        cams = np.zeros((bs,dsh, dsw))
        for idx in range(bs):
            target = features[-1].detach().cpu().numpy()[idx,:]
            # weights should be c in size -> (16,)
            weights = np.mean(grads_val, axis=(2,3))[idx,:]
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam+=w*target[i,:,:]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (dsh, dsw))
            #print(cam.min(), cam.max())
            #cam -=np.min(cam)
            #cam /= np.max(cam)
            cams[idx] = cam
        return cams

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    if len(img.shape) == 2 or img.shape[0] == 1:
        img = cv2.applyColorMap(np.uint8(255*img), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    combine = np.hstack((img,cam))
    cv2.imwrite("cam.jpg", np.uint8(255 * combine))

# ConvVAE was also imported - not sure which one was used
class ConvAct(nn.Module):
    def __init__(self, input_size=1, num_output_options=3):
        super(ConvAct, self).__init__()
        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=8,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),
           )

        self.linear_network = nn.Sequential(
                                nn.Linear(6400, 1024),
                                nn.ReLU(True),
                                nn.Linear(1024, 512),
                                nn.ReLU(True),
                                nn.Dropout2d(0.5),
                                nn.Linear(512, num_output_options),)

    def forward(self, x):
        x = self.conv_network(x)
        # output is 512,16,20,20, which flattens to bs,6400
        x = torch.flatten(x, 1)
        output = F.log_softmax(self.linear_network(x), dim=1)
        return output

def handle_plot_ckpt(do_plot, train_cnt, avg_train_loss):
    info['train_losses'].append(avg_train_loss)
    info['train_cnts'].append(train_cnt)
    test_loss = test_act(train_cnt,do_plot)
    info['test_losses'].append(test_loss)
    info['test_cnts'].append(train_cnt)
    print('examples %010d train loss %03.03f test loss %03.03f' %(train_cnt,
                              info['train_losses'][-1], info['test_losses'][-1]))
    # plot
    if do_plot:
        info['last_plot'] = train_cnt
        rolling = 3
        if len(info['train_losses'])<rolling*3:
            rolling = 1
        print('adding last loss plot', train_cnt)
        plot_name = vae_base_filepath + "_%010dloss.png"%train_cnt
        print('plotting loss: %s with %s points'%(plot_name, len(info['train_cnts'])))
        plot_losses(info['train_cnts'],
                    info['train_losses'],
                    info['test_cnts'],
                    info['test_losses'], name=plot_name, rolling_length=rolling)

def handle_checkpointing(train_cnt, avg_train_loss, force_save=False):
    if ((train_cnt-info['last_save'])>=args.save_every or force_save):
        print("Saving Model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['last_save']))
        info['last_save'] = train_cnt
        info['save_times'].append(time.time())
        handle_plot_ckpt(True, train_cnt, avg_train_loss)
        filename = vae_base_filepath + "_%010dex.pkl"%train_cnt
        state = {
                 'act_model_state_dict':act_model.state_dict(),
                 'optimizer':opt.state_dict(),
                 'info':info,
                 }
        save_checkpoint(state, filename=filename)
    elif not len(info['train_cnts'] or force_save):
        print("Logging model: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(False, train_cnt, avg_train_loss)
    elif (train_cnt-info['last_plot'])>=args.plot_every or force_save:
        print("Plotting Model at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, avg_train_loss)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every or force_save:
            print("Logging Model at cnt:%s cnt since last logged:%s"%(train_cnt, train_cnt-info['train_cnts'][-1]))
            handle_plot_ckpt(False, train_cnt, avg_train_loss)

def train_act(train_cnt):
    act_model.train()
    train_loss = 0
    init_cnt = train_cnt
    st = time.time()
    train_buffer.reset_unique()
    while train_buffer.unique_available:
        batch = train_buffer.get_unique_minibatch(args.batch_size)
        batch_idx = batch[-1]
        states, actions, rewards, next_states = make_state(batch[:-1], DEVICE, 255.)
        # given states [1,2,3,4], predict action taken bt 3 and 4
        data = next_states
        opt.zero_grad()
        pred_actions = act_model(data)
        # add the predicted codes to the input
        action_loss = F.nll_loss(pred_actions, actions, reduction='sum') # TODO - could also weight actions here
        loss = action_loss
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_loss = train_loss/float((train_cnt+data.shape[0])-init_cnt)
        if train_cnt > 50000:
            handle_checkpointing(train_cnt, avg_train_loss)
        train_cnt+=len(data)
    print("finished epoch after %s seconds at cnt %s"%(time.time()-st, train_cnt))
    return train_cnt, avg_train_loss

def test_act(train_cnt, do_plot):
    act_model.eval()
    test_loss = 0
    print('starting test', train_cnt)
    st = time.time()
    seen = 0
    plotted = False
    do_plot = True
    with torch.no_grad():
        valid_buffer.reset_unique()
        while valid_buffer.unique_available:
            batch = valid_buffer.get_unique_minibatch(args.batch_size)
            batch_idx = batch[-1]
            states, actions, rewards, next_states = make_state(batch[:-1], DEVICE, 255.)
            data = next_states
            # yhat_batch is bt 0-1

            pred_actions = act_model(data)
            action_loss = F.nll_loss(pred_actions, actions, reduction='sum') # TODO - could also weight actions here
            loss = action_loss
            test_loss+= loss.item()
            seen += data.shape[0]
            if not plotted:
                plotted = True
                if do_plot:
                     print('writing img')
                     n = min(data.size(0), 8)
                     bs = data.shape[0]
                     # sampled yhat_batch is bt 0-1
                     # yimg is bt 0.78 and 0.57 -
                     bs,_,h,w = data.shape
                     # data should be between 0 and 1 to be plotted with
                     # save_image
                     f,ax = plt.subplots(4,5, sharex=True, sharey=True, squeeze=True)
                     npdata = data.cpu().numpy()
                     npactions = actions.cpu().numpy()
                     nppactions = np.argmax(pred_actions.cpu().numpy(), 1)
                     cnt = 0
                     for cnt, idx in enumerate(np.random.choice(np.arange(data.shape[0]), 5)):
                         for xx in range(4):
                             ax[xx,cnt].imshow(npdata[idx,xx])
                         ax[xx,cnt].set_title('T%s P%s'%(npactions[idx], nppactions[idx]))
                         #ax[0,cnt].set_title('%s'%(nppactions))
                     img_name = vae_base_filepath + "_%010d_valid_action.png"%train_cnt
                     plt.savefig(img_name)
                     plt.close()
                     print('finished writing img', img_name)

                     acc = accuracy_score(npactions, nppactions)
                     conf = confusion_matrix(npactions, nppactions)
                     print('--------accuracy-------')
                     print(acc)
                     print(conf)

    test_loss /= seen
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('finished test', time.time()-st)
    return test_loss

def load_avg_grad_cam(act_model_loadpath, data_buffer, data_buffer_loadpath, DEVICE='cpu'):
    '''
    '''
    from skvideo.io import vwrite
    outpath = data_buffer_loadpath.replace('.npz', '') + '_' + os.path.split(act_model_loadpath)[1].split('.')[0]
    npy_outpath = outpath+'.npz'
    print("looking for ", npy_outpath)
    if os.path.exists(npy_outpath):
        print("loading", npy_outpath)
        return np.load(npy_outpath)['action_grad']
    else:
        num_actions = data_buffer.num_actions()
        act_model = ConvAct(input_size=4, num_output_options=num_actions).to(DEVICE)
        _dict = torch.load(act_model_loadpath, map_location=lambda storage, loc:storage)
        act_model.load_state_dict(_dict['act_model_state_dict'])

        bs = 1
        grad_cam = GradCam(model=act_model, target_layer_names=['4'], grad_on_forward_index=None)
        #for phase, data_buffer in {'train':train_buffer, 'valid':valid_buffer}.items():
        data_buffer.reset_unique()
        all_masks = np.zeros((data_buffer.unique_indexes.shape[0],
                             data_buffer.frame_height, data_buffer.frame_width),
                             np.float32)

        while data_buffer.unique_available:
            batch = data_buffer.get_unique_minibatch(bs)
            batch_idx = batch[-1]
            states, actions, rewards, next_states = make_state(batch[:-1], DEVICE, 255.)
            masks = grad_cam(next_states)
            all_masks[batch_idx] = masks

        #outpath = args.model_loadpath.replace(end, phase+'%s_gradcam.mp4'%phase)
        #npz_avgrm_outpath = args.model_loadpath.replace(end, '%s_avg_rm_gradcam'%phase)
        #avgrm_outpath = args.model_loadpath.replace(end, '%s_avg_rm_gradcam.mp4'%phase)

        pmean = np.mean(all_masks, axis=0)
        print('mean', pmean.shape)
        all_masks = all_masks-pmean
        all_masks = (all_masks-all_masks.min())/(all_masks.max()-all_masks.min())
        # otherwise use train mean for valid mask
        np.savez_compressed(outpath, action_grad=all_masks)

        uall_masks=all_masks*255.0
        uall_masks+=20
        uall_masks[uall_masks>255]=255
        uall_masks=uall_masks.astype(np.uint8)

        #vwrite(outpath, all_masks)
        vwrite(outpath+'.mp4', uall_masks)
        print('writing film %s'%(outpath+'.mp4'))
        return all_masks

def run_grad_cam():
    bs = 12
    grad_cam = GradCam(model=act_model, target_layer_names=['4'], grad_on_forward_index=None)
    valid_buffer.reset_unique()
    batch = valid_buffer.get_unique_minibatch(bs)
    batch_idx = batch[-1]
    states, actions, rewards, next_states = make_state(batch[:-1], DEVICE, 255.)
    st = time.time()
    masks = grad_cam(next_states)
    et = time.time()
    print(et-st)
    pred_actions = act_model(next_states)
    # norm bt 0 and 1
    prev_imgs = (next_states[:,-2].cpu().numpy()+1)/2.0
    next_imgs = (next_states[:,-1].cpu().numpy()+1)/2.0
    np_pred_actions = torch.argmax(pred_actions, 1).detach().cpu().numpy()
    np_actions = actions.detach().cpu().numpy()
    f,ax = plt.subplots(3, bs, sharex=True, sharey=True, figsize=(3*bs,3))
    for cnt in range(bs):
        ax[0, cnt].set_title('BI%s T%s P%s'%(cnt, np_actions[cnt], np_pred_actions[cnt]))
        ax[0, cnt].imshow(prev_imgs[cnt], cmap='gray')
        ax[1, cnt].imshow(next_imgs[cnt], cmap='gray')
        #ax[2, cnt].imshow(next_imgs[cnt], cmap='gray')
        ax[2, cnt].imshow(masks[cnt], alpha=.5, cmap='viridis')
        #ax[3, cnt].imshow(masks[cnt], alpha=1, cmap='jet')
        for xx in range(3):
            ax[xx,cnt].axis('off')

    end = '.'+args.model_loadpath.split('.')[-1]
    outpath = args.model_loadpath.replace(end, '_gradcam.png')
    plt.savefig(outpath)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-gc', '--grad_cam', action='store_true', default=True)
    parser.add_argument('-agc', '--avg_grad_cam', action='store_true', default=True)
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-md', '--model_savedir', default='../../model_savedir/')
    parser.add_argument('-se', '--save_every', default=100000*50, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000*50, type=int)
    parser.add_argument('-le', '--log_every', default=100000*50, type=int)
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=20, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=100000*1000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    parser.add_argument('-ne', '--max_training_examples', default=200000)

    parser.add_argument('--train_buffer', default='MFBreakout_train_anneal_14342_04/breakout_S014342_N0005679481_train.npz')
    parser.add_argument('--valid_buffer', default='MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval.npz')
    args = parser.parse_args()
    nexamples = args.max_training_examples

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(args.model_savedir, 'sigcacn_breakout_action')
    train_data_path = os.path.join(args.model_savedir, args.train_buffer)
    valid_data_path = os.path.join(args.model_savedir, args.valid_buffer)
    train_buffer = make_subset_buffer(train_data_path, max_examples=nexamples)
    valid_buffer = make_subset_buffer(valid_data_path, max_examples=int(nexamples*.15))
    num_actions = train_buffer.num_actions()
    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
             }

    size_training_set = train_buffer.count
    train_cnt = 0
    # given four frames, predict the action that was taken to get from [1,2,3,4]
    # frame 3 to 4
    act_model = ConvAct(input_size=4, num_output_options=num_actions).to(DEVICE)
    if args.model_loadpath !='':
        _dict = torch.load(args.model_loadpath, map_location=lambda storage, loc:storage)
        act_model.load_state_dict(_dict['act_model_state_dict'])
        info = _dict['info']
        train_cnt = info['train_cnts'][-1]
    if args.sample:
        test_act(train_cnt, True)
    if args.grad_cam:
        run_grad_cam()
    if args.avg_grad_cam:
        avg_grad_cam(args.model_loadpath, valid_buffer, small_valid_path)
        avg_grad_cam()
    else:
        parameters = list(act_model.parameters())
        opt = optim.Adam(parameters, lr=args.learning_rate)
        if args.model_loadpath !='':
            opt.load_state_dict(_dict['optimizer'])
        # test plotting first
        test_act(train_cnt, True)
        while train_cnt < args.num_examples_to_train:
            train_cnt, avg_train_loss = train_act(train_cnt)
        handle_checkpointing(train_cnt, avg_train_loss)
