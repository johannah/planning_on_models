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

"""
\cite{acn} The ACN encoder was a convolutional
network fashioned after a VGG-style classifier (Simonyan
& Zisserman, 2014), and the encoding distribution q(z|x)
was a unit variance Gaussian with mean specified by the
output of the encoder network.
size of z is 16 for mnist, 128 for others
"""

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

def handle_checkpointing(train_cnt, avg_train_loss):
    if ((train_cnt-info['last_save'])>=args.save_every):
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
    elif not len(info['train_cnts']):
        print("Logging model: %s no previous logs"%(train_cnt))
        handle_plot_ckpt(False, train_cnt, avg_train_loss)
    elif (train_cnt-info['last_plot'])>=args.plot_every:
        print("Plotting Model at cnt:%s cnt since last plotted:%s"%(train_cnt, train_cnt-info['last_plot']))
        handle_plot_ckpt(True, train_cnt, avg_train_loss)
    else:
        if (train_cnt-info['train_cnts'][-1])>=args.log_every:
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
    return train_cnt

def test_act(train_cnt, do_plot):
    act_model.eval()
    test_loss = 0
    print('starting test', train_cnt)
    st = time.time()
    seen = 0
    with torch.no_grad():
        valid_buffer.reset_unique()
        for i in range(10):
            if valid_buffer.unique_available:
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
                if i == 0:
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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-se', '--save_every', default=100000*10, type=int)
    parser.add_argument('-pe', '--plot_every', default=200000, type=int)
    parser.add_argument('-le', '--log_every', default=200000, type=int)
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=20, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)

    parser.add_argument('--train_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005679481_train.npz')
    parser.add_argument('--valid_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval.npz')
    args = parser.parse_args()

    nr_logistic_mix = 10
    num_output_channels = 1
    nmix = (2*nr_logistic_mix+nr_logistic_mix)*num_output_channels

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(config.model_savedir, 'sigcacn_breakout_action')

    train_data_path = args.train_buffer
    valid_data_path = args.valid_buffer
    train_buffer = make_subset_buffer(train_data_path, max_examples=60000)
    valid_buffer = make_subset_buffer(valid_data_path, max_examples=int(60000*.1))
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
        act_model.load_state_dict(_dict['action_model_state_dict'])
        info = _dict['info']
        train_cnt = info['train_cnts'][-1]
    if args.sample:
        test_act(train_cnt, True)
    else:
        parameters = list(act_model.parameters())
        opt = optim.Adam(parameters, lr=args.learning_rate)
        if args.model_loadpath !='':
            opt.load_state_dict(_dict['optimizer'])
        # test plotting first
        test_act(train_cnt, True)
        while train_cnt < args.num_examples_to_train:
            train_cnt = train_act(train_cnt)

