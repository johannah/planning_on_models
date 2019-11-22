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
import time
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2
from copy import deepcopy

from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.nn.utils.clip_grad import clip_grad_value_
from torchvision.utils import save_image
torch.set_num_threads(4)
torch.manual_seed(394)

from imageio import imsave
from ae_utils import save_checkpoint, handle_plot_ckpt, reshape_input
from pixel_cnn import GatedPixelCNN
from acn_mdn import ConvVAE, PriorNetwork, acn_mdn_loss_function
sys.path.append('../agents')
from replay import ReplayMemory

from IPython import embed
random_state = np.random.RandomState(3)
############

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
    # states come in at uint8 - should be converted to float between 0 and 1
    output = (reshape_input(torch.FloatTensor(st)/NORM_BY)).to(DEVICE)
    assert output.max() < 1.01
    assert output.min() > -.01
    return output

def prepare_state(st, DEVICE, NORM_BY):
    # states come in at uint8 - should be converted to float between -1 and 1
    output = (2*reshape_input(torch.FloatTensor(st)/NORM_BY)-1).to(DEVICE)
    assert output.max() < 1.01
    assert output.min() > -1.01
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

def add_losses(info, train_cnt, phase, kl_loss, rec_loss):
    info['model_%s_cnts'%phase].append(train_cnt)
    if '%s_kl_loss'%phase not in info['model_%s_losses'%phase].keys():
        info['model_%s_losses'%phase]['%s_kl_loss'%phase] = []
    info['model_%s_losses'%phase]['%s_kl_loss'%phase].append(kl_loss)
    if '%s_rec_loss'%phase not in info['model_%s_losses'%phase].keys():
        info['model_%s_losses'%phase]['%s_rec_loss'%phase] = []
    info['model_%s_losses'%phase]['%s_rec_loss'%phase].append(rec_loss)
    return info


def train_acn(info, model_dict, data_buffers, phase='train'):
    encoder_model = model_dict['encoder_model']
    prior_model = model_dict['prior_model']
    pcnn_decoder = model_dict['pcnn_decoder']
    opt = model_dict['opt']

    # add one to the rewards so that they are all positive
    # use next_states because that is the t-1 action

    if len(info['model_train_cnts']):
        train_cnt = info['model_train_cnts'][-1]
    else: train_cnt = 0

    num_batches = data_buffers['train'].count//info['MODEL_BATCH_SIZE']
    while train_cnt < 10000000:
        if phase == 'valid':
            encoder_model.eval()
            prior_model.eval()
            pcnn_decoder.eval()
        else:
            encoder_model.train()
            prior_model.train()
            pcnn_decoder.train()

        batch_num = 0
        data_buffers[phase].reset_unique()
        print('-------------new epoch %s------------------'%phase)
        print('num batches', num_batches)
        while data_buffers[phase].unique_available:
            opt.zero_grad()
            batch = data_buffers[phase].get_unique_minibatch(info['MODEL_BATCH_SIZE'])
            relative_indices = batch[-1]
            states, actions, rewards, next_states = make_state(batch[:-1], info['DEVICE'], info['NORM_BY'])
            next_state = next_states[:,-1:]
            bs = states.shape[0]
            #states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
            z, u_q = encoder_model(states)

            # add the predicted codes to the input
            #yhat_batch = torch.sigmoid(pcnn_decoder(x=next_state,
            #                                        class_condition=actions,
            #                                        float_condition=z))
            yhat_batch = encoder_model.decode(z)
            prior_model.codes[relative_indices] = u_q.detach().cpu().numpy()

            mix, u_ps, s_ps = prior_model(u_q)

            # track losses
            kl_loss, rec_loss = acn_mdn_loss_function(yhat_batch, next_state, u_q, mix, u_ps, s_ps)
            loss = kl_loss + rec_loss
            # aatch size because it hasn't been added to train cnt yet


            if not phase == 'valid':
                loss.backward()
                #parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
                parameters = list(encoder_model.parameters()) + list(prior_model.parameters())
                clip_grad_value_(parameters, 10)
                opt.step()
                train_cnt += bs

            if not batch_num%info['MODEL_LOG_EVERY_BATCHES']:
                print(phase, train_cnt, batch_num, kl_loss.item(), rec_loss.item())
                info = add_losses(info, train_cnt, phase, kl_loss.item(), rec_loss.item())
            batch_num+=1

        if (((train_cnt-info['model_last_save'])>=info['MODEL_SAVE_EVERY'])):
            info = add_losses(info, train_cnt, phase, kl_loss.item(), rec_loss.item())
            if phase == 'train':
                # run as valid phase and get back to here
                phase = 'valid'
            else:
                model_dict = {'encoder_model':encoder_model,
                              'prior_model':prior_model,
                              'pcnn_decoder':pcnn_decoder,
                              'opt':opt}
                info = save_model(info, model_dict)
                phase = 'train'



    model_dict = {'encoder_model':encoder_model,
                 'prior_model':prior_model,
                 'pcnn_decoder':pcnn_decoder,
                 'opt':opt}

    info = save_model(info, model_dict)

def sample_acn(info, model_dict, data_buffers, num_samples=4, teacher_force=False):
    print("sampling model")

    encoder_model = model_dict['encoder_model']
    prior_model = model_dict['prior_model']
    pcnn_decoder = model_dict['pcnn_decoder']

    for phase in ['train', 'validl']:
        if len(info['model_train_cnts']):
            train_cnt = info['model_train_cnts'][-1]
        else:
            train_cnt = 0

        data_buffers[phase].reset_unique()
        encoder_model.eval()
        prior_model.eval()
        #pcnn_decoder.eval()
        print('-------------sample epoch %s------------------'%phase)

        batch = data_buffers[phase].get_unique_minibatch(num_samples)
        relative_indices = batch[-1]
        states, actions, rewards, next_states = make_state(batch[:-1], info['DEVICE'], info['NORM_BY'])
        next_state = next_states[:,-1:]
        z, u_q = encoder_model(states)
        basedir = info['loaded_from'].replace('.pt', '_%s'%phase)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        if teacher_force:
            print("teacher forcing")
            canvas = next_state
            basepath = os.path.join(basedir, '_tf')
        else:
            canvas = 0.0*next_state
            basepath = os.path.join(basedir, '_')

        npcanvas = np.zeros_like(canvas.detach().cpu().numpy())
        npstates = states.cpu().numpy()
        npnextstate = deepcopy(next_state.cpu().numpy())
        npactions = actions.cpu().numpy()
        for bi in range(npstates.shape[0]): #states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
            print('sampling %s'%bi)
            for i in range(canvas.shape[1]):
                for j in range(canvas.shape[2]):
                    for k in range(canvas.shape[3]):
                        output = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1].detach(),
                                                            float_condition=z[bi:bi+1].detach(),
                                                            class_condition=actions[bi:bi+1].detach()))
                        #  use dummy output
                        #npcanvas[bi,i,j,k] = npnextstate[bi,i,j,k]
                        npcanvas[bi,i,j,k] = output[0,0,j,k].detach().numpy()
                        if teacher_force:
                            canvas[bi,i,j,k] = output[0,0,j,k]
            # npcanvas is 0 to 1, make it -1 to 1
            norm_pred = ((npcanvas[bi,0]/npcanvas[bi,0].max())*2)-1
            #norm_pred = npcanvas[bi,0]
            f,ax = plt.subplots(1,6, sharey=True, figsize=(8,2))
            for ns in range(4):
                ax[ns].imshow(npstates[bi,ns], vmin=-1, vmax=1)
                ax[ns].set_title('%s'%(ns))
            ax[4].imshow(npnextstate[bi,0], vmin=-1, vmax=1)
            ax[4].set_title('T%s'%(5))
            # this is bt 0 and a tiny num - prob need to stretch
            ax[5].imshow(norm_pred, vmin=-1, vmax=1)
            ax[5].set_title('P%s'%(5))
            print(norm_pred.min(), norm_pred.max())
            print(npnextstate.min(), npnextstate.max())
            fname = basepath+'%02d.png'%bi
            print('plotting %s'%fname)
            plt.savefig(fname)
            embed()



       # add the predicted codes to the input
#        yhat_batch = torch.sigmoid(pcnn_decoder(x=next_state,
#                                                class_condition=actions,
#                                                float_condition=z))
#        prior_model.codes[relative_indices] = u_q.detach().cpu().numpy()
#
#
#
#        mix, u_ps, s_ps = prior_model(u_q)
#        canvas = 0.0*next_state


#        print('writing img')
#        n_imgs = 8
#        n = min(states.shape[0], n_imgs)
#        #onext_states = torch.Tensor(next_states[:n].data.cpu().numpy()+train_data_loader.frames_mean)#*train_data_loader.frames_diff) + train_data_loader.frames_min)
#        #oyhat_batch =  torch.Tensor( yhat_batch[:n].data.cpu().numpy()+train_data_loader.frames_mean)#*train_data_loader.frames_diff) + train_data_loader.frames_min)
#        #onext_states = torch.Tensor(((next_states[:n].data.cpu().numpy()*train_data_loader.frames_diff)+train_data_loader.frames_min) + train_data_loader.frames_mean)/255.
#        #oyhat_batch =  torch.Tensor((( yhat_batch[:n].data.cpu().numpy()*train_data_loader.frames_diff)+train_data_loader.frames_min) + train_data_loader.frames_mean)/255.
#        bs = args.batch_size
#        h = train_data_loader.data_h
#        w = train_data_loader.data_w
#        comparison = torch.cat([next_states.view(bs,1,h,w)[:n],
#                                yhat_batch.view(bs,1,h,w)[:n]])
#        img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
#        save_image(comparison, img_name, nrow=n)
#        #embed()
#        #ocomparison = torch.cat([onext_states,
#        #                        oyhat_batch])
#        #img_name = model_base_filepath + "_%010d_valid_reconstructionMINE.png"%train_cnt
#        #save_image(ocomparison, img_name, nrow=n)
#        #embed()
#        print('finished writing img', img_name)

def init_train():
    """ use args to setup inplace training """
    train_data_path = args.train_buffer
    valid_data_path = args.valid_buffer

    data_dir = os.path.split(train_data_path)[0]

    # we are starting from scratch training this model
    if args.model_loadpath == "":
        run_num = 0
        model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        print("MODEL BASE FILEPATH", model_base_filepath)

        info = {'model_train_cnts':[],
                'model_train_losses':{},
                'model_valid_cnts':[],
                'model_valid_losses':{},
                'model_save_times':[],
                'model_last_save':0,
                'model_last_plot':0,
                'NORM_BY':255.0,
                'MODEL_BASE_FILEDIR':model_base_filedir,
                'model_base_filepath':model_base_filepath,
                'model_train_data_file':train_data_path,
                'model_valid_data_file':valid_data_path,
                'NUM_TRAINING_EXAMPLES':args.num_training_examples,
                'NUM_K':args.num_k,
                'NR_LOGISTIC_MIX':args.nr_logistic_mix,
                'NUM_PCNN_FILTERS':args.num_pcnn_filters,
                'NUM_PCNN_LAYERS':args.num_pcnn_layers,
                'ALPHA_REC':args.alpha_rec,
                'ALPHA_ACT':args.alpha_act,
                'ALPHA_REW':args.alpha_rew,
                'MODEL_BATCH_SIZE':args.batch_size,
                'NUMBER_CONDITION':args.num_condition,
                'CODE_LENGTH':args.code_length,
                'NUM_MIXTURES':args.num_mixtures,
                'REQUIRE_UNIQUE_CODES':args.require_unique_codes,
                 }

        ## size of latents flattened - dependent on architecture
        #info['float_condition_size'] = 100*args.num_z
        ## 3x logistic needed for loss
        ## TODO - change loss
    else:
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath, map_location=lambda storage, loc:storage)
        info =  model_dict['model_info']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        info['loaded_from'] = args.model_loadpath
        info['MODEL_BATCH_SIZE'] = args.batch_size
    info['DEVICE'] = DEVICE
    info['MODEL_SAVE_EVERY'] = args.save_every
    info['MODEL_LOG_EVERY_BATCHES'] = args.log_every_batches
    info['model_loadpath'] = args.model_loadpath
    info['MODEL_SAVENAME'] = args.savename
    info['MODEL_LEARNING_RATE'] = args.learning_rate
    # create replay buffer
    train_buffer = make_subset_buffer(train_data_path, max_examples=info['NUM_TRAINING_EXAMPLES'])
    valid_buffer = make_subset_buffer(valid_data_path, max_examples=int(info['NUM_TRAINING_EXAMPLES']*.1))
    #valid_buffer = ReplayMemory(load_file=valid_data_path)
    # if train buffer is too large - make random subset
    # 27588 places in 1e6 buffer where reward is nonzero

    info['num_actions'] = train_buffer.num_actions()
    info['size_training_set'] = train_buffer.num_examples()
    info['hsize'] = train_buffer.frame_height
    info['wsize'] = train_buffer.frame_width
    info['num_rewards'] = train_buffer.num_rewards()
    info['HISTORY_SIZE'] = 4


    rewards_weight = 1-np.array(train_buffer.percentages_rewards())
    actions_weight = 1-np.array(train_buffer.percentages_actions())
    actions_weight = torch.FloatTensor(actions_weight).to(DEVICE)
    rewards_weight = torch.FloatTensor(rewards_weight).to(DEVICE)
    info['actions_weight'] = actions_weight
    info['rewards_weight'] = rewards_weight


    # output mixtures should be 2*nr_logistic_mix + nr_logistic mix for each
    # decorelated channel
    #info['num_output_mixtures']= (2*args.nr_logistic_mix+args.nr_logistic_mix)*info['HISTORY_SIZE']
    #nmix = int(info['num_output_mixtures']/info['HISTORY_SIZE'])
    #info['nmix'] = nmix
    encoder_model = ConvVAE(info['CODE_LENGTH'], input_size=args.num_condition,
                            encoder_output_size=args.encoder_output_size,
                             ).to(DEVICE)
    prior_model = PriorNetwork(size_training_set=info['NUM_TRAINING_EXAMPLES'],
                               code_length=info['CODE_LENGTH'],
                               n_mixtures=info['NUM_MIXTURES'],
                               k=info['NUM_K'],
                               require_unique_codes=info['REQUIRE_UNIQUE_CODES'],
                               ).to(DEVICE)
    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=info['NUM_PCNN_FILTERS'],
                                 n_layers=info['NUM_PCNN_LAYERS'],
                                 n_classes=info['num_actions'],
                                 float_condition_size=info['CODE_LENGTH'],
                                 last_layer_bias=0.5,
                                 hsize=info['hsize'], wsize=info['wsize']).to(DEVICE)

    #parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    parameters = list(encoder_model.parameters()) + list(prior_model.parameters())
    opt = optim.Adam(parameters, lr=info['MODEL_LEARNING_RATE'])

    if args.model_loadpath != '':
        print("loading weights from:%s" %args.model_loadpath)
        encoder_model.load_state_dict(model_dict['encoder_model_state_dict'])
        prior_model.load_state_dict(model_dict['prior_model_state_dict'])
        pcnn_decoder.load_state_dict(model_dict['pcnn_decoder_state_dict'])
        #encoder_model.embedding = model_dict['model_embedding']
        opt.load_state_dict(model_dict['opt_state_dict'])

    model_dict = {'encoder_model':encoder_model,
                  'prior_model':prior_model,
                  'pcnn_decoder':pcnn_decoder,
                  'opt':opt}
    data_buffers = {'train':train_buffer, 'valid':valid_buffer}
    if args.sample:
        sample_acn(info, model_dict, data_buffers, num_samples=args.num_samples, teacher_force=args.teacher_force)
    else:
        train_acn(info, model_dict, data_buffers)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005679481_train.npz')
    parser.add_argument('--valid_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval.npz')
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-ns', '--num_samples', default=5, type=int)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('--savename', default='acn')
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    parser.add_argument('-se', '--save_every', default=100000*2, type=int)
    parser.add_argument('-le', '--log_every_batches', default=50, type=int)


    parser.add_argument('-bs', '--batch_size', default=84, type=int)
    # 4x40x40 input -> 768 output
    parser.add_argument('-eos', '--encoder_output_size', default=768, type=int)
    parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=48, type=int)
    parser.add_argument('-ncond', '--num_condition', default=4, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-ne', '--num_training_examples', default=100000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    #parser.add_argument('-pv', '--possible_values', default=1)
    parser.add_argument('-npcnn', '--num_pcnn_layers', default=6)
    parser.add_argument('-pf', '--num_pcnn_filters', default=16, type=int)
    parser.add_argument('-nm', '--num_mixtures', default=8, type=int)
    parser.add_argument('--alpha_act', default=2, type=float, help='scale for last action prediction')
    parser.add_argument('--alpha_rew', default=1, type=float, help='scale for reward prediction')
    parser.add_argument('--alpha_rec', default=2, type=float, help='scale for rec prediction')
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    if args.debug:
        args.save_every = 10
        args.plot_every = 10
        args.log_every = 10

    init_train()

