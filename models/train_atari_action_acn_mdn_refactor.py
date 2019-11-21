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
import os
import time
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2

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
    next_states = prepare_state(next_states, DEVICE, NORM_BY)
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
    filename = info['MODEL_BASE_FILEDIR'] + "_%010dex.pt"%train_cnt
    print("Saving model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['model_last_save']))
    print(filename)
    state = {
             'model_info':info,
             }
    for (model_name, model) in model_dict.items():
        state[model_name+'_state_dict'] = model.state_dict()
    save_checkpoint(state, filename=filename)
    return info


def find_rec_losses(alpha, nr, nmix, x_d, true, DEVICE):
    rec_losses = []
    rec_ests = []
    # get reconstruction losses for each channel
    for i in range(true.shape[1]):
        st = i*nmix
        en = st+nmix
        pred_x_d = x_d[:,st:en]
        rec_ests.append(pred_x_d.detach())
        rloss = alpha*discretized_mix_logistic_loss(pred_x_d, true[:,i][:,None], nr_mix=nr, DEVICE=DEVICE)
        rec_losses.append(rloss)
    return rec_losses, rec_ests


def train_acn(info, model_dict, train_buffer, valid_buffer, save_every_samples):


    if not 'avg_train_kl_loss' in info['model_train_losses'].keys():
        info['model_train_losses']['avg_train_kl_loss'] = []
    if not 'avg_train_rec_loss' in info['model_train_losses'].keys():
        info['model_train_losses']['avg_train_rec_loss'] = []


    encoder_model = model_dict['encoder_model'].train()
    prior_model = model_dict['prior_model'].train()
    pcnn_decoder = model_dict['pcnn_decoder'].train()
    opt = model_dict['opt']

    # add one to the rewards so that they are all positive
    # use next_states because that is the t-1 action

    if len(info['model_train_cnts']):
        train_cnt = info['model_train_cnts'][-1]
    else: train_cnt = 0

    num_batches = train_buffer.count//info['MODEL_BATCH_SIZE']
    while train_cnt < 10000000:
        batch_num = 0
        init_cnt = train_cnt
        train_buffer.reset_unique()
        train_kl_loss = 0.0
        train_rec_loss = 0.0
        print('-------------new epoch------------------')
        print('num batches', num_batches)
        while train_buffer.unique_available:
            opt.zero_grad()
            batch = train_buffer.get_unique_minibatch(info['MODEL_BATCH_SIZE'])
            relative_indices = batch[-1]
            states, actions, rewards, next_states = make_state(batch[:-1], info['DEVICE'], info['NORM_BY'])
            next_state = next_states[:,-1:]
            bs = states.shape[0]
            #states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_unique_minibatch()
            z, u_q = encoder_model(states)
            np_uq = u_q.detach().cpu().numpy()

            if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
                print('train bad')
                embed()

            # add the predicted codes to the input
            yhat_batch = torch.sigmoid(pcnn_decoder(x=next_state,
                                                    class_condition=actions,
                                                    float_condition=z))
            #yhat_batch = torch.sigmoid(pcnn_decoder(x=next_states, float_condition=z))
            prior_model.codes[relative_indices] = u_q.detach().cpu().numpy()
            np_uq = u_q.detach().cpu().numpy()

            if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
                print('train bad')
                embed()

            mix, u_ps, s_ps = prior_model(u_q)
            kl_loss, rec_loss = acn_mdn_loss_function(yhat_batch, next_state, u_q, mix, u_ps, s_ps)
            loss = kl_loss + rec_loss
            loss.backward()
            parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
            clip_grad_value_(parameters, 10)
            train_kl_loss+= kl_loss.item()
            train_rec_loss+= rec_loss.item()
            opt.step()
            # add batch size because it hasn't been added to train cnt yet
            avg_train_kl_loss = train_kl_loss/float((train_cnt+bs)-init_cnt)
            avg_train_rec_loss = train_rec_loss/float((train_cnt+bs)-init_cnt)
            train_cnt += bs
            batch_num+=1
            if not batch_num%10:
                print(train_cnt, batch_num, avg_train_kl_loss, avg_train_rec_loss)
            if (((train_cnt-info['model_last_save'])>=save_every_samples)):
                #valid_losses = valid_buffer.get_minibatch(info['MODEL_BATCH_SIZE'])
                info['model_train_cnts'].append(train_cnt)
                info['model_train_losses']['avg_train_kl_loss'].append(avg_train_kl_loss)
                info['model_train_losses']['avg_train_rec_loss'].append(avg_train_rec_loss)
                info = save_model(info, model_dict)

        model_dict = {'encoder_model':encoder_model,
                     'prior_model':prior_model,
                     'pcnn_decoder':pcnn_decoder,
                     'opt':opt}


#def valid_vqvae(train_cnt, vqvae_model, info, batch):
#    vqvae_model.eval()
#    states, actions, rewards, next_states = make_state(batch, info['DEVICE'], info['NORM_BY'])
#    # use next_states because that is the t-1 action
#    x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(next_states)
#    z_q_x.retain_grad()
#    rec_losses, rec_ests = find_rec_losses(alpha=info['ALPHA_REC'],
#                                 nr=info['NR_LOGISTIC_MIX'],
#                                 nmix=info['nmix'],
#                                 x_d=x_d, true=next_states,
#                                 DEVICE=info['DEVICE'])
#
#    loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
#    loss_reward = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
#    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
#    loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())
#    vqvae_model.embedding.zero_grad()
#
#    bs = float(x_d.shape[0])
#    avg_valid_losses = [loss_reward.item()/bs, loss_act.item()/bs,
#                        rec_losses[0].item()/bs, rec_losses[1].item()/bs,
#                        rec_losses[2].item()/bs, rec_losses[3].item()/bs,
#                        loss_2.item()/bs, loss_3.item()/bs]
#
#    bs,yc,yh,yw = x_d.shape
#    n = min(next_states.shape[0],5)
#    # last state
#    yhat_t = sample_from_discretized_mix_logistic(rec_ests[-1][:n], info['NR_LOGISTIC_MIX']).cpu().numpy()
#    yhat_tm1 = sample_from_discretized_mix_logistic(rec_ests[-2][:n], info['NR_LOGISTIC_MIX']).cpu().numpy()
#    true_t = next_states[:n,-1].cpu().numpy()
#    true_tm1 = next_states[:n,-2].cpu().numpy()
#    print("yhat img", yhat_t.min().item(), yhat_t.max().item())
#    print("true img", true_t.min().item(), true_t.max().item())
#    img_name = info['MODEL_BASE_FILEDIR'] + "_%010d_valid_reconstruction.png"%train_cnt
#    f,ax=plt.subplots(n,4, figsize=(4*2, n*2))
#    for nn in range(n):
#        ax[nn, 0].imshow(true_tm1[nn], vmax=1, vmin=-1)
#        ax[nn, 0].set_title('TA%s'%int(actions[nn]))
#        ax[nn, 1].imshow(true_t[nn], vmax=1, vmin=-1)
#        ax[nn, 1].set_title('TR%s'%int(rewards[nn]))
#        ax[nn, 2].imshow(yhat_tm1[nn,0], vmax=1, vmin=-1)
#        ax[nn, 2].set_title('PA%s'%int(torch.argmax(pred_actions[nn])))
#        ax[nn, 3].imshow(yhat_t[nn,0], vmax=1, vmin=-1)
#        ax[nn, 3].set_title('PR%s'%int(torch.argmax(pred_rewards[nn])))
#        for i in range(4):
#            ax[nn,i].axis('off')
#
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    plt.savefig(img_name)
#    plt.close()
#
#    #img_name2 = info['model_model_base_filepath'] + "_%010d_valid_reconstruction2.png"%train_cnt
#    img_name2 = info['MODEL_BASE_FILEDIR'] + "_%010d_valid_reconstruction2.png"%train_cnt
#    f,ax=plt.subplots(n,4, figsize=(4*2, n*2))
#    for nn in range(n):
#        ax[nn, 0].imshow(true_tm1[nn], vmax=1, vmin=0)
#        ax[nn, 0].set_title('TA%s'%int(actions[nn]))
#        ax[nn, 1].imshow(true_t[nn], vmax=1, vmin=0)
#        ax[nn, 1].set_title('TR%s'%int(rewards[nn]))
#        ax[nn, 2].imshow(yhat_tm1[nn,0])
#        ax[nn, 2].set_title('PA%s'%int(torch.argmax(pred_actions[nn])))
#        ax[nn, 3].imshow(yhat_t[nn,0])
#        ax[nn, 3].set_title('PR%s'%int(torch.argmax(pred_rewards[nn])))
#        for i in range(4):
#            ax[nn,i].axis('off')
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    plt.savefig(img_name2)
#
#
#    #bs,h,w = gold.shape
#    ## sample from discretized should be between 0 and 255
#    #print("yhat sample", yhat[:,0].min().item(), yhat[:,0].max().item())
#    #yimg = ((yhat + 1.0)/2.0).to('cpu')
#    #print("yhat img", yhat.min().item(), yhat.max().item())
#    #print("gold img", gold.min().item(), gold.max().item())
#    #comparison = torch.cat([gold.view(bs,1,h,w)[:n],
#    #                        yimg.view(bs,1,h,w)[:n]])
#    #save_image(comparison, img_name, nrow=n)
#    return avg_valid_losses
#


#def valid_acn(train_cnt, do_plot):
#    valid_kl_loss = 0.0
#    valid_rec_loss = 0.0
#    print('starting valid', train_cnt)
#    st = time.time()
#    valid_cnt = 0
#    encoder_model.eval()
#    prior_model.eval()
#    pcnn_decoder.eval()
#    opt.zero_grad()
#    i = 0
#    states, actions, rewards, next_states, terminals, is_new_epoch, relative_indexes = valid_data_loader.get_unique_minibatch()
#    states = states.to(DEVICE)
#    # 1 channel expected
#    next_states = next_states[:,args.num_condition-1:].to(DEVICE)
#    actions = actions.to(DEVICE)
#    z, u_q = encoder_model(states)
#
#    np_uq = u_q.detach().cpu().numpy()
#    if np.isinf(np_uq).sum() or np.isnan(np_uq).sum():
#        print('baad')
#        embed()
#
#    #yhat_batch = encoder_model.decode(u_q, s_q, data)
#    # add the predicted codes to the input
#    yhat_batch = torch.sigmoid(pcnn_decoder(x=next_states, class_condition=actions, float_condition=z))
#    mix, u_ps, s_ps = prior_model(u_q)
#    kl_loss,rec_loss = acn_mdn_loss_function(yhat_batch, next_states, u_q, mix, u_ps, s_ps)
#    valid_kl_loss+= kl_loss.item()
#    valid_rec_loss+= rec_loss.item()
#    valid_cnt += states.shape[0]
#    if i == 0 and do_plot:
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
#    valid_kl_loss/=float(valid_cnt)
#    valid_rec_loss/=float(valid_cnt)
#    print('====> valid kl loss: {:.4f}'.format(valid_kl_loss))
#    print('====> valid rec loss: {:.4f}'.format(valid_rec_loss))
#    print('finished valid', time.time()-st)
#    return valid_kl_loss, valid_rec_loss


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
                'model_loadpath':args.model_loadpath,
                'MODEL_BASE_FILEDIR':model_base_filedir,
                'model_base_filepath':model_base_filepath,
                'model_train_data_file':train_data_path,
                'model_valid_data_file':valid_data_path,
                'NUM_TRAINING_EXAMPLES':args.num_training_examples,
                'MODEL_SAVENAME':args.savename,
                'DEVICE':DEVICE,
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
                'MODEL_LEARNING_RATE':args.learning_rate,
                'MODEL_SAVE_EVERY':args.save_every,
                 }

        ## size of latents flattened - dependent on architecture
        #info['float_condition_size'] = 100*args.num_z
        ## 3x logistic needed for loss
        ## TODO - change loss
    else:
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info =  model_dict['model_info']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        info['loaded_from'] = args.model_loadpath
        info['MODEL_BATCH_SIZE'] = args.batch_size
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

    parameters = list(encoder_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
    opt = optim.Adam(parameters, lr=info['MODEL_LEARNING_RATE'])

    if args.model_loadpath != '':
        print("loading weights from:%s" %args.model_loadpath)
        encoder_model.load_state_dict(model_dict['encoder_state_dict'])
        prior_model.load_state_dict(model_dict['prior_state_dict'])
        pcnn_decoder.load_state_dict(model_dict['pcnn_decoder_state_dict'])
        #encoder_model.embedding = model_dict['model_embedding']
        opt.load_state_dict(model_dict['model_optimizer'])

    model_dict = {'encoder_model':encoder_model,
                  'prior_model':prior_model,
                  'pcnn_decoder':pcnn_decoder,
                  'opt':opt}

    train_acn(info, model_dict, train_buffer, valid_buffer, save_every_samples=args.save_every)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005679481_train.npz')
    parser.add_argument('--valid_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('--savename', default='acn')
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    parser.add_argument('-se', '--save_every', default=100000*2, type=int)
    parser.add_argument('-pe', '--plot_every', default=100000*2, type=int)
    parser.add_argument('-le', '--log_every',  default=100000*2, type=int)


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
    parser.add_argument('-lr', '--learning_rate', default=1e-4)
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

