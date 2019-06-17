import os
import time
import numpy as np
from torch import optim
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_
from forward_conv import ForwardResNet, BasicBlock
torch.set_num_threads(4)
from torchvision.utils import save_image
from ae_utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from IPython import embed
from lstm_utils import plot_dict_losses
from ae_utils import save_checkpoint
from datasets import ForwardLatentDataset
from vqvae import VQVAE
import sys
sys.path.append('../agents')
from replay import ReplayMemory
torch.manual_seed(394)

"""
things to try
- training multiple steps out with rollout
- mask so there is a loss on changed latents only between steps

"""

train_data_file = args.train_data_file
valid_data_file = args.valid_data_file

setup(data_dir=args.datadir, savename=args.savename, train_data_file=args.train_data_file, model_loadpath=args.model_loadpath,
      )

def setup(data_dir, savename, train_data_file, model_loadpath=''):
    data_dir = os.path.split(train_data_file)[0]
    train_buffer = ReplayMemory(load_file=train_data_file)


    if args.model_loadpath == '':
         train_cnt = 0
         run_num = 0
         model_base_filedir = os.path.join(data_dir, savename + '%02d'%run_num)
         while os.path.exists(model_base_filedir):
             run_num +=1
             model_base_filedir = os.path.join(data_dir, savename + '%02d'%run_num)
         os.makedirs(model_base_filedir)
         model_base_filepath = os.path.join(model_base_filedir, savename)
         print("MODEL BASE FILEPATH", model_base_filepath)

         info = {'train_cnts':[],
                 'train_losses_list':[],
                 'valid_cnts':[],
                 'valid_losses_list':[],
                 'save_times':[],
                 'savename':savename,
                 'data_dir':datadir,
                 #'args':[args],
                 'last_save':0,
                 'last_plot':0,
                 'reward_weights':[1,100], # should be same as num_rewards
                  }
    else:
        print('loading model from: %s' %model_loadpath)
        model_dict = torch.load(model_loadpath)
        info =  model_dict['info']
        model_base_filedir = os.path.split(model_loadpath)[0]
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        train_cnt = info['train_cnts'][-1]
        info['loaded_from'] = model_loadpath
        if 'reward_weights' not in info.keys():
            info['reward_weights'] = [1,100]

    num_actions = info['num_actions'] = n_actions
    num_rewards = info['num_rewards'] = len(train_data_loader.unique_rewards)
    args.size_training_set = train_data_loader.num_examples
    hsize = train_data_loader.data_h
    wsize = train_data_loader.data_w
    info['num_rewards'] = len(train_data_loader.unique_rewards)

    info['hsize'] = hsize
    info['num_channels'] = num_actions+1+1

    #  !!!! TODO save this in npz and pull out
    #num_k = info['num_k'] = 512
    ###########################################3
    # load vq model
    vq_model_dict = torch.load(args.vq_model_loadpath, map_location=lambda storage, loc: storage)
    vq_info = vq_model_dict['info']
    vq_largs = vq_info['args'][-1]
    nmix = int(vq_info['num_output_mixtures']/2)
    ###########################################3
    num_k = vq_largs.num_k
    vqvae_model = VQVAErl(num_clusters=num_k,
                        encoder_output_size=vq_largs.num_z,
                        num_output_mixtures=vq_info['num_output_mixtures'],
                        in_channels_size=vq_largs.number_condition,
                        n_actions=vq_info['num_actions'],
                        int_reward=vq_info['num_rewards']).to(DEVICE)
    vqvae_model.load_state_dict(vq_model_dict['vqvae_state_dict'])
    vqvae_model.eval()
    #conv_forward_model = ForwardResNet(BasicBlock, data_width=info['hsize'],
    #                                   num_channels=info['num_channels'],
    #                                   num_actions=num_actions,
    #                                   num_output_channels=num_k,
    #                                   num_rewards=num_rewards,
    #                                   dropout_prob=args.dropout_prob).to(DEVICE)


    conv_forward_model = ForwardResNet(BasicBlock, data_width=info['hsize'],
                                       num_channels=info['num_channels'],
                                       num_output_channels=num_k,
                                       dropout_prob=args.dropout_prob).to(DEVICE)

    # reweight the data based on its frequency
    info['actions_weight'] = 1-np.array(train_data_loader.percentages_actions)
    info['rewards_weight'] = 1-np.array(train_data_loader.percentages_rewards)
    actions_weight = torch.FloatTensor(info['actions_weight']).to(DEVICE)
    rewards_weight = torch.FloatTensor(info['rewards_weight']).to(DEVICE)
    parameters = list(conv_forward_model.parameters())
    opt = optim.Adam(parameters, lr=args.learning_rate)
    if args.model_loadpath != '':
        conv_forward_model.load_state_dict(model_dict['conv_forward_model'])
        opt.load_state_dict(model_dict['optimizer'])
    #args.pred_output_size = 1*80*80
    ## 10 is result of structure of network
    #args.z_input_size = 10*10*args.num_z
    train_cnt = train_forward(train_cnt)


def run():
    data = train_data_loader.get_minibatch()
    prev_latents, prev_actions, prev_rewards, prev_values, latents, actions, rewards, values, next_latents, is_new_epoch, data_indexes = data
    train_forward(train_cnt, conv_forward_model, opt, latents, actions, rewards, next_latents)

def train_forward(train_cnt, conv_forward_model, opt, latents, actions, rewards, next_latents):
    st = time.time()
    batches = 0
    while train_cnt < args.num_examples_to_train:
        conv_forward_model.train()
        opt.zero_grad()
        # we want the forward model to produce a next latent in which the vq
        # model can determine the action we gave it.
        latents = torch.FloatTensor(latents[:,None]).to(DEVICE)
        # next_latents is long because of prediction
        next_latents = torch.LongTensor(next_latents[:,None]).to(DEVICE)
        rewards = torch.LongTensor(rewards).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        # put actions into channel for conditioning
        bs, _, h, w = latents.shape
        channel_actions = torch.zeros((bs,num_actions,h,w)).to(DEVICE)
        for a in range(num_actions):
            channel_actions[actions==a,a] = 1
        channel_rewards = torch.zeros((bs,num_rewards,h,w)).to(DEVICE)
        for r in range(num_rewards):
            channel_rewards[rewards==r,r] = 1


        # combine input together
        state_input = torch.cat((channel_actions, channel_rewards, latents, next_latents), dim=1)
        bs = float(latents.shape[0])

        pred_next_latents = conv_forward_model(state_input)
        # pred_next_latents shape is bs, c, h, w - need to permute shape for
        # don't optimize vqmodel - just optimize against its understanding of
        # the latent data
        with torch.no_grad():
            N, _, H, W = latents.shape
            C = vq_largs.num_z
            pred_next_latent_inds = torch.argmax(pred_next_latents, dim=1)
            x_tilde, pred_z_q_x, pred_actions, pred_rewards = vqvae_model.decode_clusters(pred_next_latent_inds, N, H, W, C)

        # should be able to predict the input action that got us to this
        # timestep
        loss_act = args.alpha_act*F.nll_loss(pred_actions, actions, weight=actions_weight)
        loss_reward = args.alpha_rew*F.nll_loss(pred_rewards, rewards, weight=rewards_weight)

        # determine which values of latents change over this time step
        ts_change = (torch.abs(latents.long()-next_latents)>1).view(-1)

        pred_next_latents = pred_next_latents.permute(0,2,3,1).contiguous()
        next_latents = next_latents.permute(0,2,3,1).contiguous()
        latents = latents.permute(0,2,3,1).contiguous()
        loss_rec = args.alpha_rec*F.nll_loss(pred_next_latents.view(-1, num_k), next_latents.view(-1), reduction='mean')

        # we want to penalize when these are wrong in particular
        loss_diff_rec = args.alpha_rec*F.nll_loss(pred_next_latents.view(-1, num_k)[ts_change==1], next_latents.view(-1)[ts_change==1], reduction='mean')

        loss = loss_reward+loss_act+loss_rec+loss_diff_rec
        loss.backward(retain_graph=True)
        parameters = list(conv_forward_model.parameters())
        clip_grad_value_(parameters, 10)
        opt.step()
        loss_list = [loss_reward.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff_rec.item()/bs]
        if batches > 1000:
            handle_checkpointing(train_cnt, loss_list)
        train_cnt+=bs
        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def valid_forward(train_cnt, do_plot=False):
    conv_forward_model.eval()
    opt.zero_grad()
    valid_data = valid_data_loader.get_minibatch()
    prev_latents, prev_actions, prev_rewards, prev_values, latents, actions, rewards, values, next_latents, is_new_epoch, data_indexes = valid_data
    # want to predict the action taken between the prev and observed
    # we are given the current action and need to predict next latent
    prev_latents = torch.FloatTensor(prev_latents[:,None]).to(DEVICE)
    latents = torch.FloatTensor(latents[:,None]).to(DEVICE)
    # next_latents is long because of prediction
    next_latents = torch.LongTensor(next_latents[:,None]).to(DEVICE)

    # put actions into channel for conditioning
    bs, _, h, w = latents.shape
    rewards = torch.LongTensor(rewards).to(DEVICE)
    actions = torch.LongTensor(actions).to(DEVICE)
    channel_actions = torch.zeros((bs,num_actions,h,w)).to(DEVICE)
    for a in range(num_actions):
        channel_actions[actions==a,a] = 1.0
    # combine input together
    state_input = torch.cat((channel_actions, prev_latents, latents), dim=1)
    bs = float(latents.shape[0])
    pred_next_latents = conv_forward_model(state_input)

    # don't optimize vqmodel - we are just trying to figure out how good the
    # prediction was
    with torch.no_grad():
        N, _, H, W = latents.shape
        C = vq_largs.num_z
        pred_next_latent_inds = torch.argmax(pred_next_latents, dim=1)
        x_d, pred_z_q_x, pred_actions, pred_rewards = vqvae_model.decode_clusters(pred_next_latent_inds, N, H, W, C)

    # determine which values of latents change over this time step
    ts_change = (torch.abs(latents.long()-next_latents)>1).view(-1)

    # pred_next_latents shape is bs, c, h, w - need to permute shape for
    # cross entropy loss
    pred_next_latents = pred_next_latents.permute(0,2,3,1).contiguous()
    next_latents = next_latents.permute(0,2,3,1).contiguous()

    # log_softmax is done in the forward pass
    loss_rec = args.alpha_rec*F.nll_loss(pred_next_latents.view(-1, num_k), next_latents.view(-1), reduction='mean')

    loss_act = args.alpha_act*F.nll_loss(pred_actions, actions,    weight=actions_weight)
    loss_reward = args.alpha_rew*F.nll_loss(pred_rewards, rewards, weight=rewards_weight)
    # we want to penalize when these are wrong in particular
    loss_diff_rec = args.alpha_rec*F.nll_loss(pred_next_latents.view(-1, num_k)[ts_change==1], next_latents.view(-1)[ts_change==1], reduction='mean')
    # weight rewards according to the
    # cant do act because i dont have this data for the "next action"
    loss_list = [loss_reward.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff_rec.item()/bs]
    print('valid', loss_list)

    bs,yc,yh,yw = x_d.shape
    # only process 8 imgs
    n = min(bs, 8)
    rec_est =  x_d[:n, :nmix]
    #loss_rec = args.alpha_rec*discretized_mix_logistic_loss(rec_est, rec, nr_mix=vq_largs.nr_logistic_mix, DEVICE=DEVICE)
    yhat = sample_from_discretized_mix_logistic(rec_est, vq_largs.nr_logistic_mix)
    if do_plot:
        #gold = (rec.to('cpu')+1)/2.0
        bs,_,h,w = yhat.shape
        # sample from discretized should be between 0 and 255
        #print("yhat sample", yhat[:,0].min().item(), yhat[:,0].max().item())
        yimg = ((yhat + 1.0)/2.0).to('cpu')
        print("yhat img", yhat.min().item(), yhat.max().item())
        #print("gold img", gold.min().item(), gold.max().item())
        #comparison = torch.cat([gold.view(bs,1,h,w)[:n],
        #                        yimg.view(bs,1,h,w)[:n]])
        img_name = model_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
        save_image(yimg.view(n,1,h,w), img_name, nrow=n)



    return loss_list

if __name__ == '__main__':
    from argparse import ArgumentParser

    debug = 0
    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_data_file',
                        #default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward00/vqdiffactintreward_0118012272ex_train_forward.npz')
                        #default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0071507436ex_train_forward.npz')
                        #default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0131013624ex_train_forwarddebug.npz')
                        #default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0131013624ex_train_forward.npz')
                        default='../../model_savedir/MBFreeway_replay01/MBFreeway_replay_0000502997q_train_buffer.npz',)
    parser.add_argument('--valid_data_file', default='../../model_savedir/MBFreeway_replay01/MBFreeway_replay_0000002044q.pkl'),
    parser.add_argument('--vq_model_loadpath', default='../../model_savedir/FRANKbootstrap_priorfreeway00/vqdiffactintreward512q00/vqdiffactintreward512q_0035503692ex.pt'),
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--savename', default='Freeway_forward')
    #parser.add_argument('--savename', default='DEBUG')
    parser.add_argument('-l', '--model_loadpath', default='')
    if not debug:
        parser.add_argument('-se', '--save_every', default=100000*5, type=int)
        parser.add_argument('-pe', '--plot_every', default=100000*5, type=int)
        parser.add_argument('-le', '--log_every',  default=100000*5, type=int)
    else:
        parser.add_argument('-se', '--save_every', default=10, type=int)
        parser.add_argument('-pe', '--plot_every', default=10, type=int)
        parser.add_argument('-le', '--log_every',  default=10, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    # increased the alpha rec
    parser.add_argument('-ad', '--alpha_rew', default=4.0, type=float)
    parser.add_argument('-aa', '--alpha_act', default=2.0, type=float)
    parser.add_argument('-ar', '--alpha_rec', default=1.0, type=float)
    parser.add_argument('-d', '--dropout_prob', default=0.25, type=float)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=int(1e10), type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5)
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

