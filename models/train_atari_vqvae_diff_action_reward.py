import os
import sys
import time
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from torch import optim
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_
from ae_utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from ae_utils import handle_plot_ckpt, reshape_input
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from IPython import embed
from ae_utils import save_checkpoint
from vqvae import VQVAErl
from datasets import AtariDataset
torch.manual_seed(394)
sys.path.append('../agents')
from replay import ReplayMemory

def make_state(batch, DEVICE, NORM_BY):

    states, actions, rewards, next_states, terminal_flags, masks, latent_states, latent_next_states = batch
    # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
    states = (2*reshape_input(torch.FloatTensor(states)/NORM_BY)-1).to(DEVICE)
    next_states = (2*reshape_input(torch.FloatTensor(next_states)[:,-1:]/NORM_BY)-1).to(DEVICE)
    diff = states[:,-1:]-next_states
    #diff = (reshape_input(torch.FloatTensor(pred_states)[:,1][:,None])).to(DEVICE)
    actions = torch.LongTensor(actions).to(DEVICE)
    rewards = torch.LongTensor(rewards).to(DEVICE)
    bs, _, h, w = states.shape
    # combine input together
    return states, actions, rewards, next_states, diff

def find_rec_loss(alpha, est, true, nr_mix, device):
    return alpha*discretized_mix_logistic_loss(est, true, nr_mix=nr_mix, DEVICE=device)

def run(info, vqvae_model, opt, train_buffer, valid_buffer, num_samples_to_train=10000, save_every_samples=1000):
    batches = 0
    if len(info['vq_train_cnts']):
        train_cnt = info['vq_train_cnts'][-1]
    else:
        train_cnt = 0
    while train_cnt < num_samples_to_train:
        st = time.time()
        batch = train_buffer.get_minibatch(info['VQ_BATCH_SIZE'])
        avg_train_losses, vqvae_model, opt = train_vqvae(vqvae_model, opt, info, batch)
        batches+=1
        train_cnt+=info['VQ_BATCH_SIZE']
        if ((train_cnt-info['vq_last_save'])>=save_every_samples):
            info['vq_last_save'] = train_cnt
            info['vq_save_times'].append(time.time())
            valid_batch = valid_buffer.get_minibatch(info['VQ_BATCH_SIZE'])
            avg_valid_losses = valid_vqvae(train_cnt, vqvae_model, info, valid_batch)
            handle_plot_ckpt(train_cnt, info, avg_train_losses, avg_valid_losses)
            filename = info['vq_model_base_filepath'] + "_%010dex.pt"%train_cnt
            print("SAVING MODEL:%s" %filename)
            print("Saving model at cnt:%s cnt since last saved:%s"%(train_cnt, train_cnt-info['vq_last_save']))
            state = {
                     'vqvae_state_dict':vqvae_model.state_dict(),
                     'vq_optimizer':opt.state_dict(),
                     'vq_embedding':vqvae_model.embedding,
                     'vq_info':info,
                     }
            save_checkpoint(state, filename=filename)

        batches+=1
        if not batches%10:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))

def train_vqvae(vqvae_model, opt, info, batch):
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    vqvae_model.train()
    opt.zero_grad()
    #states, actions, rewards, values, pred_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_framediff_minibatch()
    states,  actions, rewards, next_states, next_state_diffs = make_state(batch, info['DEVICE'], info['NORM_BY'])
    x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(states)
    z_q_x.retain_grad()
    loss_rec = find_rec_loss(alpha=info['ALPHA_REC'], est=x_d[:,:info['nmix']], true=next_states, nr_mix=info['NR_LOGISTIC_MIX'], device=info['DEVICE'])
    loss_diff = find_rec_loss(alpha=info['ALPHA_REC'], est=x_d[:,info['nmix']:], true=next_state_diffs, nr_mix=info['NR_LOGISTIC_MIX'], device=info['DEVICE'])

    loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
    loss_reward = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())
    vqvae_model.embedding.zero_grad()

    loss_act.backward(retain_graph=True)
    loss_reward.backward(retain_graph=True)
    loss_rec.backward(retain_graph=True)
    loss_diff.backward(retain_graph=True)
    z_e_x.backward(z_q_x.grad, retain_graph=True)
    loss_2.backward(retain_graph=True)
    loss_3.backward()

    parameters = list(vqvae_model.parameters())
    clip_grad_value_(parameters, 10)
    opt.step()
    bs = float(x_d.shape[0])
    avg_train_losses = [loss_reward.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
    opt.zero_grad()
    return avg_train_losses, vqvae_model, opt

def valid_vqvae(train_cnt, vqvae_model, info, batch):
    vqvae_model.eval()
    states,  actions, rewards, next_states, next_state_diffs = make_state(batch, info['DEVICE'], info['NORM_BY'])
    x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(states)
    z_q_x.retain_grad()

    rec_est =  x_d[:, :info['nmix']]
    diff_est = x_d[:, info['nmix']:]
    loss_rec = find_rec_loss(alpha=info['ALPHA_REC'], est=rec_est, true=next_states, nr_mix=info['NR_LOGISTIC_MIX'], device=info['DEVICE'])
    loss_diff = find_rec_loss(alpha=info['ALPHA_REC'], est=diff_est, true=next_state_diffs, nr_mix=info['NR_LOGISTIC_MIX'], device=info['DEVICE'])

    loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
    loss_reward = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())


    bs,yc,yh,yw = x_d.shape
    yhat = sample_from_discretized_mix_logistic(rec_est, info['NR_LOGISTIC_MIX'])
    n_imgs = 8
    n = min(states.shape[0], n_imgs)
    gold = (next_states.to('cpu')+1)/2.0
    bs,_,h,w = gold.shape
    # sample from discretized should be between 0 and 255
    print("yhat sample", yhat[:,0].min().item(), yhat[:,0].max().item())
    yimg = ((yhat + 1.0)/2.0).to('cpu')
    print("yhat img", yhat.min().item(), yhat.max().item())
    print("gold img", gold.min().item(), gold.max().item())
    comparison = torch.cat([gold.view(bs,1,h,w)[:n],
                            yimg.view(bs,1,h,w)[:n]])
    img_name = info['vq_model_base_filepath'] + "_%010d_valid_reconstruction.png"%train_cnt
    save_image(comparison, img_name, nrow=n)
    bs = float(states.shape[0])
    loss_list = [loss_reward.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
    return loss_list

def init_train():
    train_data_file = args.train_buffer
    data_dir = os.path.split(train_data_file)[0]
    valid_data_file = args.valid_buffer
    #valid_data_file = '/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/valid_set_small.npz'
    if args.model_loadpath == '':
         train_cnt = 0
         run_num = 0
         model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
         while os.path.exists(model_base_filedir):
             run_num +=1
             model_base_filedir = os.path.join(data_dir, args.savename + '%02d'%run_num)
         os.makedirs(model_base_filedir)
         model_base_filepath = os.path.join(model_base_filedir, args.savename)
         print("MODEL BASE FILEPATH", model_base_filepath)

         info = {'vq_train_cnts':[],
                 'vq_train_losses_list':[],
                 'vq_valid_cnts':[],
                 'vq_valid_losses_list':[],
                 'vq_save_times':[],
                 'vq_last_save':0,
                 'vq_last_plot':0,
                 'NORM_BY':255.0,
                 'vq_model_loadpath':args.model_loadpath,
                 'vq_model_base_filedir':model_base_filedir,
                 'vq_model_base_filepath':model_base_filepath,
                 'vq_train_data_file':train_data_file,
                 'VQ_SAVENAME':args.savename,
                 'DEVICE':DEVICE,
                 'NUM_Z':args.num_z,
                 'NUM_K':args.num_k,
                 'NR_LOGISTIC_MIX':args.nr_logistic_mix,
                 'BETA':args.beta,
                 'ALPHA_REC':args.alpha_rec,
                 'ALPHA_ACT':args.alpha_act,
                 'ALPHA_REW':args.alpha_rew,
                 'VQ_BATCH_SIZE':args.batch_size,
                 'NUMBER_CONDITION':args.number_condition,
                 'VQ_LEARNING_RATE':args.learning_rate,
                 'VQ_SAVE_EVERY':args.save_every,
                  }

         ## size of latents flattened - dependent on architecture of vqvae
         #info['float_condition_size'] = 100*args.num_z
         ## 3x logistic needed for loss
         ## TODO - change loss
    else:
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info =  model_dict['vq_info']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        model_base_filepath = os.path.join(model_base_filedir, args.savename)
        train_cnt = info['vq_train_cnts'][-1]
        info['loaded_from'] = args.model_loadpath
        info['VQ_BATCH_SIZE'] = args.batch_size
        #if 'reward_weights' not in info.keys():
        #    info['reward_weights'] = [1,100]
    # create replay buffer
    train_buffer = ReplayMemory(load_file=train_data_file)
    valid_buffer = ReplayMemory(load_file=valid_data_file)

    info['num_actions'] = train_buffer.num_actions()
    info['size_training_set'] = train_buffer.num_examples()
    info['hsize'] = train_buffer.frame_height
    info['wsize'] = train_buffer.frame_width
    info['num_rewards'] = train_buffer.num_rewards()

    rewards_weight = 1-np.array(train_buffer.percentages_rewards())
    actions_weight = 1-np.array(train_buffer.percentages_actions())
    actions_weight = torch.FloatTensor(actions_weight).to(DEVICE)
    rewards_weight = torch.FloatTensor(rewards_weight).to(DEVICE)
    info['actions_weight'] = actions_weight
    info['rewards_weight'] = rewards_weight

    # output mixtures should be 2*nr_logistic_mix + nr_logistic mix for each
    # decorelated channel
    info['num_channels'] = 2
    info['num_output_mixtures']= (2*args.nr_logistic_mix+args.nr_logistic_mix)*info['num_channels']
    nmix = int(info['num_output_mixtures']/2)
    info['nmix'] = nmix
    vqvae_model = VQVAErl(num_clusters=info['NUM_K'],
                        encoder_output_size=info['NUM_Z'],
                        num_output_mixtures=info['num_output_mixtures'],
                        in_channels_size=info['NUMBER_CONDITION'],
                        n_actions=info['num_actions'],
                        int_reward=info['num_rewards'],
                        ).to(DEVICE)

    print('using args', args)
    parameters = list(vqvae_model.parameters())
    opt = optim.Adam(parameters, lr=info['VQ_LEARNING_RATE'])
    if args.model_loadpath != '':
        print("loading weights from:%s" %args.model_loadpath)
        vqvae_model.load_state_dict(model_dict['vqvae_state_dict'])
        opt.load_state_dict(model_dict['vq_optimizer'])
        vqvae_model.embedding = model_dict['vq_embedding']

    #args.pred_output_size = 1*80*80
    ## 10 is result of structure of network
    #args.z_input_size = 10*10*args.num_z
    #train_cnt = train_vqvae(train_cnt, vqvae_model, opt, info, train_data_loader, valid_data_loader)
    run(info, vqvae_model, opt, train_buffer, valid_buffer, num_samples_to_train=args.num_samples_to_train, save_every_samples=args.save_every)

if __name__ == '__main__':
    from argparse import ArgumentParser

    debug = 0
    parser = ArgumentParser(description='train vq')
    #parser.add_argument('--train_data_file', default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/training_set_small.npz')
    parser.add_argument('--train_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MBFreeway_init_train/MBFreeway_data_0000040001q_train_buffer.npz')
    parser.add_argument('--valid_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MBFreeway_init_valid/MBFreeway_data_0000005001q_train_buffer.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    #parser.add_argument('--savename', default='vqdiffactintreward')
    parser.add_argument('--savename', default='FreewayVQ')
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    if not debug:
        parser.add_argument('-se', '--save_every', default=50000*5, type=int)
    else:
        parser.add_argument('-se', '--save_every', default=10, type=int)
    parser.add_argument('-b', '--beta', default=0.25, type=float, help='scale for loss 3, commitment loss in vqvae')
    parser.add_argument('-arec', '--alpha_rec', default=1, type=float, help='scale for rec loss')
    parser.add_argument('-aa', '--alpha_act', default=2, type=float, help='scale for rec loss')
    parser.add_argument('-ar', '--alpha_rew', default=1, type=float, help='scale for rec loss')
    parser.add_argument('-z', '--num_z', default=64, type=int)
    # 512 greatly outperformed 256 in freeway
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-e', '--num_samples_to_train', default=100000000, type=int)
    #parser.add_argument('-lr', '--learning_rate', default=1.5e-5) #- worked but took 0131013624 to train
    parser.add_argument('-lr', '--learning_rate', default=5e-5) #- worked but took 0131013624 to train
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    init_train()


