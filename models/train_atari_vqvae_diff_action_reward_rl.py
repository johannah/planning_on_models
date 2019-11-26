import os
import time
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from torch import optim
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_
from ae_utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from IPython import embed
from lstm_utils import plot_dict_losses
from ae_utils import save_checkpoint
from vqvae import VQVAE
from datasets import AtariDataset
torch.manual_seed(394)

def handle_plot_ckpt(train_cnt, info, avg_train_losses, avg_valid_losses, do_plot=True):
    info['vq_train_losses_list'].append(avg_train_losses)
    info['vq_train_cnts'].append(train_cnt)
    info['vq_valid_losses_list'].append(avg_valid_losses)
    info['vq_valid_cnts'].append(train_cnt)
    print('examples %010d loss' %train_cnt, info['vq_train_losses_list'][-1])
    # plot
    if do_plot:
        info['vq_last_plot'] = train_cnt
        rolling = 3
        if len(info['vq_train_losses_list'])<rolling*3:
            rolling = 0
        train_losses = np.array(info['vq_train_losses_list'])
        valid_losses = np.array(info['vq_valid_losses_list'])
        for i in range(valid_losses.shape[1]):
            plot_name = info['vq_model_base_filepath'] + "_%010d_loss%s.png"%(train_cnt, i)
            print("plotting", os.path.split(plot_name)[1])
            plot_dict = {
                         'valid loss %s'%i:{'index':info['vq_valid_cnts'],
                                            'val':valid_losses[:,i]},
                         'train loss %s'%i:{'index':info['vq_train_cnts'],
                                            'val':train_losses[:,i]},
                        }
            plot_dict_losses(plot_dict, name=plot_name, rolling_length=rolling)
        tot_plot_name = info['vq_model_base_filepath'] + "_%010d_loss.png"%train_cnt
        tot_plot_dict = {
                         'valid loss':{'index':info['vq_valid_cnts'],
                                       'val':valid_losses.sum(axis=1)},
                         'train loss':{'index':info['vq_train_cnts'],
                                        'val':train_losses.sum(axis=1)},
                    }
        plot_dict_losses(tot_plot_dict, name=tot_plot_name, rolling_length=rolling)
        print("plotting", os.path.split(tot_plot_name)[1])
    return info


def reshape_input(ss):
    # reshape 84x84 because needs to be divisible by 2 for each of the 4 layers
    return ss[:,:,2:-2,2:-2]

def train_vqvae(train_cnt, vqvae_model, opt, info, train_data_loader, valid_data_loader):
    st = time.time()
    #for batch_idx, (data, label, data_index) in enumerate(train_loader):
    batches = 0
    while train_cnt < info['VQ_NUM_EXAMPLES_TO_TRAIN']:
        vqvae_model.train()
        opt.zero_grad()
        states, actions, rewards, values, pred_states, terminals, is_new_epoch, relative_indexes = train_data_loader.get_framediff_minibatch()
        # because we have 4 layers in vqvae, need to be divisible by 2, 4 times
        states = (2*reshape_input(torch.FloatTensor(states))-1).to(info['DEVICE'])
        rec = (2*reshape_input(torch.FloatTensor(pred_states)[:,0][:,None])-1).to(info['DEVICE'])
        actions = torch.LongTensor(actions).to(info['DEVICE'])
        rewards = torch.LongTensor(rewards).to(info['DEVICE'])
        # dont normalize diff
        diff = (reshape_input(torch.FloatTensor(pred_states)[:,1][:,None])).to(info['DEVICE'])
        x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(states)
        z_q_x.retain_grad()
        rec_est =  x_d[:, :info['nmix']]
        diff_est = x_d[:, info['nmix']:]
        loss_rec = info['ALPHA_REC']*discretized_mix_logistic_loss(rec_est, rec, nr_mix=info['NR_LOGISTIC_MIX'], DEVICE=info['DEVICE'])
        loss_diff = discretized_mix_logistic_loss(diff_est, diff, nr_mix=info['NR_LOGISTIC_MIX'], DEVICE=info['DEVICE'])

        loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
        loss_rewards = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())

        loss_act.backward(retain_graph=True)
        loss_rec.backward(retain_graph=True)
        loss_diff.backward(retain_graph=True)

        loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())
        vqvae_model.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)
        loss_2.backward(retain_graph=True)
        loss_3.backward()

        parameters = list(vqvae_model.parameters())
        clip_grad_value_(parameters, 10)
        opt.step()
        bs = float(x_d.shape[0])
        avg_train_losses = [loss_rewards.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
        if batches > info['VQ_MIN_BATCHES_BEFORE_SAVE']:
            if ((train_cnt-info['vq_last_save'])>=info['VQ_SAVE_EVERY']):
                info['vq_last_save'] = train_cnt
                info['vq_save_times'].append(time.time())
                avg_valid_losses = valid_vqvae(train_cnt, vqvae_model, info, valid_data_loader)
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

        train_cnt+=len(states)
        batches+=1
        if not batches%1000:
            print("finished %s epoch after %s seconds at cnt %s"%(batches, time.time()-st, train_cnt))
    return train_cnt

def valid_vqvae(train_cnt, vqvae_model, info, valid_data_loader, do_plot=True):
    vqvae_model.eval()
    states, actions, rewards, values, pred_states, terminals, is_new_epoch, relative_indexes = valid_data_loader.get_framediff_minibatch()
    states = (2*reshape_input(torch.FloatTensor(states))-1).to(info['DEVICE'])
    rec = (2*reshape_input(torch.FloatTensor(pred_states)[:,0][:,None])-1).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    rewards = torch.LongTensor(rewards).to(info['DEVICE'])
    # dont normalize diff
    diff = (reshape_input(torch.FloatTensor(pred_states)[:,1][:,None])).to(info['DEVICE'])
    x_d, z_e_x, z_q_x, latents, pred_actions, pred_rewards = vqvae_model(states)
    z_q_x.retain_grad()
    rec_est =  x_d[:, :info['nmix']]
    diff_est = x_d[:, info['nmix']:]
    loss_rec = info['ALPHA_REC']*discretized_mix_logistic_loss(rec_est, rec, info['NR_LOGISTIC_MIX'], DEVICE=info['DEVICE'])
    loss_diff = discretized_mix_logistic_loss(diff_est, diff, nr_mix=info['NR_LOGISTIC_MIX'], DEVICE=info['DEVICE'])
    loss_rewards = info['ALPHA_REW']*F.nll_loss(pred_rewards, rewards, weight=info['rewards_weight'])
    loss_act = info['ALPHA_ACT']*F.nll_loss(pred_actions, actions, weight=info['actions_weight'])
    loss_act.backward(retain_graph=True)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = info['BETA']*F.mse_loss(z_e_x, z_q_x.detach())
    bs,yc,yh,yw = x_d.shape
    yhat = sample_from_discretized_mix_logistic(rec_est, info['NR_LOGISTIC_MIX'])
    if do_plot:
        n_imgs = 8
        n = min(states.shape[0], n_imgs)
        gold = (rec.to('cpu')+1)/2.0
        bs,_,h,w = gold.shape
        # sample from discretized should be between 0 and 255
        # ^ not anymore - bt 0 and 1 or -1 and 1
        print("yhat sample", yhat[:,0].min().item(), yhat[:,0].max().item())
        yimg = ((yhat + 1.0)/2.0).to('cpu')
        print("yhat img", yhat.min().item(), yhat.max().item())
        print("gold img", gold.min().item(), gold.max().item())
        comparison = torch.cat([gold.view(bs,1,h,w)[:n],
                                yimg.view(bs,1,h,w)[:n]])
        img_name = info['vq_model_base_filepath'] + "_%010d_valid_reconstruction.png"%train_cnt
        save_image(comparison, img_name, nrow=n)
    bs = float(states.shape[0])
    loss_list = [loss_rewards.item()/bs, loss_act.item()/bs, loss_rec.item()/bs, loss_diff.item()/bs, loss_2.item()/bs, loss_3.item()/bs]
    return loss_list

def init_train():
    train_data_file = args.train_data_file
    data_dir = os.path.split(train_data_file)[0]
    #valid_data_file = train_data_file.replace('training', 'valid')
    valid_data_file = '/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/valid_set_small.npz'
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
                 'vq_train_data_file':args.train_data_file,
                 'VQ_SAVENAME':args.savename,
                 'DEVICE':DEVICE,
                 'VQ_NUM_EXAMPLES_TO_TRAIN':args.num_examples_to_train,
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
                 'VQ_MIN_BATCHES_BEFORE_SAVE':args.min_batches,
                 'REWARD_SPACE':[-1,0,1],
                 'action_space':[0,1,2],
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
    train_data_loader = AtariDataset(
                                   train_data_file,
                                   number_condition=info['NUMBER_CONDITION'],
                                   steps_ahead=1,
                                   batch_size=info['VQ_BATCH_SIZE'],
                                   norm_by=info['NORM_BY'],
                                    unique_actions=info['action_space'],
                                    unique_rewards=info['REWARD_SPACE'])
    train_data_loader.plot_dataset()
    valid_data_loader = AtariDataset(
                                   valid_data_file,
                                   number_condition=info['NUMBER_CONDITION'],
                                   steps_ahead=1,
                                   batch_size=info['VQ_BATCH_SIZE'],
                                   norm_by=info['NORM_BY'],
                                   unique_actions=info['action_space'],
                                   unique_rewards=info['REWARD_SPACE'])
    #info['num_actions'] = train_data_loader.n_actions
    info['num_actions'] = len(info['action_space'])
    info['num_rewards'] = len(info['REWARD_SPACE'])
    info['size_training_set'] = train_data_loader.num_examples
    info['hsize'] = train_data_loader.data_h
    info['wsize'] = train_data_loader.data_w

    #reward_loss_weight = torch.ones(info['num_rewards']).to(DEVICE)
    #for i, w  in enumerate(info['reward_weights']):
    #    reward_loss_weight[i] *= w
    actions_weight = 1-np.array(train_data_loader.percentages_actions)
    rewards_weight = 1-np.array(train_data_loader.percentages_rewards)
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
    vqvae_model = VQVAE(num_clusters=info['NUM_K'],
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
    train_cnt = train_vqvae(train_cnt, vqvae_model, opt, info, train_data_loader, valid_data_loader)

if __name__ == '__main__':
    from argparse import ArgumentParser

    debug = 0
    parser = ArgumentParser(description='train acn')
    parser.add_argument('--train_data_file', default='/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00/training_set_small.npz')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    #parser.add_argument('--savename', default='vqdiffactintreward')
    parser.add_argument('--savename', default='MBvqbt_reward')
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    if not debug:
        parser.add_argument('-se', '--save_every', default=50000*5, type=int)
        parser.add_argument('-mb', '--min_batches', default=100, type=int)
    else:
        parser.add_argument('-se', '--save_every', default=10, type=int)
        parser.add_argument('-mb', '--min_batches', default=2, type=int)
    parser.add_argument('-b', '--beta', default=0.25, type=float, help='scale for loss 3, commitment loss in vqvae')
    parser.add_argument('-arec', '--alpha_rec', default=1, type=float, help='scale for rec loss')
    parser.add_argument('-aa', '--alpha_act', default=2, type=float, help='scale for rec loss')
    parser.add_argument('-ar', '--alpha_rew', default=2, type=float, help='scale for rec loss')
    parser.add_argument('-z', '--num_z', default=64, type=int)
    # 512 greatly outperformed 256 in freeway
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-ncond', '--number_condition', default=4, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=1000000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1.5e-5) #- worked but took 0131013624 to train
    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    init_train()


