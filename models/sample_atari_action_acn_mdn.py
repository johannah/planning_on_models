import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# TODO conv
# TODO load function
# daydream function
import os
import sys
import time
import numpy as np
from torch import nn, optim
import torch
torch.set_num_threads(4)
#from torch.utils.data import Dataset, DataLoader
import config
from IPython import embed
from pixel_cnn import GatedPixelCNN
from datasets import AtariDataset
from acn_mdn import ConvVAE
from copy import deepcopy

torch.manual_seed(394)

def sample_batch(data, episode_number, episode_reward, name):
    with torch.no_grad():
        states, actions, rewards, next_states, terminals, reset, relative_indexes = data
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)

        print('generating %s images' %(states.shape[0]))
        next_states_frame = next_states[:,largs.number_condition-1:].to(DEVICE)
        if args.teacher_force:
            canvas = next_states_frame
            name+='_tf'
        else:
            canvas = 0.0*next_states_frame
        np_next_states_frame = deepcopy(next_states_frame.cpu().detach().numpy())[0,0]
        np_canvas = np.zeros_like(np_next_states_frame)
        total_reward = 0
        z, u_q = encoder_model(states)
        # next states 32,1,8484
        # actions 32
        # z 32,48
        #yhat_batch = torch.sigmoid(pcnn_decoder(next_states_frame, class_condition=actions, float_condition=z))
        for bi in range(canvas.shape[0]):
            # sample one at a time due to memory constraints
            z, u_q = encoder_model(states[bi:bi+1])
            total_reward += rewards[bi].item()
            title = 'step:%05d action:%d reward:%s %s/%s' %(bi, actions[bi].item(), int(rewards[bi]), total_reward, int(episode_reward))
            #if args.teacher_force:
            #    output = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1].detach(), class_condition=actions[bi:bi+1].detach(), float_condition=z.detach()))
            #    canvas[bi] = output[0]
            #else:
            if 1:
                print('sampling image', bi)
                for i in range(canvas.shape[1]):
                    for j in range(canvas.shape[2]):
                        for k in range(canvas.shape[3]):
                            #canvas[bi,i,j,k] = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1].detach(), class_condition=actions[bi:bi+1].detach(), float_condition=z.detach()))[0,i,j,k].detach()
                            output = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1].detach(), class_condition=actions[bi:bi+1].detach(), float_condition=z.detach()))
                            np_canvas[j,k] = output[0,0,j,k].detach().numpy()
                            if not args.teacher_force:
                                # put predicted pixel into output
                                canvas[bi,i,j,k] = output[0,0,j,k]

            f,ax = plt.subplots(1,2)
            iname = os.path.join(output_savepath, '%s_E%05d_R%03d_%05d.png'%(name, int(episode_number), int(episode_reward), bi))
            ax[0].imshow(np_next_states_frame)
            ax[0].set_title('true')
            ax[1].imshow(np_canvas)
            ax[1].set_title('est')
            plt.suptitle(title)
            plt.savefig(iname)
            print('saving', iname)
        search_path = iname[:-10:] + '*.png'
        gif_path = iname[:-10:] + '.gif'
        cmd = 'convert %s %s' %(search_path, gif_path)
        print('creating gif', gif_path)
        os.system(cmd)




if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='sample acn')
    parser.add_argument('model_loadname',  help='filename of pkl file to load models from')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-n', '--num_to_sample', default=5, type=int)
    parser.add_argument('-da', '--data_augmented', default=False, action='store_true')
    parser.add_argument('-daf', '--data_augmented_by_model', default="None")


    args = parser.parse_args()
    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    model_loadpath = os.path.abspath(args.model_loadname)
    if not os.path.exists(model_loadpath):
        print("Error: given model load path does not exist")
        print(model_loadpath)
        sys.exit()

    output_savepath = model_loadpath.replace('.pkl', '_samples')
    if not os.path.exists(output_savepath):
        os.makedirs(output_savepath)
    model_dict = torch.load(model_loadpath, map_location=lambda storage, loc: storage)
    info = model_dict['info']
    largs = info['args'][-1]


    run_num = 0
    train_data_file = largs.train_data_file
    valid_data_file = largs.train_data_file.replace('training', 'valid')

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
                                   batch_size=largs.batch_size,
                                   norm_by=255.0,)

    num_actions = valid_data_loader.n_actions
    args.size_training_set = valid_data_loader.num_examples
    hsize = valid_data_loader.data_h
    wsize = valid_data_loader.data_w

    encoder_model = ConvVAE(largs.code_length, input_size=largs.number_condition,
                            encoder_output_size=largs.encoder_output_size,
                             ).to(DEVICE)

    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=largs.num_pcnn_filters,
                                 n_layers=largs.num_pcnn_layers,
                                 n_classes=num_actions,
                                 float_condition_size=largs.code_length,
                                 last_layer_bias=0.5,
                                 hsize=hsize, wsize=wsize).to(DEVICE)

    train_cnt = 0
 #########################
    encoder_model.load_state_dict(model_dict['vae_state_dict'])
    pcnn_decoder.load_state_dict(model_dict['pcnn_state_dict'])

    encoder_model.eval()
    pcnn_decoder.eval()

    #valid_data, valid_label, test_batch_index = data_loader.validation_ordered_batch()
    valid_episode_batch, episode_index, episode_reward = valid_data_loader.get_entire_episode()
    sample_batch(valid_episode_batch, episode_index, episode_reward, 'valid')
#
#    train_data, train_label, train_batch_index = data_loader.ordered_batch()
#    sample_batch(train_data, train_label, train_batch_index, 'train')
#
#
##
