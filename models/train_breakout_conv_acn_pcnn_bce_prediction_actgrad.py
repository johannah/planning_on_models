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
from sklearn.neighbors import KNeighborsClassifier
import cv2
from copy import deepcopy
from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from lstm_utils import plot_losses
torch.manual_seed(394)
torch.set_num_threads(4)
from imageio import imsave
from ae_utils import save_checkpoint, handle_plot_ckpt
from train_breakout_conv_action_model import load_avg_grad_cam
from make_tsne_plot import tsne_plot
#from bad_pixel_cnn import GatedPixelCNN
from pixel_cnn import GatedPixelCNN
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

    small_path = buffer_path.replace('.npz', '_subset_%06d.npz' %max_examples)
    if os.path.exists(small_path):
        print('loading small buffer path')
        print(small_path)
        load_buffer = ReplayMemory(load_file=small_path)
    else:
        load_buffer = ReplayMemory(load_file=buffer_path)
        print('loading prescribed buffer path')
        print(buffer_path)

    # TODO if frame size is wrong - we arent handling
    #if load_buffer.count > max_examples:
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
                print('stopping after %s examples'%sbuffer.count)
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
                    if not i%100:
                        print(sbuffer.count)

        sbuffer.save_buffer(small_path)
        load_buffer = sbuffer
    print(load_buffer.count)
    assert load_buffer.count > 10
    return load_buffer, small_path

def prepare_state(st, DEVICE, NORM_BY):
    # states come in at uint8 - should be converted to float between -1 and 1
    # st.shape is bs,4,40,40
#    output = (2*torch.FloatTensor(st)/NORM_BY-1).to(DEVICE)
#    assert output.max() < 1.01
#    assert output.min() > -1.01
#    # convert to 0 and 1
    output = (torch.FloatTensor(st)/NORM_BY).to(DEVICE)
    output[output>0] = 1.0
    return output

def make_state(batch, DEVICE, NORM_BY):
    '''  batch is composed of
     states are [ts0, ts1, ts2, ts3] uint8
     actions are                  [a3] int
     next_states     [ts1, ts2, ts3, ts4] uint8
     rewards are                  [r3] int
    '''
    states, actions, rewards, next_states, terminal_flags, masks = batch
    states = prepare_state(states, DEVICE, NORM_BY)
    # only predict the unseen (most recent) state
    next_states = prepare_state(next_states[:,-1:], DEVICE, NORM_BY)
    # next state is the corresponding action
    actions = torch.LongTensor(actions).to(DEVICE)
    rewards = torch.LongTensor(rewards).to(DEVICE)
    ac = torch.ones_like(states[:,-1:])
    rc = torch.ones_like(states[:,-1:])
    # add in actions/reward as conditioning
    # states is 6 channels
    for i in range(ac.shape[0]):
        ac[i]*=actions[i]
        rc[i]*=rewards[i]
    states = torch.cat((states,ac,rc), dim=1)
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

# ConvVAE was also imported - not sure which one was used
class ConvVAE(nn.Module):
    def __init__(self, code_len, input_size=1, num_output_channels=30):
        super(ConvVAE, self).__init__()
        self.code_len = code_len
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        # found via experimentation - 3 for mnist
        # input_image == 28 -> eo=7
        # for 36x36 input shape -> eo=9, raveled is 3240
        self.eo=eo=encode_output_size = 10
        self.fc21 = nn.Linear(code_len*2*eo*eo, code_len)
        self.fc22 = nn.Linear(code_len*2*eo*eo, code_len)
        self.fc3 = nn.Linear(code_len, code_len*2*eo*eo)

        out_layer = nn.ConvTranspose2d(in_channels=16,
                        out_channels=num_output_channels,
                        kernel_size=4,
                        stride=2, padding=1)

        # set bias to 0.5 for sigmoid
        out_layer.bias.data.fill_(0.5)
        # set bias to 0 for data output bt -1 and 1
        #out_layer.bias.data.fill_(0.0)

        self.decoder = nn.Sequential(
               nn.ConvTranspose2d(in_channels=code_len*2,
                      out_channels=32,
                      kernel_size=1,
                      stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                out_layer
                     )

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        c = self.reparameterize(mu,logvar)
        return c, mu, logvar

def kl_loss_function(u_q, s_q, u_p, s_p):
    '''  coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     Args:
         u_q: mean of model posterior
         s_q: log std of model posterior
         u_p: mean of conditional prior
         s_p: log std of conditional prior

     Returns: loss
     '''
    acn_KLD = torch.sum(s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    return acn_KLD

class PriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super(PriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2_u = nn.Linear(n_hidden, self.code_length)
        self.fc2_s = nn.Linear(n_hidden, self.code_length)

        self.knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
        codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
        self.fit_knn(codes)

    def fit_knn(self, codes):
        ''' will reset the knn  given an nd array
        '''
        st = time.time()
        self.codes = codes
        assert(len(self.codes)>1)
        y = np.zeros((len(self.codes)))
        self.knn.fit(self.codes, y)

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training example as np
        '''
        neighbor_distances, neighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize)
        else:
            chosen_neighbor_index = np.zeros((bsize), dtype=np.int)
        return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]]

    def forward(self, codes):
        st = time.time()
        np_codes = codes.cpu().detach().numpy()
        previous_codes = self.batch_pick_close_neighbor(np_codes)
        previous_codes = torch.FloatTensor(previous_codes).to(DEVICE)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        logstd = self.fc2_s(h1)
        return mu, logstd

def handle_plot_ckpt(do_plot, train_cnt, avg_train_loss):
    info['train_losses'].append(avg_train_loss)
    info['train_cnts'].append(train_cnt)
    test_loss = test_acn(train_cnt,do_plot)
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
        filename = vae_base_filepath + "_%010dex.pt"%train_cnt
        state = {
                 'vae_state_dict':vae_model.state_dict(),
                 'prior_state_dict':prior_model.state_dict(),
                 'pcnn_state_dict':pcnn_decoder.state_dict(),
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

def train_acn(train_cnt):
    #test_acn(0,True)
    vae_model.train()
    prior_model.train()
    train_loss = 0
    init_cnt = train_cnt
    st = time.time()
    train_buffer.reset_unique()
    #for batch_idx, (data, _, data_index) in enumerate(train_loader):
    while train_buffer.unique_available:
        #
        batch = train_buffer.get_unique_minibatch(args.batch_size)
        batch_idx = batch[-1]
        states, actions, rewards, next_state = make_state(batch[:-1], DEVICE, 255.)
        bs = states.shape[0]
        opt.zero_grad()
        z, u_q, s_q = vae_model(states)
        # add the predicted codes to the input
        prior_model.codes[batch_idx] = u_q.detach().cpu().numpy()
        prior_model.fit_knn(prior_model.codes)
        u_p, s_p = prior_model(u_q)
        kl = kl_loss_function(u_q, s_q, u_p, s_p)
        drop_next_state = F.dropout(next_state, p=args.dropout_prob, training=True, inplace=False)
        #zero_out = torch.Tensor(random_state.randint(0,2,bs))
        #for zo in range(bs): drop_next_state[zo] *= zero_out[zo]
        yhat_batch = torch.sigmoid(pcnn_decoder(x=drop_next_state, float_condition=z))
        rec_loss = F.binary_cross_entropy(yhat_batch, next_state, reduction='none')
        #rec_loss = (rec_loss[:,0]*train_grad[batch_idx] + rec_loss*.5).sum()
        #grad_scale = train_grad[min(batch_idx-1, 0)]train_grad[batch_idx]+train_grad[min([batch_idx+1, len(train_grad)))
        grad_scale = train_grad[batch_idx]
        #rec_loss = (rec_loss[:,0]*grad_scale + rec_loss*.1).sum()
        rec_loss = (rec_loss[:,0]*grad_scale).sum()
        # input should be scaled bt -1 and 1 for dml
        #rec_loss = discretized_mix_logistic_loss(yhat_batch, next_state, nr_mix=nr_logistic_mix, DEVICE=DEVICE)
        #yhat = sample_from_discretized_mix_logistic(yhat_batch, nr_logistic_mix)
        loss = kl+rec_loss
        loss.backward()
        train_loss+= loss.item()
        opt.step()
        # add batch size because it hasn't been added to train cnt yet
        avg_train_loss = train_loss/float((train_cnt+bs)-init_cnt)
        if train_cnt > 50000:
            handle_checkpointing(train_cnt, avg_train_loss)
        train_cnt+=bs
    print("finished epoch after %s seconds at cnt %s"%(time.time()-st, train_cnt))
    return train_cnt

def test_acn(train_cnt, do_plot):
    vae_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()
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
                states, actions, rewards, next_state = make_state(batch[:-1], DEVICE, 255.)
                bs = states.shape[0]
                # yhat_batch is bt 0-1
                z, u_q, s_q = vae_model(states)
                u_p, s_p = prior_model(u_q)
                kl = kl_loss_function(u_q, s_q, u_p, s_p)
                drop_next_state = F.dropout(next_state, p=args.dropout_prob, training=True, inplace=False)
                #zero_out = torch.Tensor(random_state.randint(0,2,bs))
                #for zo in range(bs): drop_next_state[zo] *= zero_out[zo]

                yhat_batch = torch.sigmoid(pcnn_decoder(x=drop_next_state, float_condition=z))
                rec_loss = F.binary_cross_entropy(yhat_batch, next_state, reduction='none')
                #rec_loss = (rec_loss[:,0]*valid_grad[batch_idx] + rec_loss*.5).sum()
                grad_scale = valid_grad[batch_idx]
                #rec_loss = (rec_loss[:,0]*grad_scale + rec_loss*.1).sum()
                rec_loss = (rec_loss[:,0]*grad_scale).sum()
                #rec_loss = discretized_mix_logistic_loss(yhat_batch, data, nr_mix=nr_logistic_mix, DEVICE=DEVICE)
                loss = kl+rec_loss
                test_loss+= loss.item()
                seen += bs
                if i == 0:
                    if do_plot:
                         print('writing img')
                         n = min(bs, 8)
                         # sampled yhat_batch is bt 0-1
                         last_state = states[:,3:4]
                         # data should be between 0 and 1 to be plotted with
                         comparison = torch.cat([
                                                 last_state[:n],
                                                 drop_next_state[:n],
                                                 next_state[:n],
                                                 valid_grad[batch_idx][:n][:,None],
                                                 yhat_batch[:n]])
                         img_name = vae_base_filepath + "_%010d_valid_reconstruction.png"%train_cnt
                         save_image(comparison.cpu(), img_name, nrow=n)
                         print('finished writing img', img_name)

    test_loss /= seen
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('finished test', time.time()-st)
    return test_loss

def call_tsne_plot():
    vae_model.eval()
    prior_model.eval()
    pcnn_decoder.eval()
    with torch.no_grad():
        valid_buffer.reset_unique()
        batch = valid_buffer.get_unique_minibatch(args.num_tsne)
        batch_idx = batch[-1]
        states, actions, rewards, next_state = make_state(batch[:-1], DEVICE, 255.)
        # yhat_batch is bt 0-1
        z, u_q, s_q = vae_model(states)
        u_p, s_p = prior_model(u_q)
        yhat_batch = torch.sigmoid(pcnn_decoder(x=next_state, float_condition=z))
        end = args.model_loadpath.split('.')[-1]
        html_path = args.model_loadpath.replace(end, 'html')
        X = u_q.cpu().numpy()
        images = np.round(yhat_batch.cpu().numpy()[:,0], 0).astype(np.int16)
        #images = next_state[:,0].cpu().numpy()
        tsne_plot(X=X, images=images, color=batch_idx, perplexity=args.perplexity,
                  html_out_path=html_path)

def sample():
    print('starting sample', train_cnt)
    from skvideo.io import vwrite
    vae_model.eval()
    pcnn_decoder.eval()
    output_savepath = args.model_loadpath.replace('.pt', '')
    data_dict = {'train':train_buffer, 'valid':valid_buffer}
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_buffer = data_dict[phase]
            data_buffer.reset_unique()
            batch = data_buffer.get_unique_minibatch(args.batch_size)
            batch_idx = batch[-1]
            states, actions, rewards, next_state = make_state(batch[:-1], DEVICE, 255.)
            data = states
            target = next_state
            bs = data.shape[0]
            z, u_q, s_q = vae_model(data)
            # teacher forced version
            yhat_batch = torch.sigmoid(pcnn_decoder(x=target, float_condition=z))
            u_p, s_p = prior_model(u_q)
            canvas = torch.zeros_like(target)
            building_canvas = []
            for i in range(canvas.shape[1]):
                for j in range(canvas.shape[2]):
                    print('sampling row: %s'%j)
                    for k in range(canvas.shape[3]):
                        output = torch.sigmoid(pcnn_decoder(x=canvas, float_condition=z))
                        #print(output[0,i,j,k], torch.round(output[0,i,j,k]))
                        #canvas[:,i,j,k] = torch.round(output[:,i,j,k].detach())
                        canvas[:,i,j,k] = output[:,i,j,k].detach()
                    #if not k%5:
                    building_canvas.append(deepcopy(canvas[0].detach().cpu().numpy()))
            f,ax = plt.subplots(bs, 3, sharex=True, sharey=True, figsize=(2,2*bs))
            npdata = target.detach().cpu().numpy()
            npoutput = canvas.detach().cpu().numpy()
            npyhat = yhat_batch.detach().cpu().numpy()
            for idx in range(bs):
                ax[idx,0].imshow(npdata[idx,0], cmap=plt.cm.gray)
                ax[idx,0].set_title('true')
                ax[idx,1].imshow(npyhat[idx,0], cmap=plt.cm.gray)
                ax[idx,1].set_title('tf')
                ax[idx,2].imshow(npoutput[idx,0], cmap=plt.cm.gray)
                ax[idx,2].set_title('sam')
                ax[idx,0].axis('off')
                ax[idx,1].axis('off')
                ax[idx,2].axis('off')
            iname = output_savepath + '_sample_%s.png'%phase
            print('plotting %s'%iname)
            plt.savefig(iname)
            plt.close()

            building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
            print('writing building movie')
            mname = output_savepath + '_build_%s.mp4'%phase
            vwrite(mname, building_canvas)
            print('finished %s'%mname)
    sys.exit()


#def sample():
#    print('starting sample', train_cnt)
#    from skvideo.io import vwrite
#    vae_model.eval()
#    pcnn_decoder.eval()
#    basedir = args.model_loadpath.replace('.pt', '')
#    if args.use_training_set:
#        data_buffer = train_buffer
#        print('using training data')
#        name = 'tr'
#    else:
#        print('using valid data')
#        name = 'va'
#        data_buffer = valid_buffer
#
#    if not os.path.exists(basedir):
#        os.makedirs(basedir)
#    print('writing to: %s'%basedir)
#    with torch.no_grad():
#        data_buffer.reset_unique()
#        for i in range(10):
#            if data_buffer.unique_available:
#                batch = data_buffer.get_unique_minibatch(args.batch_size)
#                np_next_states = batch[3]
#                batch_idx = batch[-1]
#                states, actions, rewards, next_state = make_state(batch[:-1], DEVICE, 255.)
#                z, u_q, s_q = vae_model(states)
#                # yhat_batch is bt 0-1
#                canvas = next_state
#                output_canvas = 0.0*next_state
#                #for bi in range(canvas.shape[0]):
#                # sample one at a time due to memory constraints
#                st = time.time()
#                #if args.teacher_force:
#                #    canvas = next_state
#
#                #drop_next_state = F.dropout(next_state, p=args.dropout_prob, training=False, inplace=False)
#                #yhat_batch = torch.sigmoid(pcnn_decoder(x=drop_next_state, float_condition=z))
#                yhat_batch = torch.sigmoid(pcnn_decoder(x=next_state, float_condition=z))
#                np_yhat_batch = yhat_batch.detach().cpu().numpy()
#                if args.teacher_force:
#                    canvas = next_state
#                else:
#                    canvas = torch.zeros_like(next_state)
#                output_canvas = torch.zeros_like(next_state)
#                building_canvas = []
#                #print('sampling image', bi)
#                for i in range(canvas.shape[1]):
#                    for j in range(canvas.shape[2]):
#                        print('j', j)
#                        for k in range(canvas.shape[3]):
#                            #output = torch.sigmoid(pcnn_decoder(x=canvas[bi:bi+1] float_condition=z[bi:bi+1]))
#                            output = torch.sigmoid(pcnn_decoder(x=canvas, float_condition=z))
#                            if  args.teacher_force:
#                                output_canvas[:,i,j,k] = output[:,i,j,k].detach() #.cpu().numpy()
#                            else:
#                                #canvas[:,i,j,k] = torch.round(output[:,i,j,k].detach())
#                                output_canvas[:,i,j,k] = output[:,i,j,k].detach() #.cpu().numpy()
#                                #output_canvas = canvas
#                            building_canvas.append(output_canvas[0].detach().cpu().numpy())
#                print(yhat_batch[:,0,0,0])
#                print(output[:,0,0,0])
#                print(next_state[:,0,0,0])
#                print('-------1')
#                print(yhat_batch[:,0,0,1])
#                print(output[:,0,0,1])
#                print(next_state[:,0,0,1])
#                building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
#                print('writing building movie')
#                if args.teacher_force:
#                    name +='tf'
#                mname = os.path.join(basedir, '%010d%s.mp4'%(batch_idx[0],name))
#                vwrite(mname, building_canvas)
#                et = time.time()
#                print(et-st)
#                np_bi = output_canvas.detach().cpu().numpy()
#                print(np_bi.min(), np_bi.max())
#                for bi, true_idx in enumerate(batch_idx):
#                    iname = os.path.join(basedir, '%010d%s.png'%(true_idx,name))
#                    #np_bi = output_canvas[bi,0].detach().cpu().numpy()
#                    f,ax = plt.subplots(1,3)
#                    ax[0].imshow(np_next_states[bi,0])
#                    ax[0].set_title('true')
#                    ax[1].imshow(np_bi[bi,0])
#                    ax[1].set_title('est')
#                    ax[2].imshow(np_yhat_batch[bi,0])
#                    ax[2].set_title('est')
#                    plt.savefig(iname)
#                    print('saving', iname)
#                    plt.close()
#                embed()
#    sys.exit()
#    #    print("starting img")
#
#def save_checkpoint(state, filename='model.pt'):
#    print("starting save of model %s" %filename)
#    torch.save(state, filename)
#    print("finished save of model %s" %filename)
#

def init_train(model_load_path, train_data_path, valid_data_path):
    """ use args to setup inplace training """

    #init_train(
    #    model_load_path=args.model_loadpath
    #    train_data_path=args.train_buffer
    #valid_data_path=args.valid_buffer

    data_dir = os.path.split(train_data_path)[0]

    # we are starting from scratch training this model
    if model_loadpath == "":
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
    train_buffer = make_subset_buffer(train_data_path, max_examples=info['NUM_TRAINING_EXAMPLES'], frame_height=args.frame_height, frame_width=args.frame_width)
    valid_buffer = make_subset_buffer(valid_data_path, max_examples=int(info['NUM_TRAINING_EXAMPLES']*.1), frame_height=args.frame_height, frame_width=args.frame_width)
    valid_buffer = ReplayMemory(load_file=valid_data_path)
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
    info['num_output_mixtures']= (2*args.nr_logistic_mix+args.nr_logistic_mix)*info['HISTORY_SIZE']
    nmix = int(info['num_output_mixtures']/info['HISTORY_SIZE'])
    info['nmix'] = nmix
    #encoder_model = ConvVAE(info['CODE_LENGTH'], input_size=args.num_condition,
    #                        encoder_output_size=args.encoder_output_size,
    #                        num_output_channels=nmix,
    #                         ).to(DEVICE)
    encoder_model = ConvVAE(info['CODE_LENGTH'], input_size=args.num_condition,
                            encoder_output_size=args.encoder_output_size,
                            num_output_channels=1
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

#if __name__ == '__main__':
#    from argparse import ArgumentParser

    #parser = ArgumentParser(description='train acn')
    #parser.add_argument('--train_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005679481_train.npz')
    #parser.add_argument('--valid_buffer', default='/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval.npz')
    #parser.add_argument('-s', '--sample', action='store_true', default=False)
    #parser.add_argument('-ns', '--num_samples', default=5, type=int)
    #parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    #parser.add_argument( '--use_pcnn', action='store_true', default=False)
    #parser.add_argument('-c', '--cuda', action='store_true', default=False)
    #parser.add_argument('-d', '--debug', action='store_true', default=False)
    #parser.add_argument('--savename', default='acn')
    #parser.add_argument('-l', '--model_loadpath', default='')
    #parser.add_argument('-uniq', '--require_unique_codes', default=False, action='store_true')
    #parser.add_argument('-se', '--save_every', default=100000*2, type=int)
    #parser.add_argument('-le', '--log_every_batches', default=100, type=int)


    #parser.add_argument('-bs', '--batch_size', default=256, type=int)
    ## 4x36x36 input -> 768 output
    #parser.add_argument('-eos', '--encoder_output_size', default=768, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    #parser.add_argument('-cl', '--code_length', default=48, type=int)
    #parser.add_argument('-ncond', '--num_condition', default=4, type=int)
    #parser.add_argument('-k', '--num_k', default=5, type=int)
    #parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    #parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    #parser.add_argument('-ne', '--num_training_examples', default=100000, type=int)
    #parser.add_argument('-lr', '--learning_rate', default=1e-5)
    ##parser.add_argument('-pv', '--possible_values', default=1)
    #parser.add_argument('-npcnn', '--num_pcnn_layers', default=6)
    #parser.add_argument('-pf', '--num_pcnn_filters', default=16, type=int)
    #parser.add_argument('-nm', '--num_mixtures', default=8, type=int)
    #parser.add_argument('--alpha_act', default=2, type=float, help='scale for last action prediction')
    #parser.add_argument('--alpha_rew', default=1, type=float, help='scale for reward prediction')
    #parser.add_argument('--alpha_rec', default=2, type=float, help='scale for rec prediction')
    #args = parser.parse_args()
    #if args.cuda:
    #    DEVICE = 'cuda'
    #else:
    #    DEVICE = 'cpu'

    #if args.debug:
    #    args.save_every = 10
    #    args.plot_every = 10
    #    args.model_log_every_batches = 1

    #init_train()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('--use_training_set', action='store_true', default=False)
    parser.add_argument('-t', '--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=3)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-nt', '--num_tsne', default=300, type=int)
    parser.add_argument('-dp', '--dropout_prob', default=0.2, type=float)
    parser.add_argument('-l', '--model_loadpath', default='')
    parser.add_argument('-se', '--save_every', default=60000*10, type=int)
    parser.add_argument('-pe', '--plot_every', default=200000, type=int)
    parser.add_argument( '--log_every', default=200000, type=int)
    parser.add_argument('-me', '--max_examples', default=100000, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    #parser.add_argument('-nc', '--number_condition', default=4, type=int)
    #parser.add_argument('-sa', '--steps_ahead', default=1, type=int)
    parser.add_argument('-cl', '--code_length', default=128, type=int)
    parser.add_argument('--frame_height', default=40, type=int)
    parser.add_argument('--frame_width', default=40, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-nl', '--nr_logistic_mix', default=10, type=int)
    parser.add_argument('-e', '--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=2e-5)
    parser.add_argument('-md', '--model_savedir', default='../../model_savedir')
    parser.add_argument('--train_buffer', default='MFBreakout_train_anneal_14342_05/breakout_S014342_N0001303598_train.npz')
    parser.add_argument('--valid_buffer', default='MFBreakout_train_anneal_14342_05/breakout_S014342_N0000550155_train.npz')
    #parser.add_argument('--train_buffer', default='BreakoutNewActionNOAnnealingPRIOR00/BreakoutNewActionNOAnnealingPRIOR_0007014244q_train_buffer.npz')
    #parser.add_argument('--valid_buffer', default='BreakoutNewActionNOAnnealingPRIOR00/BreakoutNewActionNOAnnealingPRIOR_0000500549q_train_buffer.npz')


    parser.add_argument('-aml', '--action_model_loadpath', default='results_train_breakout_action/sigcacn_breakout_action_0075002880ex.pt')

    args = parser.parse_args()

    nr_logistic_mix = 10
    num_output_channels = 1
    nmix = (2*nr_logistic_mix+nr_logistic_mix)*num_output_channels

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    vae_base_filepath = os.path.join(args.model_savedir, 'sigcacn_breakout_binary_bce_pcnn_pred_allactgrad_dropout_redopcnn')
    action_model_loadpath = os.path.join(args.model_savedir, args.action_model_loadpath)

    train_data_path = os.path.join(args.model_savedir, args.train_buffer)
    valid_data_path = os.path.join(args.model_savedir, args.valid_buffer)
    train_buffer, train_small_path = make_subset_buffer(train_data_path, max_examples=args.max_examples, frame_height=args.frame_height, frame_width=args.frame_width)
    valid_buffer, valid_small_path = make_subset_buffer(valid_data_path, max_examples=int(args.max_examples*.1), frame_height=args.frame_height, frame_width=args.frame_width)

    num_actions = len(set(train_buffer.actions))
    hsize = train_buffer.frames.shape[1]
    wsize = train_buffer.frames.shape[2]
    info = {'train_cnts':[],
            'train_losses':[],
            'test_cnts':[],
            'test_losses':[],
            'save_times':[],
            'args':[args],
            'last_save':0,
            'last_plot':0,
            'NUM_PCNN_FILTERS':1,
            'NUM_PCNN_LAYERS':12,
            'num_actions':num_actions,
            'hsize':hsize,
            'wsize':wsize,
            'last_layer_bias':0.5,
            'size_training_set':train_buffer.count,
             }


    print(info['size_training_set'])
    train_cnt = 0

    if args.model_loadpath !='':
        tmlp =  args.model_loadpath+'.tmp'
        os.system('cp %s %s'%(args.model_loadpath, tmlp))
        _dict = torch.load(tmlp, map_location=lambda storage, loc:storage)
        info = _dict['info']
        if 'size_training_set' not in info.keys():
            info['size_training_set'] = train_buffer.count
        largs = info['args'][-1]
        args.code_length = largs.code_length
        args.num_k = largs.num_k

    vae_model = ConvVAE(args.code_length, input_size=6, num_output_channels=1).to(DEVICE)
    #vae_model = ConvVAE(args.code_length, input_size=nmix, num_output_channels=1).to(DEVICE)

    prior_model = PriorNetwork(size_training_set=info['size_training_set'],
                               code_length=args.code_length, k=args.num_k).to(DEVICE)
    pcnn_decoder = GatedPixelCNN(input_dim=1,
                                 dim=info['NUM_PCNN_FILTERS'],
                                 n_layers=info['NUM_PCNN_LAYERS'],
                                 n_classes=info['num_actions'],
                                 float_condition_size=args.code_length,
                                 last_layer_bias=info['last_layer_bias'],
                                 hsize=info['hsize'], wsize=info['wsize']).to(DEVICE)


    if args.model_loadpath !='':
        vae_model.load_state_dict(_dict['vae_state_dict'])
        prior_model.load_state_dict(_dict['prior_state_dict'])
        pcnn_decoder.load_state_dict(_dict['pcnn_state_dict'])

        info = _dict['info']
        train_cnt = info['train_cnts'][-1]

    vae_model.to(DEVICE)
    prior_model.to(DEVICE)
    # TODO write model loader and args.sample
    if args.test:
        test_acn(train_cnt, True)
    elif args.sample:
        sample()
    elif args.tsne:
        call_tsne_plot()
    else:
        valid_grad = load_avg_grad_cam(action_model_loadpath, valid_buffer, valid_small_path, DEVICE=DEVICE)
        train_grad = load_avg_grad_cam(action_model_loadpath, train_buffer, train_small_path, DEVICE=DEVICE)
        valid_grad = torch.Tensor(valid_grad).to(DEVICE)
        train_grad = torch.Tensor(train_grad).to(DEVICE)
        parameters = list(vae_model.parameters()) + list(prior_model.parameters()) + list(pcnn_decoder.parameters())
        opt = optim.Adam(parameters, lr=args.learning_rate)
        test_acn(train_cnt, do_plot=True)
        if args.model_loadpath !='':
            opt.load_state_dict(_dict['optimizer'])
        while train_cnt < args.num_examples_to_train:
            train_cnt = train_acn(train_cnt)

