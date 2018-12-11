# take in current pose of agent and predict next state pose
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
from mdn_lstm import mdnLSTM
from utils import save_checkpoint, plot_losses, plot_strokes, get_dummy_data, DataLoader, load_sim_data
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)

def get_basepath_data(basepath, rerun=False):
    data_path, summary_path = get_data_paths(basepath)
    records = pickle.load(open(summary_path, 'r'))
    arr = np.load(data_path)
    img_path = basepath+'_imgs'
    for i in [os.path.join(img_path, obs_dirname),
              os.path.join(img_path, pred_dirname),
              os.path.join(img_path, pobs_dirname)]:
        if rerun:
            if os.path.exists(i):
                print("removing directory: %s" %i)
                shutil.rmtree(i)
        if not os.path.exists(i):
            os.makedirs(i)
    return records, arr, img_path

def newest(search_path):
    files = glob(search_path)
    newest_file = max(files, key=os.path.getctime)
    newest_time = os.path.getctime(newest_file)
    print('Using file: %s created at %s' %(newest_file, newest_time))
    return newest_file


class mdnAgent():
    def __init__(self, model_savedir, model_base_save_name='pose_model', DEVICE='cpu', batch_size=32, seq_length=15,
                 data_input_size=10, hidden_size=1024, number_mixtures=20,
                 grad_clip=5, save_every=1000, learning_rate=1e-4)

        super(mdnAgent, self).__init__()
        self.train_losses, self.test_losses, self.train_cnts, self.test_cnts = [], [], [], []
        if not os.path.exists(self.model_savedir):
            os.makedirs(self.model_savedir)

        self.last_save = 0
        self.cnt = 0
        self.lstm = mdnLSTM(input_size=self.data_input_size,
                            hidden_size=self.hidden_size,
                            number_mixtures=self.number_mixtures).to(self.DEVICE)
        self.optim = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

    def load_data_from_file(self):
        data_loader = DataLoader(load_sim_data, train_load_path='../data/train_2d_controller.npz',
                                 test_load_path='../data/train_2d_controller.npz',
                                 batch_size=batch_size)

        v_xnp, v_ynp = data_loader.validation_data()
        v_x = Variable(torch.FloatTensor(v_xnp))
        v_y = Variable(torch.FloatTensor(v_ynp))
        input_size = v_x.shape[2]
        output_shape = v_y.shape[2]

    def load_model(self, model_loadname):
        if not os.path.exists(model_loadname):
            print("load model: %s does not exist"%model_loadname)
            return False
        else:
            print("loading %s" %args.model_loadname)
            self.lstm_dict = torch.load(args.model_loadname)
            self.lstm.load_state_dict(lstm_dict['state_dict'])
            self.optim.load_state_dict(lstm_dict['optimizer'])
            #self.train_cnts = lstm_dict['train_cnts']
            #self.train_losses = lstm_dict['train_losses']
            #self.test_cnts = lstm_dict['test_cnts']
            #self.test_losses = lstm_dict['test_losses']
            # resume cnt from last save
            self.last_save = train_cnts[-1]
            self.cnt = train_cnts[-1]
            return True

    def train_loop(self, num_batches=10):
        ecnt = 0
        batch_loss = []
        for b in range(data_loader.num_batches):
            xnp, ynp = self.data_loader.next_batch()
            x = Variable(torch.FloatTensor(xnp))
            y = Variable(torch.FloatTensor(ynp))
            y_pred, loss = train(x,y,validation=False)
            train_cnts.append(cnt)
            train_losses.append(loss)
            if cnt%100:
                valy_pred, val_mean_loss = train(v_x,v_y,validation=True)
                test_losses.append(val_mean_loss)
                test_cnts.append(cnt)
            if cnt-last_save >= save_every:
                last_save = cnt
                # find test loss
                print('epoch: {} saving after example {} train loss {} test loss {}'.format(e,cnt,loss,val_mean_loss))
                state = {
                        'train_cnts':train_cnts,
                        'train_losses':train_losses,
                        'test_cnts':  test_cnts,
                        'test_losses':test_losses,
                        'state_dict':lstm.state_dict(),
                        'optimizer':optim.state_dict(),
                         }
                basename = os.path.join(savedir, '%s_%015d'%(model_save_name,cnt))
                n = 500
                plot_losses(rolling_average(train_cnts, n),
                            rolling_average(train_losses, n),
                            rolling_average(test_cnts, n),
                            rolling_average(test_losses, n), name=basename+'_loss.png')
                save_checkpoint(state, filename=basename+'.pkl')

            cnt+= x.shape[1]
            ecnt+= x.shape[1]


        loop(data_loader, save_every=save_every, num_epochs=args.num_epochs,
         train_losses=train_losses, test_losses=test_losses,
         train_cnts=train_cnts, test_cnts=test_cnts, dummy=args.dummy)

    embed()


def train(x, y, r, validation=False):
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    r = r.to(DEVICE)
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        xin = x[i]
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(xin, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    #y_pred_flat = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
    y1_flat = y[:,:,0]
    y2_flat = y[:,:,1]
    y3_flat = y[:,:,2]
    y1_flat = y1_flat.reshape(y1_flat.shape[0]*y1_flat.shape[1])[:,None]
    y2_flat = y2_flat.reshape(y2_flat.shape[0]*y2_flat.shape[1])[:,None]
    y3_flat = y3_flat.reshape(y3_flat.shape[0]*y3_flat.shape[1])[:,None]
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos = lstm.get_mixture_coef(y_pred_flat)
    # TODO not sure
    #out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr = lstm.get_mixture_coef(y_pred_flat)
    loss = lstm.get_lossfunc(out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos, y1_flat, y2_flat, y3_flat)
    #loss = lstm.get_lossfunc(out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, y1_flat, y2_flat)
    if not validation:
        loss.backward()
        for p in lstm.parameters():
            p.grad.data.clamp_(min=-grad_clip,max=grad_clip)
        optim.step()
    rloss = loss.cpu().data.numpy()
    return y_pred, rloss

def rolling_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', default='None', type=str, help='''filename of base
                        project to load ie d2018-11-29-T15-53-18_SI000''')
    parser.add_argument('-r', '--base_dir', default=config.results_savedir, type=str, help='''base dir to
                        load project from. Default is %s''' %config.results_savedir)
    parser.add_argument('--run_all', default=False, type=bool, help='''flag to run all project files from the basedir''')
    parser.add_argument('--rerun', default=False, action='store_true', help='''flag to rewrite images even if they are there''')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--dummy',action='store_true', default=False)
    parser.add_argument('-po', '--plot',action='store_true', default=False)
    parser.add_argument('-m', '--model_loadname',default=default_model_loadname)
    parser.add_argument('-ne', '--num_epochs',default=300, help='num epochs to train')
    parser.add_argument('-lr', '--learning_rate',default=1e-4,type=float, help='learning_rate')
    parser.add_argument('-se', '--save_every',default=20000,type=int,help='how often in epochs to save training model')
    parser.add_argument('--limit', default=-1, type=int, help='limit training data to reduce convergence time')
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load model to continue training or to generate. model path is specified with -m')
    parser.add_argument('-v', '--validate', action='store_true', default=False, help='test results')


    args = parser.parse_args()

    if args.run_all:
        # search for most recent file in dir
        npz_list = glob(args.base_dir+'*.npz')
        project_paths = []
        for n in npz_list:
            project_path = n.replace('_data.npz', '')
            project_paths.append(project_path)
    else:
        if args.project_name == 'None':
            # search in base dir for most recent
            # get newest path
            newest_path = newest(os.path.join(args.base_dir,'*.npz'))
            project_paths = [newest_path.replace('_data.npz', '')]
        else:
            project_paths = [os.path.join(args.base_dir, args.project_name)]


