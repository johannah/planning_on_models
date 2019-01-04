import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
from torch.autograd import Variable # storing data while learning
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    plt.savefig(name)
    plt.close()

def plot_losses(train_cnts, train_losses, test_cnts, test_losses, name='loss_example.png', rolling_length=4):
    f,ax=plt.subplots(1,1,figsize=(3,3))
    ax.plot(rolling_average(train_cnts, rolling_length), rolling_average(train_losses, rolling_length), label='train loss', lw=1, c='orangered')
    ax.plot(rolling_average(test_cnts, rolling_length),  rolling_average(test_losses, rolling_length), label='test loss', lw=1, c='cornflowerblue')
    ax.scatter(rolling_average(test_cnts, rolling_length), rolling_average(test_losses, rolling_length), s=4, c='cornflowerblue')
    ax.scatter(rolling_average(train_cnts, rolling_length),rolling_average(train_losses, rolling_length), s=4, c='orangered')
    ax.legend()
    plt.savefig(name)
    plt.close()

def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    strokes += [points[b: e + 1, :2].copy()]
    return strokes

def get_dummy_data(v_x, v_y):
    for i in range(v_x.shape[1]):
        v_x[:,i] = v_x[:,0]
        v_y[:,i] = v_y[:,0]
    return v_x, v_y

def plot_strokes_vo(strokes_x_in, strokes_y_in, strokes_vo_in, lead_in=0, name='example.png',pen=True):
    f, ax1 = plt.subplots(1,1, figsize=(6,6))
    strokes_x = deepcopy(strokes_x_in)
    for i in range(strokes_x.shape[1]):
        strokes_xi = np.cumsum(deepcopy(strokes_x[:,i]), axis=0)
        if not i:
            ax1.plot(strokes_xi[:,0], strokes_xi[:,1], c='b', label='%s pred paths'%strokes_x.shape[1], linewidth=.5, alpha=0.5)
        else:
            ax1.plot(strokes_xi[:,0], strokes_xi[:,1], c='b', linewidth=.5, alpha=.5)
        ax1.scatter(strokes_xi[:,0], strokes_xi[:,1], c='b', s=.2, alpha=.5)
    strokes_y = np.cumsum(strokes_y_in, axis=0)
    sc=ax1.scatter(strokes_y[:,0,0], strokes_y[:,0,1], c=np.arange(strokes_y.shape[0]),  s=10)
    #ax1.scatter(strokes_y[:,0,0], strokes_y[:,0,1], c='g', s=.9)
    #ax1.plot(strokes_y[:,0,0], strokes_y[:,0,1], c='g',label='gt', linewidth=2, alpha=.9)
    if lead_in:
        ax1.scatter([strokes_y[lead_in,0,0]], [strokes_y[lead_in,0,1]], c='r', marker='o', s=10, label='lead in')
    strokes_vo = np.cumsum(deepcopy(strokes_vo_in), axis=0)
    ax1.plot(strokes_vo[:,0,0], strokes_vo[:,0,1], c='orangered', label='vo', linewidth=.9, alpha=0.5)
    ax1.scatter([[strokes_y[0,0,0]]], [[strokes_y[0,0,1]]], c='k', marker='o', s=10, edgecolor='k', label='start')
    ax1.legend()
    plt.colorbar(sc)
    print('plotting %s'%name)
    plt.savefig(name)
    plt.close()

def plot_strokes(strokes_x_in, strokes_y_in, lead_in=0, name='example.png',pen=True):
    strokes_x = deepcopy(strokes_x_in)
    f, ax1 = plt.subplots(1,1, figsize=(6,3))
    gt = 'lightseageen'

    if pen: # pen up pen down is third channel
        strokes_x[:, :2] = np.cumsum(strokes_x[:, :2], axis=0)
        ax1.scatter(strokes_x[:,0], -strokes_x[:,1], c='b', s=2, label='pred path')
        for stroke in split_strokes(strokes_x):
            ax1.plot(stroke[:,0], -stroke[:,1], c='b', linewidth=1)

        if np.abs(strokes_y_in).sum()>0:
            strokes_y = deepcopy(strokes_y_in)
            strokes_y[:, :2] = np.cumsum(strokes_y[:, :2], axis=0)
            ax1.scatter(strokes_y[:,0], -strokes_y[:,1], c=gt, s=2, label='true path')
            for stroke in split_strokes(strokes_y):
                ax1.plot(stroke[:,0], -stroke[:,1], c=gt, linewidth=1)
    else:
        # no pen indicator
        pc = 'cornflowerblue'
        gt = 'lightsalmon'
        for i in range(strokes_x.shape[1]):
            strokes_xi = np.cumsum(deepcopy(strokes_x[:,i]), axis=0)
            if not i:
                ax1.plot(strokes_xi[:,0], -strokes_xi[:,1], c=pc, label='%s pred paths'%strokes_x.shape[1], linewidth=.5, alpha=0.5)
            else:
                ax1.plot(strokes_xi[:,0], -strokes_xi[:,1], c=pc, linewidth=.5, alpha=.5)
            ax1.scatter(strokes_xi[:,0], -strokes_xi[:,1], c=pc, s=.2, alpha=.5)
        if np.abs(strokes_y_in).sum()>0:
            strokes_y = deepcopy(strokes_y_in)
            strokes_y = np.cumsum(strokes_y, axis=0)
            ax1.scatter(strokes_y[:,0,0], -strokes_y[:,0,1], c=gt, s=.9)
            ax1.plot(strokes_y[:,0,0], -strokes_y[:,0,1], c=gt, label='true path', linewidth=2, alpha=.9)
        if lead_in:
            ax1.scatter([strokes_y[lead_in,0,0]], [-strokes_y[lead_in,0,1]], c='r', marker='o', s=15, label='lead in')

    plt.legend()
    print('plotting %s'%name)
    plt.savefig(name)
    plt.close()

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    torch.save(state, filename)
    print("finishing save of {}".format(filename))

def load_xyr_agent(filepath):
    records = pickle.load(open(filepath+'_summary.pkl'))
    arr = np.load(filepath+'_data.npz')


class DataLoader():
    def __init__(self, load_function, train_load_path, test_load_path, batch_size=32, random_number=394):
        self.rdn = np.random.RandomState(random_number)
        self.batch_size = batch_size
        self.x,self.y,self.x_subgoals,self.x_keys,self.x_pts,self.x_state, self.state_keys = load_function(train_load_path)
        self.num_batches = self.x.shape[1]//self.batch_size
        self.batch_array = np.arange(self.x.shape[1])
        self.valid_x,self.valid_y,self.v_x_subgoals,self.v_x_keys,self.v_x_pts, self.v_state, _ = load_function(test_load_path)

    def validation_data(self):
        max_idx = min(self.batch_size, self.valid_x.shape[1])
        return self.valid_x[:,:max_idx], self.valid_y[:,:max_idx]

    def next_batch(self):
        batch_choice = self.rdn.choice(self.batch_array, self.batch_size,replace=False)
        return self.x[:,batch_choice], self.y[:,batch_choice]

def plot_traces(trues_e, tf_predicts_e, predicts_e, filename):
    ugty = np.cumsum(trues_e[:,0])
    ugtx = np.cumsum(trues_e[:,1])
    tfy = np.cumsum(tf_predicts_e[:,0])
    tfx = np.cumsum(tf_predicts_e[:,1])
    py = np.cumsum(predicts_e[:,0])
    px = np.cumsum(predicts_e[:,1])

    xmin = np.min([px.min(), tfx.min(), ugtx.min()])-10
    xmax = np.max([px.max(), tfx.max(), ugtx.max()])+10
    ymin = np.min([py.min(), tfy.min(), ugty.min()])-10
    ymax = np.max([py.max(), tfy.max(), ugty.max()])+10
    f,ax=plt.subplots(1,3, figsize=(9,3))
    ## original coordinates
    ax[0].scatter(ugtx,ugty, c=np.arange(ugty.shape[0]))
    ax[0].set_xlim([xmin,xmax])
    ax[0].set_ylim([ymin,ymax])
    ax[0].set_title("target")
    ax[1].scatter(tfx,tfy, c=np.arange(tfy.shape[0]))
    ax[1].set_xlim([xmin,xmax])
    ax[1].set_ylim([ymin,ymax])
    ax[1].set_title('teacher force predict')
    ax[2].scatter(px, py, c=np.arange(py.shape[0]))
    ax[2].set_xlim([xmin,xmax])
    ax[2].set_ylim([ymin,ymax])
    ax[2].set_title('predict')
    plt.savefig(filename)
    plt.close()




