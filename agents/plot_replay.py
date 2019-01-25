import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from IPython import embed
from imageio import mimsave


def plot_replay_buffer(data):
    states = data['states']
    rewards = data['rewards']
    actions = data['actions']
    ongoing = data['ongoing_flags']
    #mimsave('example.gif',states)
    epoch_cnt = 0
    r = 0
    width = states.shape[1]
    for i in range(states.shape[0]):
        if i > 2000:
            f,ax = plt.subplots(1)
            ax.imshow(states[i], cmap='gray',interpolation="None")
            r += rewards[i]
            ax.set_title("%05d E%03d R%02d A%d"%(i,epoch_cnt,r,actions[i]))
            ax.plot(rewards[i:i+width], label='rewards', c='orange')
            ax.plot(actions[i:i+width], label='actions', c='red')
            #ax.plot(ongoing[i:i+width], label='end')
            ax.legend(loc='center left')
            plt.savefig(os.path.join(bdir, "S%05d.png"%i))
            plt.close()
        if ongoing[i]:
            epoch_cnt +=1
            r = 0
            print('new epoch', epoch_cnt)
    os.system("convert %s %s"%(os.path.join(bdir,"S*.png"), os.path.join(bdir,"o.gif")))


def plot_rae(data):
    rewards = data['rewards']
    actions = data['actions']
    ongoing = data['ongoing_flags']
    plt.figure()
    plt.plot(rewards, label='rewards')
    plt.plot(actions, label='actions')
    plt.plot(ongoing, label='end')
    plt.legend()
    plt.savefig(os.path.join(bdir, "rae.png"))
    plt.close()

if __name__ == '__main__':
    f = sys.argv[1]#'buffer_0000001001.npz'
    bdir = f.replace(".npz","_states")
    if not os.path.exists(bdir):
        os.makedirs(bdir)

    data = np.load(f)
    plot_rae(data)
    plot_replay_buffer(data)
    #from argparse import ArgumentParser
    #parser = ArgmentParser()
