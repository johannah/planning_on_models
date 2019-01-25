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
    skip = rewards.shape[0]-1000
    epoch_cnt = np.sum(data['ongoing_flags'][:skip])
    r = 0
    width = states.shape[1]
    for i in range(skip,states.shape[0]):
        f,ax = plt.subplots(1)
        ax.imshow(states[i], cmap='gray',interpolation="None")
        r += rewards[i]
        ax.set_title("%6d E%05d K%02d R%03d A%d"%(i,epoch_cnt,r,data['heads'][i],actions[i]))
        ax.plot(rewards[i:i+width], label='rewards', c='orange')
        ax.plot(actions[i:i+width], label='actions', c='green')
        #ax.plot(ongoing[i:i+width], label='end')
        ax.legend(loc='center left')
        plt.savefig(os.path.join(bdir, "S%05d.png"%i))
        plt.close()
        if ongoing[i]:
            epoch_cnt +=1
            r = 0
            print('new epoch', epoch_cnt)
    os.system("convert %s %s"%(os.path.join(bdir,"S*.png"), os.path.join(bdir,"o.gif")))

def plot_cum_reward(data):
    rewards = data['rewards']
    plt.figure()
    plt.plot(np.cumsum(rewards), label='rewards')
    plt.legend()
    plt.savefig(os.path.join(bdir, "cumr.png"))
    plt.close()

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
    plot_cum_reward(data)
    plot_rae(data)
    plot_replay_buffer(data)
    #from argparse import ArgumentParser
    #parser = ArgmentParser()
