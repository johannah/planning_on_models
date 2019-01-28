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
    n = states.shape[0]
    # number of pixels in plot
    width = states.shape[1]
    if back > n:
        skip = 0
    else:
        skip = n-back

    epoch_cnt = np.sum(data['ongoing_flags'][:skip])
    r = 0
    for i in range(skip,n):
        f,ax = plt.subplots(1)
        ax.imshow(states[i], cmap='gray',interpolation="None")
        r += rewards[i]
        ax.set_title("%6d E%05d K%02d R%03d A%d"%(i,epoch_cnt,data['heads'][i],r,actions[i]))
        ax.plot(np.cumsum(rewards[i:i+width]), label='rewards', c='orange')
        ax.plot(actions[i:i+width], label='actions', c='green')
        #ax.plot(ongoing[i:i+width], label='end')
        ax.legend(loc='center left')
        plt.savefig(os.path.join(bdir, "ZE%05d_%05d.png"%(epoch_cnt,i)))
        plt.close()
        if ongoing[i]:
            epoch_cnt +=1
            print('new epoch', epoch_cnt, r)
            r = 0
    os.system("convert %s %s"%(os.path.join(bdir,"ZE*.png"), os.path.join(bdir,"o.gif")))

def plot_cum_reward(data):
    rewards = data['rewards']
    plt.figure()
    plt.plot(np.cumsum(rewards), label='rewards')
    plt.legend()
    plt.savefig(os.path.join(bdir, "cumr.png"))
    plt.close()

def plot_rae(data):
    embed()
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

def plot_head_actions(data):
    acts = data['acts']
    n_heads = acts.shape[1]
    n = acts.shape[0]
    if back > n:
        r = 0
    else:
        r = n-back
    frames = np.arange(r,n)
    #ymin = acts.min()-1
    #ymax = acts.max()+1
    chosen = data['actions']
    heads = data['heads']


    plt.figure()
    plt.plot(frames,data['mask'][r:],label='mask')
    plt.title('mask')
    plt.legend()
    plt.savefig(os.path.join(bdir, "mask.png"))
    plt.close()

    plt.figure()
    plt.plot(frames,heads[r:],label='head used')
    plt.title('head')
    plt.legend()
    plt.savefig(os.path.join(bdir, "heads.png"))
    plt.close()

    for i in range(n_heads):
        plt.figure()
        plt.plot(frames,chosen[r:],label='actual')
        plt.plot(frames,heads[r:],label='head used')
        plt.plot(frames,acts[r:,i],label='H%s action'%i,linewidth=2)
        #plt.ylim(ymin,ymax)
        plt.title('head %s action' %i)
        plt.legend()
        plt.savefig(os.path.join(bdir, "head%02d_acts.png"%i))
        plt.close()

if __name__ == '__main__':
    back = 500
    f = sys.argv[1]#'buffer_0000001001.npz'
    bdir = f.replace(".npz","_states")
    if not os.path.exists(bdir):
        os.makedirs(bdir)

    data = np.load(f)
    plot_head_actions(data)
    plot_cum_reward(data)
    plot_rae(data)
    plot_replay_buffer(data)
    #from argparse import ArgumentParser
    #parser = ArgmentParser()
