import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import sys
import config
from IPython import embed
from imageio import imsave, mimsave
sys.path.append('../agents')
from replay import ReplayMemory


def find_episodic_rewards(fname):
    n = np.load(fname)
    episode_reward_cnt = 0
    episodic_starts = []
    episodic_rewards = []
    # make last index an end
    terminal_flags = n['terminal_flags']
    ends = np.where(terminal_flags==True)[0]
    #start_cnt = ends[0]+1
    start_cnt = 0
    #episodic_ends = list(ends[1:-1])
    episodic_ends = list(ends)
    for cc, end_cnt in enumerate(episodic_ends):
        episode_reward_cnt = np.sum(n['rewards'][start_cnt:end_cnt+1])
        n_ends = np.sum(terminal_flags[start_cnt:end_cnt+1])
        if n_ends != 1:
            print(cc, os.path.split(fname)[1])
            print("episode terminals", n_ends)
            print("reward", np.sum(n['rewards'][start_cnt:end_cnt+1]))
        episodic_rewards.append(int(episode_reward_cnt))
        episodic_starts.append(start_cnt)
        start_cnt = end_cnt+1
    del n
    return episodic_rewards, episodic_starts, episodic_ends

def make_dataset(all_rewards, all_starts, all_ends, file_names, gamma, kind='training'):
    num = 0
    rewards = []
    terminals = []
    actions = []
    values = []
    episodic_reward = []

    for idx in range(len(all_rewards)):
        ffrom = file_names[idx]
        n = np.load(ffrom)
        fr = n['frames'][all_starts[idx]:all_ends[idx]+1]
        if not num:
            frames = fr
        else:
            frames = np.vstack((frames, fr))
        terminals.extend(n['terminal_flags'][all_starts[idx]:all_ends[idx]+1])
        actions.extend(n['actions'][all_starts[idx]:all_ends[idx]+1])
        rs = n['rewards'][all_starts[idx]:all_ends[idx]+1]
        del n
        #mimsave('I%05d_r%s.gif'%(idx,r), fr)
        rewards.extend(rs)
        vals = []
        # discounted future rewards
        for n in range(len(rs)):
            rs_n = rs[n:]
            power = np.arange(len(rs_n))
            gammas = np.ones(len(rs_n), dtype=np.float)*gamma
            gammas = gammas ** power
            cum_rs_n = np.sum(rs_n * gammas)
            vals.append(cum_rs_n)
        values.extend(vals)
        episodic_reward.append(np.sum(rs))
        print("adding sum", np.sum(rs), len(episodic_reward))
        num+=len(rs)
        #print(r,num,episodic_reward[-1])

    print(len(values), len(rewards))
    fname = os.path.join(os.path.split(ffrom)[0], '%s_set.npz'%kind)
    print('writing %s with %s examples' %(fname,len(rewards)))
    np.savez(fname, frames=frames,
                    rewards=rewards,
                    terminals=terminals,
                    values=np.array(values),
                    actions=actions,
                    episodic_reward=episodic_reward)
    return fname

if __name__ == '__main__':
    #fpath = '/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00'
    fpath = '/usr/local/data/jhansen/planning/model_savedir/DEBUGMB23/'
    npz_files = sorted(glob(os.path.join(fpath, '*train_buffer.npz')))
    assert(len(npz_files)>0)
    gamma = 0.99
    file_names = []
    all_rewards = []
    all_starts = []
    all_ends = []
    for f in npz_files:
        erewards, estarts, eends = find_episodic_rewards(f)
        fhist = f.replace('.npz', '_hist.png')
        if not os.path.exists(fhist):
            print("plotting hist", fhist)
            plt.figure()
            plt.hist(erewards, bins=range(34))
            plt.savefig(fhist)
            plt.close()

        all_rewards.extend(erewards)
        all_starts.extend(estarts)
        all_ends.extend(eends)
        file_names.extend([f for x in range(len(erewards))])

    used = make_dataset(all_rewards, all_starts, all_ends, file_names, kind='training')





