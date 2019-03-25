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
    n = np.load(f)
    episode_reward_cnt = 0
    episodic_starts = []
    episodic_rewards = []
    # start at least one end so that we don't encounter rollover
    ends = np.where(n['terminal_flags']==True)[0]
    start_cnt = ends[0]+1
    episodic_ends = list(ends[1:-1])
    for cc, end_cnt in enumerate(episodic_ends[:2]):
        episode_reward_cnt = np.sum(n['rewards'][start_cnt:end_cnt+1])
        n_ends = np.sum(n['terminal_flags'][start_cnt:end_cnt+1])
        if n_ends != 1:
            print(cc, os.path.split(fname)[1])
            print("episode terminals", n_ends)
            print("reward", np.sum(n['rewards'][start_cnt:end_cnt+1]))
        episodic_rewards.append(int(episode_reward_cnt))
        episodic_starts.append(start_cnt)
        start_cnt = end_cnt+1
    del n
    return episodic_rewards, episodic_starts, episodic_ends

def make_rewards_set(all_rewards, all_starts, all_ends, file_names, num_examples=10000, seed=12, kind='training', used=[]):
    random_state = np.random.RandomState(seed)
    num = 0
    rewards = []
    terminals = []
    actions = []
    episodic_reward = []
    reward_options = []
    while num < num_examples:
        if not  len(reward_options):
           reward_options = list(set(all_rewards))
           random_state.shuffle(reward_options)
        r = reward_options.pop()
        reward_indexes = np.where(np.array(all_rewards) == r)[0]
        idx = random_state.choice(reward_indexes)
        if idx not in used:
            used.append(idx)
            ffrom = file_names[idx]
            n = np.load(ffrom)
            print(r, 'getting', all_ends[idx]-all_starts[idx])
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
            for n in range(len(rs)):
                rs_n = rs[n:]
                power = np.arange(len(rs_n))
                gammas = np.ones(len(rs_n), dtype=np.float)*gamma
                gammas = gammas ** power
                cum_rs_n = np.sum(rs_n * gammas)
                vals.append(cum_rs_n)
            embed()
            episodic_reward.append(np.sum(rs))
            num+=len(rs)
            print(r,num,episodic_reward[-1])

    fname = os.path.join(os.path.split(ffrom)[0], '%s_set.npz'%kind)
    print("writing", fname)
    np.savez(fname, frames=frames,
                    rewards=rewards,
                    terminals=terminals,
                    actions=actions,
                    episodic_reward=episodic_reward)
    return used

if __name__ == '__main__':
    fpath = '/usr/local/data/jhansen/planning/model_savedir/FRANKbootstrap_priorfreeway00'
    npz_files = sorted(glob(os.path.join(fpath, '*train_buffer.npz')))
    gamma = 0.99
    file_names = []
    all_rewards = []
    all_starts = []
    all_ends = []
    for f in npz_files[:2]:
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

    n_training = 100000
    n_val = n_training*.1
    n_test = n_training*.05
    seed = 1903
    used = make_rewards_set(all_rewards, all_starts, all_ends, file_names, num_examples=n_test, seed=seed+15, kind='test',used=[])
    used = make_rewards_set(all_rewards, all_starts, all_ends, file_names, num_examples=n_training, seed=seed, kind='training', used=[])
    used = make_rewards_set(all_rewards, all_starts, all_ends, file_names, num_examples=n_val, seed=seed+10, kind='valid', used=[])





