import numpy as np
import os, sys
from models import config
from IPython import embed

def experience_replay(batch_size, max_size, history_size=4,
                      write_buffer_every=10000, random_seed=4455, name='buffer'):
    """
    indexes start at zero - end at len()-history_size
    """
    last_write = 0
    cnt = 0
    random_state = np.random.RandomState(random_seed)
    rewards = []
    states = []
    ongoing_flags = []
    masks = []
    actions = []
    heads = []
    acts = []
    while True:
        inds = np.arange(len(rewards))
        # get experiences out - now need to update states
        if (len(rewards)-(history_size+1)) <= batch_size*5:
            yield_val = None
        else:
            # get observations from each
            batch_indexes = random_state.choice(inds[:-(history_size+1)], size=batch_size, replace=False)
            # index refers to the first observation required to understand an
            # experience - for instance, when index=10, return  will include
            # observation indexes [10,11,12,13] and
            # next observation indexes [11,12,13,14] and
            # experience = (S, S_prime, action, reward, ongoing, exp_mask)
            # where experience[1:] is taken from index 13 from memory
            #yield_val = [[np.array(states[i:i+history_size])]+[np.array(states[i+1:i+1+history_size])]+memory[i+history_size] for i in batch_indexes]
            yield_val = [[np.array(states[i:i+history_size])]+
                         [np.array(states[i+1:i+1+history_size])]+
                         [actions[i+history_size]]+
                         [rewards[i+history_size]]+
                         [ongoing_flags[i+history_size]]+
                         [masks[i+history_size]] for i in batch_indexes]
        experience = yield yield_val
        #experience = yield [memory[i:i+history_size] for i in random_state.choice(inds[:-history_size], size=batch_size, replace=True)] if batch_size <= len(memory) else None
        if experience is not None:
            # add experience
            cnt+=1
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            ongoing_flags.append(experience[3])
            masks.append(experience[4])
            heads.append(experience[5])
            acts.append(experience[6])
            if rewards[-1] > 0:
                print('------------------------------------')
                print('adding positive reward',experience[1:])
            if len(rewards)>max_size:
                states.pop(0)
                rewards.pop(0)
                actions.pop(0)
                ongoing_flags.pop(0)
                masks.pop(0)
                heads.pop(0)
                acts.pop(0)
            if not cnt%100:
                print(name,'buffer cnt %s' %cnt,'last write was %s ago' %(cnt-last_write),'will write in %s' %((last_write+write_buffer_every)-cnt))
            if (cnt-last_write)>=write_buffer_every:
                last_write = cnt
                basename = '%s_%010d.npz'%(name,cnt)
                bname = os.path.join(config.model_savedir,basename)
                print("saving new experience buffer:%s"%bname)
                try:
                    np.savez_compressed(bname, states=np.array(states),
                                                 actions=np.array(actions),
                                                 rewards=np.array(rewards),
                                                 ongoing_flags=np.array(ongoing_flags),
                                                 mask=np.array(masks), heads=np.array(heads),
                                                 cnt=cnt, acts=acts)
                except:
                    print('bad save experience')
                    embed()
                # write file so we know there is a new file here
                a=open(bname.replace('.npz', '_new.txt'), 'w')
                a.write(str(cnt))
                a.close()

# eval 137000 - eval plus 1

