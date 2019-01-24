import numpy as np
from IPython import embed

def experience_replay(batch_size, max_size, history_size=4, write_buffer_every=10000, random_seed=4455):
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
    while True:
        inds = np.arange(len(rewards))
        # get experiences out - now need to update states
        if len(rewards)-(history_size+1) <= batch_size:
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
            if len(rewards)>max_size:
                states.pop(0)
                rewards.pop(0)
                actions.pop(0)
                ongoing_flags.pop(0)
                masks.pop(0)
            if (cnt-last_write)>write_buffer_every:
                last_write = cnt
                print("saving new experience buffer")
                np.savez('buffer_%010d.npz'%cnt, states=np.array(states),
                                                 actions=np.array(actions),
                                                 rewards=np.array(rewards),
                                                 ongoing_flags=np.array(ongoing_flags),
                                                 mask=np.array(masks))

