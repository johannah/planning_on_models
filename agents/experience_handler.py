import numpy as np
from IPython import embed

def experience_replay(batch_size, max_size, history_size=4, random_seed=4455):
    """
    indexes start at zero - end at len()-history_size
    """
    random_state = np.random.RandomState(random_seed)
    memory = []
    states = []
    while True:
        inds = np.arange(len(memory))
        # get experiences out - now need to update states
        if len(memory)-(history_size+1) <= batch_size:
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
            yield_val = [[np.array(states[i:i+history_size])]+[np.array(states[i+1:i+1+history_size])]+memory[i+history_size] for i in batch_indexes]
        experience = yield yield_val
        #experience = yield [memory[i:i+history_size] for i in random_state.choice(inds[:-history_size], size=batch_size, replace=True)] if batch_size <= len(memory) else None
        if experience is not None:
            # add experience
            states.append(experience[0])
            memory.append(experience[1:])
            if len(memory)>max_size:
                memory.pop(0)
                states.pop(0)
