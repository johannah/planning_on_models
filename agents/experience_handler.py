import numpy as np
import os, sys
from models import config
import time
from IPython import embed

def experience_replay(batch_size, max_size, history_size=4,
                      random_seed=4455,
                      name='buffer', buffer_file='',
                      is_eval=False):
    """
    indexes start at zero - end at len()-history_size
    """
    random_state = np.random.RandomState(random_seed)
    est = time.time()
    if buffer_file != '':
        data = np.load(buffer_file)
        states = list(data['states'])
        others = list(data['others'])
        masks = list(data['masks'])
        cnt = data['cnt']
        del data
        print("loaded buffer from %s" %buffer_file)
    else:
        print("init empty buffer")
        states=[]
        others = []
        masks = []
    if is_eval:
        heads=[]
        acts=[]
    while True:
        if (len(others)-(history_size+1)) < batch_size:
            yield_val = None
        else:
            st = time.time()
            # start indexes that can be used to find states
            inds = np.arange(history_size,len(others)-1)
            # get observations from each
            batch_indexes = random_state.choice(inds, size=batch_size, replace=False)
            # index refers to the true state observed by the agent
            # this index require history_size previous frames
            # it also requires one future frame
            # experience - for instance, when index=10, return  will include
            # observation S indexes [7,8,9,10] and
            # next observation s_prime indexes [8,9,10,11] and
            # experience = (S, S_prime, [action, reward, ongoing], exp_mask)

            S = np.array([states[i-history_size:i+1] for i in batch_indexes]).astype(np.float32)/256.
            _other = np.array([others[i] for i in batch_indexes])
            _masks = np.array([masks[i] for i in batch_indexes])
            yield_val = [S[:,:history_size], S[:,1:], _other, _masks]
            et = time.time()
            # on gpu/cpu - with small array - find indexes takes .02
            # on gpu/cpu - with small array - add experience takes 1.66e-6
            # on gpu with len(6000) array - add experience takes 9.53e-7
            # on gpu with len(6000) array - find indexes takes 0.836
            #print('buffer add',et-est)

        do_checkpoint, experience = yield yield_val
        if experience is not None:
            est = time.time()
            # add experience
            states.append((experience[0]*256).astype(np.uint8))
            masks.append(experience[4])
            others.append([experience[1], experience[2], not(experience[3])])
            if is_eval:
                heads.append(experience[5])
                acts.append(experience[6])
            cnt = experience[7]
            #if experience[2] > 0:
            #    print('------------------------------------')
            #    print('adding positive reward',experience[1:])
            if len(others)>max_size:
                states.pop(0)
                others.pop(0)
                masks.pop(0)
                if is_eval:
                    heads.pop(0)
                    acts.pop(0)

            if do_checkpoint is not '':
                bname = do_checkpoint.replace('.pkl', '_%s.npz'%name)
                print("saving new experience buffer:%s"%bname)
                try:
                    np.savez_compressed(bname, states=np.array(states),
                                               others=np.array(others),
                                               masks=np.array(masks),
                                               cnt=cnt)

                    if is_eval:
                        np.savez_compressed(bname, states=np.array(states),
                                                others=np.array(others),
                                                masks=np.array(masks),
                                                cnt=cnt, acts=np.array(acts),
                                                heads=np.array(heads))
                    print("finished experience buffer save")

                except Exception as e:
                    # sometimes not enough memory to save on shared lab machine
                    # (docker was using a lot)
                    print('bad save experience')
                    print(e)
                    time.sleep(5)
                    #embed()

