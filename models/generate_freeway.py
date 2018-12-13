import gym
import sys, time
from imageio import imwrite
import os
import config
import numpy as np
from IPython import embed
from config import freeway_gt_dir, freeway_test_frames_dir, freeway_train_frames_dir
from datasets import prepare_img
import imageio
# it seems like with only 500 examples at 4-frame-skip,
# the model doesn't generalize well -test performs poorly
bs = 1000
num_train = int(bs)
num_test = int(.3*bs)
rdn = np.random.RandomState(555)

for dd in [config.freeway_test_frames_dir, config.freeway_train_frames_dir]:
    if not os.path.exists(dd):
        os.makedirs(dd)

env = gym.make('FreewayNoFrameskip-v4')
last_o = env.reset()
cnt =  0
test_cnt = 0
h, w = 80,80
frame_skip = 4


def take_null_steps(env, null_steps, last_o):
    for c in range(null_steps):
        o, r, done, info = env.step(0)
        if done:
            last_o = env.reset()
        last_o = o
    return env, last_o

def get_data(env, num_frames, last_o):
    idx = 0
    data_array = np.zeros((num_frames, h, w), dtype=last_o.dtype)
    for idx in range(num_frames):
        env, last_o = take_null_steps(env, frame_skip, last_o)
        o, r, done, info = env.step(0)
        max_o = np.maximum(last_o, o)
        chicken,out = prepare_img(max_o)
        data_array[idx] = out
        if done:
            last_o = env.reset()
        else:
            last_o = o
    return env, data_array

env, last_o = take_null_steps(env, 2, last_o)
env,train_data =  get_data(env, num_train, last_o)
last_o=env.reset()

env, last_o = take_null_steps(env, 14, last_o)
env,test_data =  get_data(env, num_test, last_o)
train_name = os.path.join(config.base_datadir,'freeway_train_%05d.npz'%num_train)
test_name =  os.path.join(config.base_datadir,'freeway_test_%05d.npz'%num_test)
np.savez(train_name, train_data)
np.savez(test_name, test_data)

imageio.mimsave(train_name.replace('.npz', '.gif'), train_data[:100])
imageio.mimsave(test_name.replace('.npz', '.gif'), test_data)
