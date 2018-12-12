import gym
import sys, time
from imageio import imwrite
import os
import config
import numpy as np
from IPython import embed
from config import freeway_gt_dir, freeway_test_frames_dir, freeway_train_frames_dir
from datasets import prepare_img
bs = 500
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
train_data = np.zeros((num_train, h, w), dtype=last_o.dtype)
test_data = np.zeros((num_test, h, w),   dtype=last_o.dtype)

while cnt<(num_train+num_test):
    o, r, done, info = env.step(0)
    if done:
        last_o = env.reset()
    max_o = np.maximum(last_o, o)
    chicken,out = prepare_img(max_o)
    if cnt < num_train:
        train_data[cnt] = out
    else:
        test_data[test_cnt] = out
        test_cnt+=1
    last_o = o
    cnt +=1

np.savez('freeway_train_%05d.npz'%num_train, train_data)
np.savez('freeway_test_%05d.npz'%num_test, test_data)
