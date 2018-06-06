import gym
import sys, time
from imageio import imwrite
import os
import config
import numpy as np
from IPython import embed
from datasets import transform_freeway
num_train = 50000
num_test = 64*4
rdn = np.random.RandomState(555)

for dd in [config.freeway_test_frames_dir, config.freeway_train_frames_dir]:
    if not os.path.exists(dd):
        os.makedirs(dd)

env = gym.make('FreewayNoFrameskip-v4')
last_o = env.reset()
cnt =  0

while cnt<(num_train+num_test):
    o, r, done, info = env.step(0)
    max_o = o # np.maximum(last_o, o)
    out = transform_freeway(max_o)
    last_o = o
    if cnt < num_train:
        fname = os.path.join(config.freeway_train_frames_dir, 'freeway_train_%09d.png'%cnt)
    else:
        fname = os.path.join(config.freeway_test_frames_dir, 'freeway_test_%09d.png'%cnt)
        imwrite(fname, out)
    if not cnt%100:
        print('writing',os.path.abspath(fname))
    cnt +=1

