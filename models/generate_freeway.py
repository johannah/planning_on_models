import gym
import sys, time
from imageio import imwrite
import os
import config
import numpy as np
from IPython import embed
from config import freeway_gt, freeway_test_frames_dir, freeway_train_frames_dir
from datasets import transform_freeway, remove_background
bs = 64
num_train = 1000*bs
num_test = 4*bs
rdn = np.random.RandomState(555)

for dd in [freeway_gt_dir, config.freeway_test_frames_dir, config.freeway_train_frames_dir]:
    if not os.path.exists(dd):
        os.makedirs(dd)

env = gym.make('FreewayNoFrameskip-v4')
last_o = env.reset()
cnt =  0
while cnt<(num_train+num_test):
    o, r, done, info = env.step(0)
    if done:
        last_o = env.reset()
    max_o = np.maximum(last_o, o)
    out = transform_freeway(max_o)
    outb = remove_background(out)
    if cnt < num_train:
        fname = os.path.join(config.freeway_train_frames_dir, 'freeway_train_%09d.png'%cnt)
    else:
        fname = os.path.join(config.freeway_test_frames_dir, 'freeway_test_%09d.png'%cnt)
    if not cnt%100:
        print('writing',os.path.abspath(fname))
    imwrite(fname, outb.astype(np.uint8))
    last_o = o
    cnt +=1

