import gym
import sys, time
from imageio import imwrite, imread
import os
import config
import numpy as np
from IPython import embed
from datasets import transform_freeway, remove_background
from glob import glob
import matplotlib.pyplot as plt
fs = glob(os.path.join(config.freeway_train_frames_dir, '*.png'))
print("len", len(fs))
num = 10000
rdn = np.random.RandomState(100)
nn = rdn.choice(range(len(fs)), num)
empty = np.zeros((num, 80, 80))
for xx, n in enumerate(nn):
    img_name = fs[n]
    img = imread(img_name)
    empty[xx] = img


# max and min are 214, 142
o = np.zeros((80,80), np.int)
for i in range(80):
    for j in range(80):
        this_pixel = empty[:,i,j]
        (values,cnts) = np.unique(this_pixel, return_counts=True)
        o[i,j] = int(values[np.argmax(cnts)])
#imwrite('median.png', o.astype(np.uint8))
#imwrite('cars_only.png', remove_background(img).astype(np.uint8))

embed()
