import gym
import sys, time
from imageio import imwrite
import os
datadir = 'datasets/train_freeway/'
if not os.path.exists(datadir):
    os.makedirs(datadir)
env = gym.make('Freeway-v0')
env.reset()
cnt =  0
while True:
    o, r, done, info = env.step(0)
    imwrite(os.path.join(datadir, 'freeway_%09d.png'%cnt), o)
    env.render()
    time.sleep(.1)
