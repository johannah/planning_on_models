#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#
from datasets import transform_freeway, remove_background, remove_chicken, prepare_img, undo_img_scaling
import matplotlib.pyplot as plt
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from imageio import imwrite, imread
env = gym.make('FreewayNoFrameskip-v4')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

def plot_transformation(orig_img, input_img, out_img, step):
    f, ax = plt.subplots(1,3)
    ax[0].imshow(orig_img, vmin=0, vmax=255)
    ax[0].set_title("step %s" %(step))
    ax[1].imshow(out_img, vmin=0, vmax=255)
    ax[2].imshow(input_img, vmin=0, vmax=255)
    plt.show()

#def sim_step(current_location, action):
    # action 1 is up
    # action 0 is
    # action 2 is

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    aobs = []
    tobs = []
    trans = []
    atrans = []
    win_steps = []
    obser = env.reset()
    tobs.append(obser)
    small_chicken, input_img = prepare_img(obser)
    trans.append(input_img)
    skip = 0
    total_reward = 0
    total_timesteps = 0
    step = 0
    ep_num=0

    while 1:
        a = 1
        obser, r, done, info = env.step(a)
        small_chicken,input_img = prepare_img(obser)
        tobs.append(obser)
        trans.append(input_img)
        total_timesteps += 1
        imwrite('ex/trans_ep%03d_step%03d.png'%(ep_num,step), input_img)
        step +=1
        if r != 0:
            print("reward %0.3f total_timesteps %d" % (r, total_timesteps))
            aobs.append(tobs)
            tobs = [obser]
            atrans = [trans]
            win_steps.append(total_timesteps)
            step = 0
        if done:
            ep_num+=1
            obser = env.reset()
            embed()
        total_reward += r


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    rollout(env)

