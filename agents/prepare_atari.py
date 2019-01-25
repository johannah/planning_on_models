from IPython import embed
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import numpy as np
import gym
import os
import warnings
from imageio import imwrite
warnings.simplefilter('ignore', UserWarning)
# per bootstrapped dqn paper
NETWORK_INPUT_SIZE = (84,84)
CLIP_REWARD_MIN = -1
CLIP_REWARD_MAX = 1

def prepare_frame(frame):
    small_frame = resize(rgb2gray(frame),NETWORK_INPUT_SIZE)
    return small_frame

# TODO - what happens when finished = True?
class DMAtariEnv():
    def __init__(self, gamename='Breakout', random_seed=223):
        self.gym_gamename = '%sNoFrameskip-v4'%gamename
        self.random_state = np.random.RandomState(random_seed)
        self.env = gym.make(self.gym_gamename)
        self.noop_action = self.env.env.get_action_meanings().index('NOOP')
        self.num_true_steps = 0
        self.num_episodes = 0
        if not os.path.exists('imgs'):
            os.makedirs('imgs')
        self.reset()

    def reset(self):
        frame = self.env.reset()
        self.total_reward = 0
        finished = False
        for i in range(self.random_state.randint(0,30)):
            # noop steps in beginning
            frame, r, finished, info = self.env.step(self.noop_action)
            obs = prepare_frame(frame)
            self.total_reward += r
        obs = prepare_frame(frame)
        if finished:
            print("received end in init routine")
            self.reset()
        return obs, self.noop_action, self.total_reward, finished

    # expects single step atari - will repeat 4 times
    def step4(self, action):
        # min frame skips is two
        reward = 0
        # frame 1
        frame1, r1, finished1, info = self.env.step(action)
        # frame 2
        frame2, r2, finished2, info = self.env.step(action)
        # frame 3
        frame3, r3, finished3, info = self.env.step(action)
        obs3 = prepare_frame(frame3)
        # frame 4
        frame4, r4, finished4, info = self.env.step(action)
        obs4 = prepare_frame(frame4)
        # take maximum to avoid frame flicker
        obs_step4 = np.maximum(obs3,obs4)
        reward = r1+r2+r3+r4
        finished = min([finished1,finished2,finished3,finished4])
        # clip bt -1 and 1
        reward_clipped = min(reward,CLIP_REWARD_MAX)
        reward_clipped = max(reward_clipped,CLIP_REWARD_MIN)
        self.num_true_steps+=4
        #if not self.num_true_steps%10:
        #    print(self.num_true_steps,reward,reward_clipped)
        #imwrite('imgs/af%05d.png'%self.num_true_steps,img_as_ubyte(frame4))
        #imwrite('imgs/ao%05d.png'%self.num_true_steps,img_as_ubyte(obs4))
        #imwrite('imgs/am%05d.png'%self.num_true_steps,img_as_ubyte(obs_step4))
        if finished:
            self.num_true_steps+=1
            #blank = np.zeros((48,48))
            #imwrite('imgs/af%05de.png'%self.num_true_steps,blank)
            #imwrite('imgs/ao%05de.png'%self.num_true_steps,blank)
            #imwrite('imgs/am%05de.png'%self.num_true_steps,blank)
            #os.system('convert imgs/af*.png imgs/af.gif')
            #os.system('convert imgs/ao*.png imgs/ao.gif')
            #os.system('convert imgs/am*.png imgs/am.gif')
            self.num_episodes +=1
        return obs_step4, reward_clipped, finished

if __name__ == '__main__':
    env = DMAtariEnv(gamename='Breakout')
    while True:
        env.step4(np.random.randint(2))
