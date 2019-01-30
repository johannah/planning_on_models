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
def prepare_frame(frame, network_input_size):
    small_frame = resize(rgb2gray(frame),network_input_size).astype(np.float32)
    return small_frame

# TODO - what happens when finished = True?
class DMAtariEnv():
    def __init__(self, gamename='Breakout', network_input_size=(84,84),
                 clip_reward_max=1, clip_reward_min=-1, random_seed=223):
        self.clip_reward_max = clip_reward_max
        self.clip_reward_min = clip_reward_min
        self.network_input_size = network_input_size
        self.gym_gamename = '%sNoFrameskip-v4'%gamename
        self.random_state = np.random.RandomState(random_seed)
        self.env = gym.make(self.gym_gamename)
        self.noop_action = self.env.env.get_action_meanings().index('NOOP')
        self.num_episodes = 0
        if not os.path.exists('imgs'):
            os.makedirs('imgs')
        self.reset()

    def reset(self):
        frame = self.env.reset()
        self.num_true_steps = 0
        self.total_reward = 0
        finished = False
        for i in range(self.random_state.randint(0,30)):
            # noop steps in beginning
            frame, r, finished, info = self.env.step(self.noop_action)
            self.start_info = info
            obs = prepare_frame(frame,self.network_input_size)
            self.total_reward += r
        obs = prepare_frame(frame,self.network_input_size)
        if finished:
            print("received end in init routine")
            self.reset()
        return obs, self.noop_action, self.total_reward, finished

    # expects single step atari - will repeat 4 times
    def step4(self, action):
        # min frame skips is two
        reward = 0.0
        # frame 1
        frame1, r1, finished1, info1 = self.env.step(action)
        # frame 2
        frame2, r2, finished2, info2 = self.env.step(action)
        # frame 3
        frame3, r3, finished3, info3 = self.env.step(action)
        obs3 = prepare_frame(frame3,self.network_input_size)
        # frame 4
        frame4, r4, finished4, info4 = self.env.step(action)
        # per mnih nature paper - end game if life lost
        end = [finished1,finished2,finished3,finished4]
        infos = [info1,info2,info3,info4]
        lives = [info!=self.start_info for info in infos]
        obs4 = prepare_frame(frame4,self.network_input_size)
        # take maximum to avoid frame flicker
        obs_step4 = np.maximum(obs3,obs4)
        reward = r1+r2+r3+r4
        finished = max(end+lives)
        self.num_true_steps+=4
        if self.num_true_steps >= 18000:
            finished = True
        if finished:
            print('finished epoch')
            #print(finished,'action',action,'lives', lives, infos)
            self.num_episodes +=1
        # clip bt -1 and 1
        reward_clipped = min(reward,self.clip_reward_max)
        reward_clipped = max(reward_clipped,self.clip_reward_min)
        return obs_step4, reward_clipped, finished

if __name__ == '__main__':
    env = DMAtariEnv(gamename='Breakout')
    while True:
        env.step4(np.random.randint(2))
