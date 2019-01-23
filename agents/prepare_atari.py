from IPython import embed
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import numpy as np
import gym

# per bootstrapped dqn paper
NETWORK_INPUT_SIZE = (48,48)
CLIP_REWARD_MIN = -1
CLIP_REWARD_MAX = 1

def prepare_frame(frame):
    small_frame = img_as_ubyte(resize(rgb2gray(frame),NETWORK_INPUT_SIZE))
    return small_frame

# TODO - what happens when finished = True?
class DMAtariEnv():
    def __init__(self, gamename='Breakout', random_seed=223):
        self.gym_gamename = '%sNoFrameskip-v4'%gamename
        self.random_state = np.random.RandomState(random_seed)
        self.env = gym.make(self.gym_gamename)
        self.noop_action = self.env.env.get_action_meanings().index('NOOP')
        self.reset()

    def reset(self):
        frame = self.env.reset()
        self.total_reward = 0
        for i in range(self.random_state.randint(0,30)):
            # noop steps in beginning
            frame, r, finished, info = self.env.step(self.noop_action)
            self.total_reward += r
        obs = prepare_frame(frame)
        self.num_true_steps = 0
        self.num_episodes = 0
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
        reward_clipped = max(reward,CLIP_REWARD_MAX)
        reward_clipped = min(reward_clipped,CLIP_REWARD_MIN)
        self.num_true_steps+=4
        return obs_step4, reward_clipped, finished

if __name__ == '__main__':
    env = DMAtariEnv(gamename='Breakout')
    oo = env.step4(1)
    embed()
