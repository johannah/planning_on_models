import os
import numpy as np
from IPython import embed
import datetime
import time
from env import Environment
from replay import ReplayMemory
from config_handler import ConfigHandler

def collect_random_experience(env, rbuffer, num_random_steps, seed, savepath):
    step_number = 1
    epoch_num = 0
    random_state = np.random.RandomState(seed)
    heads = np.arange(rbuffer.num_heads)
    while step_number < num_random_steps:
        epoch_frame = 0
        terminal = False
        life_lost = True
        # use real state
        episode_reward_sum = 0
        random_state.shuffle(heads)
        active_head = heads[0]
        print("Gathering data with head=%s"%active_head)
        # at every new episode - recalculate action/reward weight
        state = env.reset()
        while not terminal:
            action = random_state.randint(0, env.num_actions)
            next_state, reward, life_lost, terminal = env.step(action)
            # Store transition in the replay memory
            replay_memory.add_experience(action=action,
                                         frame=next_state[-1],
                                         reward=1+np.sign(reward), # add one so that rewards are <=0
                                         terminal=life_lost,
                                         )
            step_number += 1
            epoch_frame += 1
            episode_reward_sum += reward
        print("finished epoch %s: reward %s" %(epoch_num, episode_reward_sum))
        epoch_num += 1
    rbuffer.save_buffer(savepath)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config_path', help='pass name of config file that will be used to generate random data')
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--valid_only', action='store_true', default=False)
    parser.add_argument('--train_name', default='')
    parser.add_argument('--valid_name', default='')
    args = parser.parse_args()
    assert os.path.exists(args.config_path)
    ch = ConfigHandler(args.config_path)

    # create environment
    for phase in ['train', 'eval']:
        seed = ch.cfg['RUN']['%s_seed'%phase]
        if phase == 'train' and args.valid_only:
            continue
        if phase == 'valid' and  args.train_only:
            continue
        buffer_savepath = ch.cfg['RUN']['%s_buffer_name'%phase]
        if buffer_savepath == '':
            buffer_savepath = ch.get_default_random_buffer_name()

        env = Environment(rom_file=ch.cfg['ENV']['game'],
                      frame_skip=ch.cfg['ENV']['frame_skip'],
                      num_frames=ch.cfg['ENV']['history_size'],
                      no_op_start=ch.cfg['ENV']['max_no_op_frames'],
                      seed=seed,
                      dead_as_end=ch.cfg['ENV']['dead_as_end'],
                      max_episode_steps=ch.cfg['ENV']['max_episode_steps'])

        replay_memory = ReplayMemory(action_space=ch.cfg['ENV']['action_space'],
                          size=ch.cfg['RUN']['buffer_size'],
                          frame_height=ch.cfg['ENV']['obs_height'],
                          frame_width=ch.cfg['ENV']['obs_width'],
                          agent_history_length=ch.cfg['ENV']['history_size'],
                          batch_size=ch.cfg['DQN']['batch_size'],
                          num_heads=ch.cfg['DQN']['n_ensemble'],
                          bernoulli_probability=ch.cfg['DQN']['bernoulli_probability'],
                          seed=seed,
                                  )

        collect_random_experience(env, replay_memory,
                              num_random_steps=ch.cfg['RUN']['min_steps_to_learn'],
                              seed=seed, savepath=buffer_savepath)
        del replay_memory
        del env


