import numpy as np
import torch
import os
import sys
from imageio import mimsave
import matplotlib.pyplot as plt
#from skimage.transform import resize
import cv2

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

def handle_step(random_state, cnt, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer, checkpoint='', n_ensemble=1, bernoulli_p=1.0):
    # mask to determine which head can use this experience
    exp_mask = random_state.binomial(1, bernoulli_p, n_ensemble).astype(np.uint8)
    # at this observed state
    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
    batch = replay_buffer.send((checkpoint, experience))
    # update so "state" representation is past history_size frames
    S_hist.pop(0)
    S_hist.append(S_prime)
    episodic_reward += reward
    cnt+=1
    return cnt, S_hist, batch, episodic_reward

def linearly_decaying_epsilon(num_warmup_steps, num_annealing_steps, final_epsilon, step):
    """ from Dopamine - Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
      Use epsilon from there on.
    Args:
      num_annealing_steps: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      final_epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    if step < num_warmup_steps:
        return 1.0
    if num_annealing_steps > 0:
        steps_left = num_annealing_steps + num_warmup_steps - step
        bonus = (1.0 - final_epsilon) * steps_left / num_annealing_steps
        bonus = np.clip(bonus, 0., 1. - final_epsilon)
        return final_epsilon + bonus
    else:
        return final_epsilon

def write_info_file(info, model_base_filepath, cnt):
    info_filename = model_base_filepath + "_%010d_info.txt"%cnt
    info_f = open(info_filename, 'w')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()

def generate_gif(base_dir, step_number, frames_for_gif, reward, name='', results=[], resize=True):
    if resize:
        for idx, frame_idx in enumerate(frames_for_gif):
            frames_for_gif[idx] = cv2.resize(frame_idx, (110, 160)).astype(np.uint8)
    else:
        frames_for_gif = np.array(frames_for_gif).astype(np.uint8)

    if len(frames_for_gif[0].shape) == 2:
        name+='gray'
    else:
        name+='color'
    gif_fname = os.path.join(base_dir, "ATARI_step%010d_r%04d_%s.gif"%(step_number, int(reward), name))

    print("WRITING GIF", gif_fname)
    mimsave(gif_fname, frames_for_gif, duration=1/30)
    if len(results):
        txt_fname = gif_fname.replace('.gif', '.txt')
        ff = open(txt_fname, 'w')
        for ex in results:
            ff.write(ex+'\n')
        ff.close()

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()

def matplotlib_plot_all(p, model_base_filedir):
    try:
        epoch_num = len(p['steps'])
        epochs = np.arange(epoch_num)
        steps = p['steps']
        plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_step']}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
        plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_relative_times']}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
        plot_dict_losses({'episode head':{'index':epochs, 'val':p['episode_head']}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
        plot_dict_losses({'episode loss':{'index':epochs, 'val':p['episode_loss']}}, name=os.path.join(model_base_filedir, 'episode_loss.png'))
       #plot_dict_losses({'steps eps':{'index':steps, 'val':p['eps_list']}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
        plot_dict_losses({'steps reward':{'index':steps,'val':p['episode_reward']}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
        plot_dict_losses({'episode reward':{'index':epochs, 'val':p['episode_reward']}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
        plot_dict_losses({'episode times':{'index':epochs,'val':p['episode_times']}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
        plot_dict_losses({'steps avg reward':{'index':steps,'val':p['avg_rewards']}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
        plot_dict_losses({'eval rewards':{'index':p['eval_steps'], 'val':p['eval_rewards']}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)
        for i in range(len(p['head_rewards'])):
            prange = range(len(p['head_rewards'][i]))
            pvals = p['head_rewards'][i]
            pfname = os.path.join(model_base_filedir, 'head_%02d_rewards.png'%i)
            plot_dict_losses({'head %s rewards'%i:{'index':prange, 'val':pvals}}, name=pfname, rolling_length=0)
    except Exception as e:
        print('fail matplotlib', e)
        from IPython import embed; embed()



