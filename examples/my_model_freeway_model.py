# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import numpy as np
from IPython import embed
from copy import deepcopy
import logging
import os
import subprocess
import sys
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from imageio import imwrite

import gym
from models.vqvae import AutoEncoder
from models.pixel_cnn import GatedPixelCNN
from skimage.morphology import  disk
from skimage.filters.rank import  median

from models import config
from models.datasets import prepare_img, undo_img_scaling, chicken_color, input_ysize, input_xsize, base_chicken
min_pixel, max_pixel = 0., 255.
from models.utils import discretized_mix_logistic_loss, get_cuts, to_scalar
from models.utils import sample_from_discretized_mix_logistic
from planning.mcts import PTreeNode

def get_relative_bearing(gy, gx, ry, rx):
    dy = gy-ry
    dx = gx-rx
    rel_angle = (np.rad2deg(math.atan2(dy,dx))+180.0)%360
    #if rel_angle<0:
    #    rel_angle += 360.0
    return rel_angle

def goal_node_probs_fn(action_space, action_angles, goal_bearing):
    action_distances = np.abs(np.cos(np.deg2rad(action_angles-goal_bearing))-1)
    actions_and_distances = list(zip(action_space, action_distances))
    best_ = sorted(actions_and_distances, key=lambda tup: tup[1])
    best_actions = np.array([b[0] for b in best_])
    # will be between 0 and 2 because it is abs(cosine-1)
    best_angles = np.array([b[1] for b in best_])/2.0

    # need to flip it so lowest value is high
    best_angles = (1-best_angles)
    best_angles = best_angles/np.sum(best_angles)
    # goal has more influence when smoothing is near 0.0
    best_angles = best_angles*(1.0-args.smoothing)
    best_angles+=args.smoothing/float(len(action_space))
    prob_sorted_actions_and_probs = list(zip(best_actions, best_angles))
    actions_and_probs = sorted(prob_sorted_actions_and_probs, key=lambda tup: tup[0])
    return actions_and_probs

def equal_node_probs_fn(action_space, action_angles, goal_bearing):
    probs = np.ones(len(action_space))/float(len(action_space))
    actions_and_probs = list(zip(action_space, probs))
    return actions_and_probs

def get_vqvae_pcnn_model(state_index, est_inds, cond_states, num_samples=0):
    rollout_length = len(est_inds)
    print("starting vqvaepcnn - %s predictions for state index %s " %(len(est_inds), state_index))
    # normalize data before putting into vqvae
    st = time.time()
    broad_states = ((cond_states-min_pixel)/float(max_pixel-min_pixel) ).astype(np.float32)[:,None]
    # transofrms HxWxC in range 0,255 to CxHxW and range 0.0 to 1.0
    nroad_states = Variable(torch.FloatTensor(broad_states)).to(DEVICE)
    embed()
    x_d, z_e_x, z_q_x, cond_latents = vmodel(nroad_states)

    # (6,6) or (10,10)
    _, ys, xs = cond_states.shape
    latent_shape = (10,10)
    est = time.time()
    print("condition prep time", round(est-st,2))
    for ind, frame_num  in enumerate(est_inds):
        pst = time.time()
        print("predicting latent: %s" %frame_num)
        # predict next
        spat_cond = cond_latents[None].to(DEVICE)
        try:
            pred_latents = pcnn_model.generate(spatial_cond=spat_cond, shape=latent_shape, batch_size=1)
        except Exception, e:
            print('gen', e)
            embed()
        # add this predicted one to the tail
        cond_latents = torch.cat((cond_latents[1:],pred_latents))
        if not ind:
            all_pred_latents = pred_latents
        else:
            all_pred_latents = torch.cat((all_pred_latents, pred_latents))

        ped = time.time()
        print("latent pred time", round(ped-pst, 2))
    proad_states = np.zeros((rollout_length,ys,xs))
    print("starting image")
    ist = time.time()
    # generate road
    z_q_x = vmodel.embedding(all_pred_latents.view(all_pred_latents.size(0),-1))
    z_q_x = z_q_x.view(all_pred_latents.shape[0],latent_shape[0], latent_shape[1], -1).permute(0,3,1,2)
    x_d = vmodel.decoder(z_q_x)
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=True)
    proad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel

    for cc in range(num_samples):
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=False)
        sroad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
        # for each predicted state
        proad_states = np.maximum(proad_states, sroad_states)

    iet = time.time()
    print("image pred time", round(iet-ist, 2))
    return proad_states.astype(np.int)[:,0]

def get_zero_model(state_index, est_inds, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions" %rollout_length)
    # normalize data before putting into vqvae
    return np.zeros_like(ref_frames_prep[est_inds])

def get_none_model(state_index, est_inds, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions for state index %s " %(len(est_inds), state_index))
    # normalize data before putting into vqvae

    scaled_est_inds = np.array([(start_index+(e*frame_skip)) for e in est_inds])
    max_frame_ind = ref_frames_prep.shape[0]-1
    scaled_est_inds[scaled_est_inds>max_frame_ind] = max_frame_ind

    #print('forward for state index', state_index)
    #print(est_inds)
    #print(scaled_est_inds)
    return ref_frames_prep[scaled_est_inds]

def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_false_neg_counts(true_road_map, pred_road_map):
    road_true_road_map = deepcopy(true_road_map)
    road_pred_road_map = deepcopy(pred_road_map)
    road_true_road_map[road_true_road_map>0] = 1
    road_pred_road_map[road_pred_road_map>0] = 1
    # false_neg predict free where there was car # bad
    false_neg = (road_true_road_map*np.abs(road_true_road_map-road_pred_road_map))
    false_neg[false_neg>0] = 1
    false_neg_count = false_neg.sum()
    false_pos = road_pred_road_map*np.abs(road_true_road_map-road_pred_road_map)
    false_neg = road_true_road_map*np.abs(road_true_road_map-road_pred_road_map)
    error = np.ones_like(road_true_road_map)*254
    error[false_pos>0] = 30
    error[false_neg>0] = 1
    return false_neg_count, error

def get_local_image(umap, loc, ysize, xsize, buffer_size=5):
    bs = buffer_size
    if len(loc[0]):
        ry = loc[0][0]
        rx = loc[1][0]
    else:
        ry = 0; rx = 0
    lby = min(ry+bs, ysize-1)
    lbx = min(rx+bs, xsize-1)
    iby = max(ry-bs, 0)
    ibx = max(rx-bs, 0)
    loc_image = deepcopy(umap[iby:lby, ibx:lbx])
    return loc_image

def get_local_false_neg_counts(true_map, loc,  pred_map, ysize, xsize, buffer_size=5):
    # predict free where there was car  # bad
    false_neg = (true_map*np.abs(true_map-pred_map))
    loc_image_ones = get_local_image(false_neg, loc, ysize, xsize, buffer_size=5)
    loc_image_ones[loc_image_ones>0] = 1
    local_false_neg_count = np.sum(loc_image_ones)
    return local_false_neg_count, loc_image_ones

class PMCTS(object):
    def __init__(self, env, random_state,
                       estimator,
                       node_probs_fn, c_puct=1.4, future_discount=1,
                       n_playouts=1000, rollout_length=20,
                       history_size=4):
        # use estimator for planning, if false, use env
        # make sure it does a full rollout the first time
        self.full_rollouts_every = args.full_rollouts_every
        self.last_full_rollout = 1000000
        self.env = env
        self.rdn = random_state
        self.node_probs_fn = node_probs_fn
        self.root = PTreeNode(None, prior_prob=1.0, name=(0,-1))
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.tree_subs_ = []
        self.warn_at_tree_size = 1000
        self.tree_subs_ = []
        self.step = 0
        self.rollout_length = rollout_length
        self.nodes_seen = {}
        self.estimator = eval('get_'+estimator) # get_vqvae_pcnn_model
        self.road_map_ests = np.zeros((self.env.max_steps, self.env.ysize, self.env.xsize))
        self.history_size = history_size
        # infil the first road maps
        #  what was estimated when we received a state
        self.decision_time_road_maps = []

    def get_children(self, node):
        print('node name', node.name)
        for i in node.children_.keys():
            print(node.children_[i].__dict__)
        return [node.children_[i].__dict__ for i in node.children_.keys()]

    def playout(self, playout_num, state, state_index):
        # set new root of MCTS (we've taken a step in the real game)
        # only sets robot and goal states
        # get future playouts from past states
        cnt = 0
        logging.debug('+++++++++++++START PLAYOUT NUM: {} FOR STATE: {}++++++++++++++'.format(playout_num,state_index))
        init_state = state
        init_state_index = state_index
        node = self.root
        won = False
        frames = []
        # stack true state then vstate
        reward = 0
        vstate = self.road_map_ests[state_index]
        frames.append((state, vstate))
        # always use true state for first
        finished = False
        next_steps = []
        while True:
            if node.is_leaf():
                if (not finished) and (state_index+1 < self.last_state_index_est):
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action

                    loc, tobs = state
                    gb = get_relative_bearing(self.goal_location[0], self.goal_location[1], loc[0][0], loc[1][0])
                    actions_and_probs = self.node_probs_fn(self.env.action_space, self.env.action_angles,  gb)
                    node.expand(actions_and_probs)
                    reward = self.rollout_from_state(state, state_index, cnt, reward, next_steps)
                # finished the rollout
                node.update(reward)
                if reward > 0:
                    node.n_wins+=1
                    won = True
                return won, reward
            else:
                # greedy select
                # trys actions based on c_puct
                action, new_node = node.get_best(self.c_puct)
                # cant actually use next state because we dont know it
                next_state_index = state_index + 1
                vnext_road_map = self.road_map_ests[next_state_index]

                next_vstate, r, finished, next_steps = self.env.model_step(state, state_index, vnext_road_map, action, next_steps)
                reward += r
                logging.debug("GREEDY SELECT state_index:%s action:%s reward:%s finished %s" %(state_index,action,reward,finished))
                cnt+=1
                node = new_node
                state = next_vstate
                state_index = next_state_index
                self.add_agent_playout(state, state_index, reward)

    def get_rollout_action(self, state):
        loc, tobs = state
        gb = get_relative_bearing(self.goal_location[0], self.goal_location[1], loc[0][0], loc[1][0])
        if self.rdn.rand()<args.neo_goal_prior:
            actions_and_probs = self.node_probs_fn(self.env.action_space, self.env.action_angles,  gb)
        else:
            actions_and_probs = equal_node_probs_fn(self.env.action_space, self.env.action_angles,  gb)
        #actions_and_probs = equal_node_probs_fn(self.env.action_space, self.env.action_angles,  gb)
        acts, probs = zip(*actions_and_probs)
        act = self.rdn.choice(acts, p=probs)
        return act, actions_and_probs

    def rollout_from_state(self, state, state_index, cnt, reward, next_steps):
        logging.debug('-------------------------------------------')
        logging.debug('starting random rollout from state: {} limit {} rollout length'.format(state_index,self.rollout_length))
        # comes in already transformed
        c = 0
        locations = []
        actions = []
        finished = False
        while not finished:
            # one less because we want next_state to be modeled
            if state_index < self.last_state_index_est-1:
                action, action_probs = self.get_rollout_action(state)
                logging.debug("rollout --- state_index %s action %s"%( state_index,action))
                next_state_index = state_index + 1
                vnext_road_map = self.road_map_ests[next_state_index]
                next_vstate, r, finished, next_steps = self.env.model_step(state, state_index, vnext_road_map, action, next_steps)
                reward +=r
                state_index = next_state_index
                #print("steppping", state[0][0], next_vstate[0][0])
                state = next_vstate
                # get robot location from previous step
                self.add_agent_playout(state, state_index, reward)
                loc = next_vstate[0]
                actions.append(action)
                locations.append(np.min(loc[0]))
                if next_vstate[1].sum() < 1:
                    print("rollout next_state has no sum!", next_state_index)
                    embed()
                # true and vq state
                c+=1
                if finished:
                    logging.debug('finished rollout after {} steps with reward {}'.format(c,reward))
            else:
                # stop early
                logging.debug('stopping rollout after {} steps with reward {}'.format(c,reward))
                break
        #print('state index rollout', state_index)
        #print(locations)
        #print(actions)
        #embed()
        return reward

    def reset_playout_states(self, start_state_index, length):
        self.start_state_index = start_state_index
        self.playout_agents = np.zeros((length, self.env.ysize, self.env.xsize))
        self.playout_road_maps = np.zeros((length, self.env.ysize, self.env.xsize))
        self.playout_agent_locs_y = {}
        self.playout_agent_locs_x = {}

    def get_relative_index(self, state_index):
        return state_index-self.start_state_index

    def add_agent_playout(self, state, state_index, bonus):
        # bonus - can feed in reward and we will convert it to pixel color
        relative_state = self.get_relative_index(state_index)
        loc = state[0]
        #print('robot playout', self.start_state_index, state_index, relative_state)
        if relative_state not in self.playout_agent_locs_y.keys():
            self.playout_agent_locs_y[relative_state] = []
            self.playout_agent_locs_x[relative_state] = []
        try:
            self.playout_agents[relative_state,loc[0],loc[1]] = self.env.agent_color
            self.playout_agent_locs_y[relative_state].append(np.min(loc[0]))
            self.playout_agent_locs_x[relative_state].append(np.min(loc[1]))
        except Exception, e:
            print(e, 'rob')
            embed()

    def get_action_probs(self, state, state_index, temp=1e-2):
        #print("-----------------------------------")
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0
        logging.debug("starting playouts for state_index %s" %state_index)
        # only run last rollout
        for n in range(self.n_playouts):
            from_state = deepcopy(state)
            from_state_index = deepcopy(state_index)
            self.playout(n, from_state, from_state_index)
        act_visits = [(act,float(node.n_visits_)) for act, node in self.root.children_.items()]
        try:
            actions, visits = zip(*act_visits)
        except Exception, e:
            print("ACTIONS VISITS")
            print(e)
            embed()
        action_probs = softmax(1.0/temp*np.log(visits))
        return actions, action_probs


    def sample_action(self, state, state_index, temp=1E-3, add_noise=True,
                      dirichlet_coeff1=0.25, dirichlet_coeff2=0.3):
        vsz = len(self.state_manager.get_action_space())
        act_probs = np.zeros((vsz,))
        acts, probs = self.get_action_probs(state, temp)
        act_probs[list(acts)] = probs
        if add_noise:
            act = self.random_state.choice(acts, p=(1. - dirichlet_coeff1) * probs + dirichlet_coeff1 * self.random_state.dirichlet(dirichlet_coeff2 * np.ones(len(probs))))
        else:
            act = self.random_state.choice(acts, p=probs)
        return act, act_probs

    def estimate_the_future(self, state, state_index):
        #######################
        # determine how different the predicted was from true for the last state
        # pred_road should be all zero if no cars have been predicted
        loc,tobs = state

        self.goal_location = [0,21]
        self.decision_time_road_maps.append(tobs)
        false_neg_count, error = get_false_neg_counts(tobs, self.road_map_ests[state_index])
        local_false_neg_count,loc_image = get_local_false_neg_counts(tobs, loc, self.road_map_ests[state_index], self.env.ysize, self.env.xsize, 8)

        # error are 244,256
        print("ESTIMATING FUTURE", state_index)
        #if state_index>4:
        #    tobs[0,0] = 255
        #    eobs = deepcopy(self.road_map_ests[state_index])
        #    eobs[0,0] = 255
        #    if local_false_neg_count > 0:
        #        imwrite('tobs.png', tobs)
        #        imwrite('eobs.png', eobs)
        #        embed()
        # false_neg_count is ~ 25 when the pcnn predicts all zeros
        print('false neg is', false_neg_count, 'local false neg is', local_false_neg_count, 'last full rollout', self.last_full_rollout)
        if (local_false_neg_count > 0) or (false_neg_count > 50) or (self.last_full_rollout >= self.full_rollouts_every):
            self.last_full_rollout = 1
            print("running all rollouts")
            # ending index is noninclusive
            # starting index is inclusive
            est_from = state_index+1
            pred_length = self.rollout_length
            self.last_full_rollout = 0
        else:
            print("running one rollout")
            # only run last rollout that was not finished
            est_from = state_index+self.rollout_length
            pred_length = 1
            self.last_full_rollout += 1
        try:
            # put in the true road map for this step
            self.road_map_ests[state_index] = tobs
            s = range(self.road_map_ests.shape[0])
            # limit prediction lengths
            est_from = min(self.road_map_ests.shape[0], est_from)
            est_to = est_from + pred_length
            est_to = min(self.road_map_ests.shape[0], est_to)
            cond_to = est_from
            # end of conditioning
            cond_from = cond_to-self.history_size
            rinds = range(est_from, est_to)
            print("for state index: %s, estimating future to %s - actual steps %s"%(state_index, est_to-1, len(rinds)))
            print(rinds)

            self.last_state_index_est = est_to
            if not len(rinds):
                ests = []
            else:
                # can use past frames because we add them as we go
                cond_frames = self.road_map_ests[cond_from:cond_to]
                ests = self.estimator(state_index, rinds, cond_frames)
                self.road_map_ests[rinds] = ests
                est_inds = range(state_index,est_to)
                #print('this_rollout', est_inds) #should be rollout_length +1 for the current_state
                self.playout_road_maps = self.road_map_ests[state_index:est_to]
                self.playout_goal_locs = []
                # start assuming it is zero
                #gl = [[self.env.ysize//2], [self.env.xsize//2]]
                #for i, rm in enumerate(self.playout_road_maps):
                #    ccc = state_index+i
                #    false_neg_count, err = get_false_neg_counts(ref_frames_prep[ccc], self.road_map_ests[ccc])
            false_negs = []
            #for xx, i in enumerate(rinds):
            #    fnc,fn = get_false_neg_counts(ref_frames_prep[i], self.road_map_ests[i])
            #    false_negs.append(fnc)
        except Exception, e:
            print(e)
            embed()

    def get_best_action(self, observ, state_index):
        state = prepare_img(observ)
        loc, tobs = state
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        if (state_index > self.history_size):
            self.reset_playout_states(state_index, self.rollout_length+1)
            #print("rolling out to find action", state_index, loc)
            self.estimate_the_future(state, state_index)
            acts, probs = self.get_action_probs(state, state_index, temp=1e-3)
            act = self.rdn.choice(acts, p=probs)
            is_random = False
        else:
            self.reset_playout_states(state_index, 1)
            print("not enough states seen to predict, choosing random action", state_index)
            probs = np.ones(len(self.env.action_space),dtype=np.float)/float(len(self.env.action_space))
            act = self.rdn.choice(self.env.action_space)
            is_random = True

        logging.info("mcts chose action {} in state: {}".format(act,state_index))
        return act, probs, is_random

    def update_tree_move(self, action):
        # keep previous info
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_tree_size:
                logging.warn("WARNING: over {} tree_subs_ detected".format(len(self.tree_subs_)))
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            logging.error("Move argument {} to update_to_move not in actions {}, resetting".format(action, self.root.children_.keys()))

    def reset_tree(self):
        logging.warn("Resetting tree")
        self.root = PTreeNode(None, prior_prob=1.0, name=(0,-1))
        self.tree_subs_ = []


def plot_playout_scatters(true_env, base_path,  fname,
                         model_type,
                         seed, reward,rewards,
                         observed_frames, playout_frames,
                         model_road_maps,  rollout_length,
                         plot_error=True, gap=3, min_agents_alive=4, start_index=0,
                         do_plot_playouts=False, history_size=4):
    plt.ioff()
    pfpath = os.path.join(base_path,model_type,'E_seed_%04d_si_%03d_%s'%(seed,start_index,fname))

    if do_plot_playouts:
        if not os.path.exists(pfpath):
            os.makedirs(pfpath)

    fast_path = os.path.join(base_path,model_type,'T_seed_%04d_si_%03d_%s'%(seed,start_index,fname))
    if not os.path.exists(fast_path):
        os.makedirs(fast_path)

    start_state_index = playout_frames[0]['state_index']
    last_state_index = playout_frames[-1]['state_index']
    total_steps = start_state_index+len(playout_frames)


    fast_path = os.path.abspath(fast_path)
    fast_gif_path = os.path.join(fast_path, 'a_fast_seed_{}.gif'.format(seed))
    cmd = 'convert -delay 1/30 %s/*.png %s\n'%(fast_path, fast_gif_path)
    fast_sh_path = os.path.join(fast_path, 'run_fast_seed_{}.sh'.format(seed))
    of = open(fast_sh_path, 'w')
    of.write(cmd)
    of.close()


    for ts, step_frame in enumerate(playout_frames):
        state_index = step_frame['state_index']
        print("plotting true frame {}/{} state_index {}/{}".format(ts,total_steps,state_index, last_state_index))
        true_obs = observed_frames[state_index]
        true_state = prepare_img(true_obs)
        true_frame = true_env.get_state_plot(true_state)
        try:
            fast_fname = 'fast_seed_%06d_step_%04d.png'%(seed, state_index)
            ft,axt=plt.subplots(1,1, figsize=(3,3))
            axt.imshow(true_frame, vmin=0, vmax=255 )
            axt.set_title("true step:{}/{} reward {}".format(ts,total_steps,reward))
            ft.tight_layout()
            plt.savefig(os.path.join(fast_path,fast_fname))
            plt.close()
        except Exception, e:
            print(e, 'plot')
            embed()

        # playtouts is size episode_length, y, x
        c = 0

    #print('writing gif')
    #try:
    #    subprocess.call(['sh', fast_sh_path])
    #except Exception, e:
    #    print(e); embed()
    print("FINISHED WRITING TO", os.path.split(fast_sh_path)[0])


    if do_plot_playouts:

        for ts, step_frame in enumerate(playout_frames):
            state_index = step_frame['state_index']
            print("plotting true frame {}/{} state_index {}/{}".format(ts,total_steps,state_index, last_state_index))
            true_obs = observed_frames[state_index]
            true_state = prepare_img(true_obs)
            true_frame = true_env.get_state_plot(true_state)
            # playtouts is size episode_length, y, x
            c = 0

            playout_agent_states = step_frame['playout_agent_states']
            playout_agent_locs_y = step_frame['playout_agent_locs_y']
            playout_agent_locs_x = step_frame['playout_agent_locs_x']
            playout_model_states = step_frame['playout_model_states']
            num_playout_steps = playout_model_states.shape[0]
            plot_inds = range(0,num_playout_steps,gap)
            if (num_playout_steps-1) not in plot_inds:
                plot_inds.append(num_playout_steps-1)
            for playout_ind in plot_inds:
                playout_state_index = min(state_index+playout_ind, ref_frames_prep.shape[0]-1)
                print("plotting playout state_index {}/{} - {} step {}/{}".format(
                                                state_index, total_steps, playout_state_index, playout_ind, num_playout_steps))
                ref_ind = min(((start_index+(ts*frame_skip)) + (playout_ind*frame_skip)), ref_frames_prep.shape[0]-1)
                print("step",ts,'ref', ref_ind, 'play',playout_ind)
                true_playout_frame = ref_frames_prep[ref_ind]
                est_playout_frame = playout_model_states[playout_ind]
                _, rollout_model_error  = get_false_neg_counts(deepcopy(true_playout_frame), deepcopy(est_playout_frame))
                fname = 'seed_%06d_tstep_%04d_pstep_%04d_ps_%04d.png'%(seed, state_index, playout_state_index, playout_ind)
                f,ax=plt.subplots(1,4, figsize=(12,3))
                ax[0].imshow(true_frame, vmin=0, vmax=255 )
                ax[0].set_title("decision t: {}/{}".format(state_index,last_state_index))
                ax[1].imshow(true_playout_frame, vmin=0, vmax=255 )
                ax[1].set_title("oracle rollout {} step:{} {}/{}".format(state_index, playout_state_index, playout_ind, num_playout_steps))
                ax[2].imshow(est_playout_frame,  vmin=0, vmax=255 )
                ax[2].set_title("model rollout {} step: {} {}/{}".format(state_index, playout_state_index, playout_ind, num_playout_steps))
                ax[3].imshow(rollout_model_error, cmap='Set1')
                ax[3].set_title("error in model")
                if playout_ind in playout_agent_locs_y.keys():
                    agent_x = playout_agent_locs_x[playout_ind]
                    agent_y = playout_agent_locs_y[playout_ind]
                    ax[1].scatter(agent_x, agent_y, alpha=0.5, s=4, c='y')
                    ax[2].scatter(agent_x, agent_y, alpha=0.5, s=4, c='y')

                f.tight_layout()
                plt.savefig(os.path.join(pfpath,fname))
                plt.close()
        print("making gif")
        gif_path = 'a_seed_{}.gif'.format(seed)
        search = os.path.join(pfpath, 'seed_*.png')
        cmd = 'convert -delay 1/100000 *.png %s \n'%( gif_path)
        sh_path = os.path.join(pfpath, 'run_seed_{}.sh'.format(seed))
        sof = open(sh_path, 'w')
        sof.write(cmd)
        sof.close()



class FreewayEnv():
    def __init__(self, action_space, model_step_fn='equiv_model_step', seed=455, ysize=input_ysize, xsize=input_xsize, frame_skip=4):
        self.model_step = eval('self.'+model_step_fn)
        self.action_space = action_space
        self.rdn = np.random.RandomState(seed)
        # TODO - these are not right
        self.frame_skip = frame_skip
        self.max_steps = np.int(18000/float(self.frame_skip))+self.frame_skip
        if self.frame_skip == 4:
            self.step_offset = 2
        self.agent_color = chicken_color
        self.ysize = ysize
        self.xsize = xsize
        #      90
        #      |
        #180 --a-- 0
        #      |
        #     270
        self.action_angles = [0.0,90.0,270.0]
        self.state_index = 0
        self.step = 0
        self.do_nothing_action = 0
        self.finished = False
        self.total_reward = 0
        self.penalty_stuck = np.int(25/float(self.frame_skip))

    def get_state_plot(self, state):
        loc, road = deepcopy(state)
        road[loc[0], loc[1]] = self.agent_color
        return road

    def check_for_goal(self, next_agent_location, goal_location=[0]):
        # check y index
        if min(next_agent_location[0]) in goal_location:
            return 1
        else:
            return 0

    def step_up(self, agent_location):
        ny = agent_location[0]
        nx = agent_location[1]
        ny = np.array([yy-self.step_offset for yy in ny])
        ny[ny<0] = 0
        next_agent_location = (ny, nx)
        return next_agent_location


    def step_down(self, agent_location):
        ny = agent_location[0]
        nx = agent_location[1]
        ny = np.array([yy+self.step_offset for yy in ny])
        ny[ny>(self.ysize-1)] = self.ysize-1
        next_agent_location = (ny, nx)
        return next_agent_location

    def determine_collisions(self, al, input_img):
        wide_chicken_y = al[0]
        wide_chicken_x = al[1]
        for i in [-1, 1, -2, 2]:
            wide_chicken_y = np.hstack((wide_chicken_y, wide_chicken_y+i) )
            wide_chicken_x = np.hstack((wide_chicken_x, wide_chicken_x+i) )
        select1 = wide_chicken_y < self.ysize
        select2 = wide_chicken_x < self.xsize
        select3 = wide_chicken_y >= 0
        select4 = wide_chicken_x >= 0
        select = np.all([select1, select2, select3, select4], axis=0)
        wide_chicken = (wide_chicken_y[select], wide_chicken_x[select])
        ss = input_img[wide_chicken].sum()
        if ss:
            return True
        else:
            return False

    def equiv_model_step(self, state, state_index, next_road, action, next_steps=[]):
        #self.observ, self.reward, self.finished, _ = self.env.step(action)
        # agent location is a tuple of y,x
        # if 'stunned' - stay still
        finished = False
        loc, this_road = state
        al = deepcopy(loc)
        if not len(next_steps):
            next_step = 'free'
            if action == 1:
                next_agent_location = self.step_up(al)
            elif action == 2:
                next_agent_location = self.step_down(al)
            else:
                next_agent_location = al
        else:
            next_step = next_steps.pop(0)
            if next_step == 'penalty':
                next_agent_location = self.step_down(al)
            if next_step == 'stuck':
                next_agent_location = al
        # this prob isnt right
        reward = 0
        nal = deepcopy(next_agent_location)
        if next_step is 'free':
            next_steps = []
            if self.determine_collisions(nal, next_road):
                # penalty
                for i in range(self.penalty_stuck):
                    next_steps.append('penalty')
                for i in range(self.penalty_stuck):
                    next_steps.append('stuck')
            else:
                reward = self.check_for_goal(nal)
                if reward == 1:
                    finished = True

        next_state = [next_agent_location, next_road]
        return next_state, reward, finished, next_steps

def frame_skip_step(ee, lo, action, reward, done, num_steps, rindex):
    observ = lo
    # 0th frame is not maxed
    for i in range(num_steps):
        if not done:
            #print('stepping from ', rindex)
            o, r, done, _ = ee.step(action)
            reward +=r
            observ = np.maximum(o, lo)
            rindex +=1
            lo = o
            #if ref_frames[rindex].sum() != observ.sum():

            #    # these wont match bc when the chicken moves, it leaves a change
            #    # of pixel values slightly - these are the same colors as cars,
            #    # so they can't actually be removed
            #    (pr,vr) = np.unique(ref_frames[rindex], return_counts=True)
            #    (po,vo) = np.unique(observ, return_counts=True)
            #    print('not equal0', rindex)

            #    print(pr)
            #    print(po)
            #    print(vr)
            #    print(vo)
            #    if po.sum() != pr.sum():
            #        for p in po:
            #            if p not in pr:
            #                print(p, 'not in ref')
            #    else:
            #        for xx in range(vr.shape[0]):
            #            if vr[xx] != vo[xx]:
            #                print('val:', pr[xx], vr[xx], vo[xx])
            #    embed()
    return ee, observ, lo, reward, done, rindex


def run_trace(fname, seed=3432,
        n_playouts=50,
        max_rollout_length=10, estimator='empty',
        prob_fn=goal_node_probs_fn, fskip=4,
        history_size=4, start_index=0,
        do_render=False):

    global frame_skip
    global ref_state_index
    frame_skip = fskip
    ref_state_index = 0
    # log params
    states = []

    true_env = gym.make('FreewayNoFrameskip-v4')
    action_space = range(true_env.action_space.n)
    model_env = FreewayEnv(action_space, model_step_fn=args.env_model_type, seed=seed, frame_skip=frame_skip)
    mcts_rdn = np.random.RandomState(seed+1)
    local_rdn = np.random.RandomState(seed)
    #print("starting at index", start_index)
    results = {'decision_ts':[],'decision_sts':[],'frame_skip':frame_skip,
               'dis_to_goal':[], 'actions':[],
               'n_playouts':n_playouts, 'seed':seed,
               'ests':[], 'est_inds':[], 'start_index':start_index,
               'max_rollout_length':max_rollout_length}


    # prepare initial
    done = False
    reward = 0
    lo  = true_env.reset()
    true_env, observ, lo, reward, done, ref_state_index = frame_skip_step(true_env, lo, 0, reward, done, start_index, ref_state_index)
    state = prepare_img(deepcopy(observ))
    loc,tobserv = state
    observ_frames = [observ]
    tobserv_frames = [tobserv]

    # fast forward history steps so agent observes 4
    pmcts = PMCTS(env=model_env,random_state=mcts_rdn,
                  node_probs_fn=prob_fn,
                  n_playouts=n_playouts,
                  rollout_length=max_rollout_length,
                  estimator=estimator,history_size=history_size)

    is_random = True
    playout_frames = []
    state_index = 0
    new_reward = 0
    rewards = []
    while not done:
        # search for best action
        st = time.time()
        action, action_probs, is_random = pmcts.get_best_action(deepcopy(observ), state_index)
        print("CHOSE ACTION", action)
        et = time.time()
        # do action for frame_skip times
        true_env, observ, lo,  reward, done, ref_state_index = frame_skip_step(true_env, lo, action, reward, done, frame_skip, ref_state_index)
        print("ACTUAL STEP", state_index, ref_state_index, reward, new_reward)
        observ_frames.append(observ)
        rewards.append(reward)
        playout_frames.append({'state_index':state_index,
                               'playout_agent_states':deepcopy(pmcts.playout_agents),
                               'playout_agent_locs_y':deepcopy(pmcts.playout_agent_locs_y),
                               'playout_agent_locs_x':deepcopy(pmcts.playout_agent_locs_x),
                               'playout_model_states':deepcopy(pmcts.playout_road_maps),
                               })

        print("decision took %s seconds"%round(et-st, 2))
        results['decision_sts'].append(st)
        results['decision_ts'].append(et-st)
        results['actions'].append(action)
        if ref_state_index  > args.step_limit:
            done = True
        if not done:
            state_index +=1
            if not is_random:
                print("update tree move")
                pmcts.update_tree_move(action)
            if reward != new_reward:
                # when receiving reward, shortcut and start new tree
                print("resetting tree")
                pmcts.reset_tree()
                new_reward = reward
        else:
            results['reward'] = reward
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("robot reward={} after {} steps".format(reward,state_index))
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")

    plt.clf()
    plt.close()
    if args.save_plots:
        plot_playout_scatters(model_env, os.path.join(config.results_savedir, 'trials'), fname.replace('.pkl',''),
                          str(estimator), seed, reward, rewards,
                          observed_frames=observ_frames,
                          playout_frames=playout_frames,
                          model_road_maps=pmcts.road_map_ests,
                          rollout_length=pmcts.rollout_length,
                          plot_error=args.do_plot_error,
                          gap=args.plot_playout_gap,
                          min_agents_alive=4,start_index=start_index,
                          do_plot_playouts=args.plot_playouts,
                          history_size=history_size)
    print("FINISHED")
    return results

if __name__ == "__main__":
    import argparse
    vq_name = 'nfreeway_vqvae4layer_nl_k512_z64e00250.pkl'
    vq_name = 'nfreeway_vqvae4layer_nl_k512_z64e00250_good.gpkl'
    #pcnn_name = 'mrpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00034.pkl'
    pcnn_name = 'erpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00040_good.gpkl'

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=100, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=48, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=48, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=10, help='limit number of steps taken be random rollout')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')
    parser.add_argument('-t', '--model_type', type=str, default='vqvae_pcnn_model')
    parser.add_argument('-msf', '--env_model_type', type=str, default='equiv_model_step')
    parser.add_argument('-sams', '--num_samples', type=int , default=5)
    parser.add_argument('-gs', '--goal_speed', type=float , default=0.5)
    parser.add_argument('-neo', '--neo_goal_prior', type=float , default=0.01)
    parser.add_argument('-sm', '--smoothing', type=float , default=0.5)
    parser.add_argument('-sl', '--step_limit', type=float , default=18000)
    parser.add_argument('-as', '--agent_max_speed', type=float , default=1.0)
    parser.add_argument('-fre', '--full_rollouts_every', type=float , default=10)
    parser.add_argument('--save_pkl', action='store_false', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--do_plot_error', action='store_false', default=True)
    parser.add_argument('--plot_playouts', action='store_true', default=False)
    parser.add_argument('--save_plots', action='store_true', default=False)
    parser.add_argument('-gap', '--plot_playout_gap', type=int, default=5, help='gap between plot playouts for each step')
    parser.add_argument('-f', '--prior_fn', type=str, default='goal', help='options are goal or equal')

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    args.full_rollouts_every = min(args.full_rollouts_every, args.rollout_steps)
    goal_dis = args.max_goal_distance
    if args.prior_fn == 'goal':
        prior = goal_node_probs_fn
    else:
        prior = equal_node_probs_fn

    use_cuda = args.cuda
    seed = args.seed
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    ref_file = 'reference_freeway.npz'
    if os.path.exists(ref_file):
        print("loading reference frames")
        rf = np.load(open(ref_file, 'r'))
        ref_frames = rf['ref_frames']
        ref_frames_prep = rf['ref_frames_prep']
    else:
        print("creating reference frames")
        ref_env = gym.make('FreewayNoFrameskip-v4')
        lo = ref_env.reset()
        _, ref_tobs = prepare_img(lo)
        ref_frames = [lo]
        ref_frames_prep = [ref_tobs]
        f = False
        while not f:
            o, r, f, _ = ref_env.step(0)
            om = np.maximum(o, lo)
            _,to = prepare_img(om)
            ref_frames.append(om)
            ref_frames_prep.append(to)
            lo = o
        np.savez(open(ref_file, 'w'), ref_frames=ref_frames, ref_frames_prep=ref_frames_prep)
    #ref_env = gym.make('FreewayNoFrameskip-v4')
    #lo = ref_env.reset()
    #aref_frames = [lo]
    #renv, observ, lo, reward, done, ref_state_index = frame_skip_step(ref_env, lo, 0, 0, False, 500, 0)
    #embed()

    DIM = 256
    history_size = 4
    cond_size = history_size*DIM
    upcnn_name  = 'na'
    uvq_name = 'na'
    if args.model_type == 'vqvae_pcnn_model':
        dsize = 80
        nr_logistic_mix = 10
        probs_size = (2*nr_logistic_mix)+nr_logistic_mix
        num_z = 64
        nr_logistic_mix = 10
        num_clusters = 512
        N_LAYERS = 10 # layers in pixelcnn

        upcnn_name  = pcnn_name.split('e00')[1].replace('.pkl', '')
        uvq_name  = vq_name.split('e00')[1].replace('.pkl', '')
        default_pcnn_model_loadpath = os.path.join(config.model_savedir, pcnn_name)
        default_vqvae_model_loadpath = os.path.join(config.model_savedir, vq_name)
        if os.path.exists(default_vqvae_model_loadpath):
            vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=num_z).to(DEVICE)
            vqvae_model_dict = torch.load(default_vqvae_model_loadpath, map_location=lambda storage, loc: storage)
            vmodel.load_state_dict(vqvae_model_dict['state_dict'])
            epoch = vqvae_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_vqvae_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_vqvae_model_loadpath))
            sys.exit()

        if os.path.exists(default_pcnn_model_loadpath):
            pcnn_model = GatedPixelCNN(num_clusters, DIM, N_LAYERS,
                                        history_size, spatial_cond_size=cond_size).to(DEVICE)
            pcnn_model_dict = torch.load(default_pcnn_model_loadpath, map_location=lambda storage, loc: storage)
            pcnn_model.load_state_dict(pcnn_model_dict['state_dict'])
            epoch = pcnn_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_pcnn_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_pcnn_model_loadpath))
            sys.exit()

    for start_index in range(30):
        # load relevent file
        fname = 'afay_seed_%03d_si_%02d_sl_%09d_p%03d_r%03d_pr_%s_mod_%s_vq_%s_pcnn_%s_sam_%s_neo_%01.02f_fre_%03d_smooth_%.02f_%s.pkl' %(
                                    seed,
                                    start_index,
                                    args.step_limit,
                                    args.num_playouts,
                                    args.rollout_steps,
                                    args.prior_fn,
                                    args.model_type,
                                    uvq_name,
                                    upcnn_name,
                                    args.num_samples,
                                    args.neo_goal_prior,
                                    args.full_rollouts_every,
                                    args.smoothing,
                                    args.env_model_type)

        if not os.path.exists(config.results_savedir):
            os.makedirs(config.results_savedir)
        fpath = os.path.join(config.results_savedir, fname)
        if not os.path.exists(fpath):
            all_results = {'args':args}
            print("STARTING EPISODE start_index %s" %(start_index))
            print(args.save_pkl)
            st = time.time()
            r = run_trace(fname, seed=seed,
                          n_playouts=args.num_playouts,
                          max_rollout_length=args.rollout_steps,
                          prob_fn=prior, estimator=args.model_type,
                          history_size=history_size, start_index=start_index, do_render=args.render)

            et = time.time()
            r['full_end_time'] = et
            r['full_start_time'] = st
            r['seed'] = seed
            all_results[start_index] = r
            if args.save_pkl:
                ffile = open(fpath, 'w')
                pickle.dump(all_results,ffile)
                print("saved start_index %s"%start_index)
                ffile.close()
    embed()
    print("FINISHED")


