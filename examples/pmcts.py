from planning.mcts import PTreeNode
import numpy as np

def get_relative_bearing(gy, gx, ry, rx):
    # TODO - wrong?
    dy = gy-ry
    dx = gx-rx
    rel_angle = (np.rad2deg(math.atan2(dy,dx))+180.0)%360
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


class PMCTS(object):
    def __init__(self, random_state,
                       future_model,
                       node_probs_name, c_puct=1.4, future_discount=1,
                       n_playouts=1000, rollout_steps=20,
                       history_size=4):
        # use estimator for planning, if false, use env
        # make sure it does a full rollout the first time
        self.last_full_rollout = 1000000
        self.rdn = random_state
        self.node_probs_fn = eval(node_probs_name)
        self.root = PTreeNode(None, prior_prob=1.0, name=(0,-1))
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.tree_subs_ = []
        self.warn_at_tree_size = 1000
        self.tree_subs_ = []
        self.step = 0
        self.rollout_steps = rollout_steps

        self.reset_playout_states(0, self.rollout_steps)
        self.nodes_seen = {}
        self.future_model = eval('get_'+future_model) # get_vqvae_pcnn_model
        self.road_map_ests = np.zeros((model_env.max_steps, model_env.ysize, model_env.xsize))
        self.history_size = history_size
        # infil the first road maps
        #  what was estimated when we received a state

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
        # stack true state then vstate
        reward = 0
        vstate = self.road_map_ests[state_index]
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
                    actions_and_probs = self.node_probs_fn(model_env.action_space, model_env.action_angles,  gb)
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

                next_vstate, r, finished, next_steps = model_env.model_step(state, state_index, vnext_road_map, action, next_steps)
                reward += r
                logging.debug("GREEDY SELECT state_index:%s action:%s reward:%s finished %s" %(state_index,action,reward,finished))
                cnt+=1
                node = new_node
                state = next_vstate
                state_index = next_state_index
                if args.plot_playouts:
                    self.add_agent_playout(state, state_index, reward)

    def get_rollout_action(self, state):
        loc, tobs = state
        gb = get_relative_bearing(self.goal_location[0], self.goal_location[1], loc[0][0], loc[1][0])
        if self.rdn.rand()<args.neo_goal_prior:
            actions_and_probs = self.node_probs_fn(model_env.action_space, model_env.action_angles,  gb)
        else:
            actions_and_probs = equal_node_probs_fn(model_env.action_space, model_env.action_angles,  gb)
        #actions_and_probs = equal_node_probs_fn(model_env.action_space, model_env.action_angles,  gb)
        acts, probs = zip(*actions_and_probs)
        act = self.rdn.choice(acts, p=probs)
        return act, actions_and_probs

    def rollout_from_state(self, state, state_index, cnt, reward, next_steps):
        logging.debug('-------------------------------------------')
        logging.debug('starting random rollout from state: {} limit {} rollout length'.format(state_index,self.rollout_steps))
        # comes in already transformed
        iloc = loc = state[0]
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
                next_vstate, r, finished, next_steps = model_env.model_step(state, state_index, vnext_road_map, action, next_steps)
                reward +=r
                #print("steppping", state[0][0], next_vstate[0][0])
                # get robot location from previous step
                self.add_agent_playout(state, state_index, reward)
                actions.append(action)
                loc = next_vstate[0]
                if next_vstate[1].sum() < 1:
                    print("rollout next_state has no sum!", next_state_index)
                    #embed()
                # true and vq state
                c+=1
                if finished:
                    logging.debug('finished rollout after {} steps with reward {}'.format(c,reward))
                else:
                    state = next_vstate
                    state_index = next_state_index
            else:
                # stop early
                logging.debug('stopping rollout after {} steps with reward {}'.format(c,reward))
                break
        #print('state index rollout', state_index)
        #print(locations)
        #print(actions)
        #embed()
        #print("FINISHING ROLLOUT", reward, min(iloc[0]), min(loc[0]))
        return reward

    def reset_playout_states(self, start_state_index, length):
        self.start_state_index = start_state_index
        #if args.save_plots:
        self.playout_agents = np.zeros((length, model_env.ysize, model_env.xsize))
        self.playout_road_maps = np.zeros((length, model_env.ysize, model_env.xsize))
        self.playout_agent_locs_y = {}
        self.playout_agent_locs_x = {}

    def get_relative_index(self, state_index):
        return state_index-self.start_state_index

    def add_agent_playout(self, state, state_index, bonus):
        if args.plot_playouts:
            # bonus - can feed in reward and we will convert it to pixel color
            relative_state = self.get_relative_index(state_index)
            loc = state[0]
            #print('robot playout', self.start_state_index, state_index, relative_state)
            if relative_state not in self.playout_agent_locs_y.keys():
                self.playout_agent_locs_y[relative_state] = []
                self.playout_agent_locs_x[relative_state] = []
            self.playout_agents[relative_state,loc[0],loc[1]] = model_env.agent_color
            self.playout_agent_locs_y[relative_state].append(np.min(loc[0]))
            self.playout_agent_locs_x[relative_state].append(np.min(loc[1]))

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
        actions, visits = zip(*act_visits)
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
        dt_img = deepcopy(self.road_map_ests[state_index])
        loc,tobs = state
        self.goal_location = [0,21]
        false_neg_count, error = get_false_neg_counts(tobs, self.road_map_ests[state_index])
        local_false_neg_count,loc_image = get_local_false_neg_counts(tobs, loc, self.road_map_ests[state_index], model_env.ysize, model_env.xsize, buffer_size=8)
        est_from = state_index+1
        pred_length = self.rollout_steps
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
        #print(rinds)

        self.last_state_index_est = est_to
        if not len(rinds):
            ests = []
        else:
            # can use past frames because we add them as we go
            cond_frames = self.road_map_ests[cond_from:cond_to]
            ests = self.future_model(state_index, rinds, cond_frames)
            self.road_map_ests[rinds] = ests
            est_inds = range(state_index,est_to)
            #print('this_rollout', est_inds) #should be rollout_steps +1 for the current_state
            self.playout_road_maps = self.road_map_ests[state_index:est_to]
            self.playout_goal_locs = []
        return dt_img

    def get_best_action(self, observ, state_index):
        state = prepare_img(observ)
        loc, tobs = state
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        if (state_index > self.history_size):
            self.reset_playout_states(state_index, self.rollout_steps+1)
            #print("rolling out to find action", state_index, loc)
            dt_img = self.estimate_the_future(state, state_index)
            acts, probs = self.get_action_probs(state, state_index, temp=1e-3)
            act = self.rdn.choice(acts, p=probs)
            is_random = False
        else:
            self.reset_playout_states(state_index, 1)
            print("not enough states seen to predict, choosing random action", state_index)
            probs = np.ones(len(model_env.action_space),dtype=np.float)/float(len(model_env.action_space))
            act = self.rdn.choice(model_env.action_space)
            dt_img = base_img
            is_random = True

        logging.info("mcts chose action {} in state: {}".format(act,state_index))
        return act, probs, is_random, dt_img

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



