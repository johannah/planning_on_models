# Author: Kyle Kastner
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# See similar implementation here
# https://github.com/junxiaosong/AlphaZero_Gomoku

# changes from high level pseudo-code in survey
# expand all children, but only rollout one
# section biases to unexplored nodes, so the children with no rollout
# will be explored quickly

import numpy as np
from IPython import embed

def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class CountingManager(object):
    def __init__(self):
        self.size = 5
        self.random_state = np.random.RandomState(1999)

    def get_next_state(self, state, action):
        if action == state:
            return state + 1
        else:
            return 0

    def get_valid_actions(self, state):
        return tuple(range(self.size))

    def get_init_state(self):
        return 0

    def _rollout_fn(self, state):
        return self.random_state.choice(self.get_valid_actions(state))

    def rollout_from_state(self, state, rollout_limit):
        s = state
        w, e = self.is_finished(s)
        if e:
            return 1.

        c = 0
        while True:
            a = self._rollout_fn(s)
            s = self.get_next_state(s, a)
            w, e = self.is_finished(s)
            c += 1
            if e:
                return 1. / float(c)
            if c > rollout_limit:
                return 0.

    def is_finished(self, state):
        if state == self.size - 1:
            return 1, True
        else:
            return 0, False


class TreeNode(object):
    def __init__(self, parent):
        self.parent = parent
        self.W_ = 0
        # action -> tree node
        self.children_ = {}
        self.n_visits_ = 0

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                self.children_[action] = TreeNode(self)

    def is_leaf(self):
        return self.children_ == {}

    def is_root(self):
        return self.parent is None

    def _update(self, value):
        self.n_visits_ += 1
        self.W_ += value

    def update(self, value):
        if self.parent != None:
            # negative in the original code due to being the opposing player
            #self.parent.update(-value)
            self.parent.update(value)
        self._update(value)

    def get_value(self, c_uct):
        if self.n_visits_ == 0:
            lp = 0.
        else:
            lp = self.W_ / float(self.n_visits_)
        if self.n_visits_ == 0:
            rp = np.inf
        else:
            rp = c_uct * np.sqrt(2 * np.log(self.parent.n_visits_) / float(self.n_visits_))
        return lp + rp

    def get_best(self, c_uct):
        #best = max(self.children_.iteritems(), key=lambda x: x[1].get_value(c_uct))
        # python 3 uses items() rather than iteritems()
        best = max(self.children_.items(), key=lambda x: x[1].get_value(c_uct))
        return best


class MCTS(object):
    def __init__(self, state_manager, c_uct=1.4, n_playout=100, random_state=None):
        if random_state is None:
            raise ValueError("Must pass random_state object")
        self.random_state = random_state
        self.root = TreeNode(None)
        # state manager must, itself have *NO* state / updating behavior
        # internally. Otherwise we need deepcopy() in get_move_probs
        self.state_manager = state_manager
        self.c_uct = c_uct
        self.n_playout = n_playout
        self.tree_subs_ = []
        self.warn_at_ = 10000

    def playout(self, state):
        # transform to latents state that is needed
        prev_states = []
        node = self.root
        while True:
            if node.is_leaf():
                #winner, end = self.state_manager.is_finished(state)
                # I dont have a way to determine if agent is dead rn
                end = False
                if not end:
                    # uniform prior probs
                    actions = self.state_manager.get_valid_actions(state)
                    probs = np.ones((len(actions))) / float(len(actions))
                    actions_and_probs = list(zip(actions, probs))
                    node.expand(actions_and_probs)
                # randomly walk
                value = self.state_manager.rollout_from_state(state)
                # negative in the original code due to being the opposing player
                #node.update(-value)
                node.update(value)
                return value
            else:
                # if we've seen this state before -
                action, node = node.get_best(self.c_uct)
                state = self.state_manager.get_next_state(state, action)
                prev_states.append(state)


    def get_action_probs(self, state, temp=1E-3):
        # low temp -> nearly argmax
        for n in range(self.n_playout):
            self.playout(state)

        act_visits = [(act, node.n_visits_) for act, node in self.root.children_.items()]
        actions, visits = zip(*act_visits)
        action_probs = softmax(1. / temp * np.log(visits))
        return actions, action_probs

    def sample_action(self, state, temp=1E-3, add_noise=True,
                      dirichlet_coeff1=0.25, dirichlet_coeff2=0.3):
        vsz = len(self.state_manager.get_valid_actions(state))
        act_probs = np.zeros((vsz,))
        acts, probs = self.get_action_probs(state, temp)
        act_probs[list(acts)] = probs
        if add_noise:
            act = self.random_state.choice(acts, p=(1. - dirichlet_coeff1) * probs + dirichlet_coeff1 * self.random_state.dirichlet(dirichlet_coeff2 * np.ones(len(probs))))
        else:
            act = self.random_state.choice(acts, p=probs)
        return act, act_probs

    def get_action(self, state):
        vsz = len(self.state_manager.get_valid_actions(state))
        act_probs = np.zeros((vsz,))
        # temp doesn't matter for argmax
        acts, probs = self.get_action_probs(state, temp=1.)
        act_probs[list(acts)] = probs
        maxes = np.max(act_probs)
        opts = np.where(act_probs == maxes)[0]
        if len(opts) > 1:
            # if 2 options are *exactly* equal, just choose 1 at random
            self.random_state.shuffle(opts)
        act = opts[0]
        return act, act_probs

    def update_tree_root(self, action):
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_:
                print("WARNING: Over {} tree_subs_ detected, watch memory".format(self.warn_at_))
                # only print the warning a few times
                self.warn_at_ = 10 * self.warn_at_
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            raise ValueError("Action argument {} neither in root.children_ {} and not == -1 (reset)".format(self.root.children_.keys()))

    def reconstruct_tree(self):
        # walk the list back to front, putting parents back in place
        # should reconstruct tree while still preserving counts...
        # this might be a bad idea for large state spaces
        for pair in self.tree_subs_[::-1]:
            self.root.parent = pair[0]
            self.root = pair[0]
        self.tree_subs_ = []

    def reset_tree(self):
        print("Resetting tree")
        self.root = TreeNode(None)
        self.tree_subs_ = []

if __name__ == "__main__":
    import time
    cm = CountingManager()
    mcts_random = np.random.RandomState(1110)
    mcts = MCTS(cm, n_playout=100, random_state=mcts_random)
    n_games = 20
    all_game_timings = []
    all_game_steps = []
    for reset in [False, True]:
        game_timings = []
        game_steps = []
        for g in range(n_games):
            state = mcts.state_manager.get_init_state()
            winner, end = mcts.state_manager.is_finished(state)
            states = [state]
            start_time = time.time()
            steps = 0
            while True:
                if not end:
                    if g >= n_games - 5:
                        a, ap = mcts.get_action(state)
                    else:
                        if steps < 10:
                            a, ap = mcts.sample_action(state, temp=1., add_noise=True)
                        else:
                            a, ap = mcts.sample_action(state, temp=1E-3, add_noise=False)
                        if steps > 5000:
                            print("Game hard terminated after 5k steps")
                            break
                    mcts.update_tree_root(a)
                    state = mcts.state_manager.get_next_state(state, a)
                    states.append(state)
                    winner, end = mcts.state_manager.is_finished(state)
                    steps += 1
                    print("step {}, state {}".format(steps, state))
                else:
                    end_time = time.time()
                    game_timings.append(end_time - start_time)
                    game_steps.append(steps)
                    if reset:
                        mcts.reset_tree()
                    else:
                        mcts.reconstruct_tree()
                    print("Game {} finished in {} steps".format(g + 1, len(states)))
                    break
        all_game_timings.append(game_timings)
        all_game_steps.append(game_steps)
    from IPython import embed; embed(); raise ValueError()
