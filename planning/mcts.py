# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

import math
import time
import numpy as np
from copy import deepcopy
import logging
import os
import sys
import pickle


class PTreeNode(object):
    def __init__(self, parent, prior_prob, name='unk'):
        self.name = name
        self.parent = parent
        self.Q_ = 0.0
        self.P_ = float(prior_prob)
        # action -> tree node
        self.children_ = {}
        # JRH TIRED AND LIKELY HAVE A BUG
        #self.n_visits_ = 0
        self.n_visits_ = 1e-6
        self.past_actions = []
        self.n_wins = 0

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                child_name = (self.name[0],action)
                self.children_[action] = PTreeNode(self, prior_prob=prob, name=child_name)

    def is_leaf(self):
        return self.children_ == {}

    def is_root(self):
        return self.parent is None

    def _update(self, value):
        self.n_visits_ += 1
        self.Q_ += (value-self.Q_)/float(self.n_visits_)

    def update(self, value):
        if self.parent != None:
            self.parent.update(value)
        self._update(value)

    def get_value(self, c_puct):
        self.U_ = c_puct * self.P_ * np.sqrt(float(self.parent.n_visits_)) / float(1+self.n_visits_)
        return self.Q_ + self.U_

    def get_best(self, c_puct):
        best = max(self.children_.iteritems(), key=lambda x: x[1].get_value(c_puct))
        return best



