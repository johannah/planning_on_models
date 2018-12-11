class FreewayEnv():
    def __init__(self, action_space, model_step_fn='equiv_model_step', ysize=input_ysize, xsize=input_xsize, frame_skip=4):
        self.model_step = eval('self.'+model_step_fn)
        # keys is [current state, past state, action]
        self.action_space = action_space
        self.state_traces = {}
        # TODO - these are not right
        self.frame_skip = frame_skip
        self.max_steps = np.int(18000/float(self.frame_skip))+self.frame_skip
        if self.frame_skip == 4:
            self.step_offset = 2
        self.action_offset = {0:0.0, 1:2, 2:-2}
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
        self.do_nothing_action = 0
        self.finished = False
        self.total_reward = 0
        self.penalty_stuck = np.int(25/float(self.frame_skip))

    def add_new_state(self, state_key):
        self.state_traces[state_key] = {}
        for a in self.action_space:
            self.state_traces[state_key][a] = {'q':0.0, 'cnt':0.0, 'diff':self.action_offset[a]}

    def add_state_trace(self, state_key, action, value, diff):
        # all should be ints
        if state_key not in self.state_traces:
            print("adding new state", state_key)
            print(action, value, diff)
            self.add_new_state(state_key)
        self.state_traces[state_key][action]['q'] += value
        self.state_traces[state_key][action]['diff'] += diff
        self.state_traces[state_key][action]['cnt'] += 1.0

    def get_state_trace(self, state_key, action):
        if state_key not in self.state_traces:
            self.add_new_state(state_key)
        adiff = self.state_traces[state_key][action]['diff']
        aq = self.state_traces[state_key][action]['q']
        acnt = self.state_traces[state_key][action]['cnt']

        v = aq/float(acnt) if acnt>0 else 0
        diff = adiff/float(acnt+1.0)
        #print("GET STATE", state_key, action, adiff, aq, acnt)
        return v, diff

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

    def step(self, agent_location, diff):
        ny = agent_location[0]
        nx = agent_location[1]
        ny = np.array([yy+diff for yy in ny])
        ny[ny>(self.ysize-1)] = self.ysize-1
        ny[ny<0] = 0
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

    def value_model_step(self, state, state_index, next_road, action, next_steps=[]):
        #self.observ, self.reward, self.finished, _ = model_env.step(action)
        # agent location is a tuple of y,x
        # if 'stunned' - stay still
        finished = False
        reward = 0.0
        loc, this_road = state
        this_small = get_local_ones_image(deepcopy(this_road), loc, ysize=input_ysize, xsize=input_xsize, buffer_size=args.buffer_size)
        s = make_state_key(this_small)
        #sa = np.array(s)
        #sa = sa.reshape(this_small.shape)
        _,diff = self.get_state_trace(s,action)
        al = deepcopy(loc)
        next_agent_location = self.step(al,diff)
        #if not len(next_steps):
        #next_step = 'free'
        #if action == 1:
        #    next_agent_location = self.step_up(al)
        #elif action == 2:
        #    next_agent_location = self.step_down(al)
        #else:
        #    next_agent_location = al
        ### this prob isnt right
        #next_steps = ['free']
        #nal = deepcopy(next_agent_location)
        #found_goal = self.check_for_goal(nal)
        #if found_goal == 1:
        #    reward = 1
        #    finished = True
        #    next_agent_location = base_chicken

        #next_state = [next_agent_location, next_road]

        #if not len(next_steps):
        #    next_step = 'free'
        #    if action == 1:
        #        next_agent_location = self.step_up(al)
        #    elif action == 2:
        #        next_agent_location = self.step_down(al)
        #    else:
        #        next_agent_location = al
        #else:
        #    next_step = next_steps.pop(0)
        #    if next_step == 'penalty':
        #        next_agent_location = self.step_down(al)
        #    if next_step == 'stuck':
        #        next_agent_location = al
        # this prob isnt right
        nal = deepcopy(next_agent_location)
        #if next_step is 'free':
        #    next_steps = []
        #    if self.determine_collisions(nal, next_road):
        #        # penalty
        #        for i in range(self.penalty_stuck):
        #            next_steps.append('penalty')
        #        for i in range(self.penalty_stuck):
        #            next_steps.append('stuck')
        #    else:
        found_goal = self.check_for_goal(nal)
        if found_goal== 1:
            reward = 1
            finished = True
            next_agent_location = base_chicken

        next_state = [next_agent_location, next_road]

        return next_state, reward, finished, next_steps

    def equiv_model_step(self, state, state_index, next_road, action, next_steps=[]):
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
                found_goal = self.check_for_goal(nal)
                if found_goal == 1:
                    reward = 1
                    finished = True
                    next_agent_location = base_chicken

        next_state = [next_agent_location, next_road]
        return next_state, reward, finished, next_steps


