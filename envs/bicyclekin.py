import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
# env registering https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai

# class BicycleKin(gym.Env):
#     """
#     Description:
            
#     Time:
#             discretizing_time: 
#             sampling_time: 
            
#     Observation*:
#             Type: Box(3,)
#             Num     Observation       Min     Max
#             0       x                 -inf       inf
#             1       y                 -inf       inf
#             2       theta              -pi       +pi
            
#     Actions*:
#             Type: Box(2,), min=0 max=2
#             Num     Action     Min     Max
#             0       v          -inf       inf
#             1       w      -pi         pi      
            
#     Reward: is affected by 6 things
#         1. per each step
#         2. Hitting an obstacle
#         3. reaching the goal
#         4. Reaching the end and not reaching the goal?
#         5. input v
#         6. input w (or w)
#     weight:

#     Episode Termination:

#     """


#     metadata = {'render.modes': ['console']}


#     def __init__(
#         self, 
#         length: float = 0.256, 
#         sampling_time: float = 0.02, 
#         discretization_steps: int = 10,
#         obstacle_loc = [[4.0, 0.0], [7.0, 2.0]],
#         obstacle_radius = [1.0, 1.0],
#         goal_pos = [10., 0.],
#         init_pos = [0., 0.],
#         rand_initial_pos = False,
#         v_bound = [0.1, 1.],
#         w_bound = [-0.5, 0.5],
#         total_steps: int = 50000,
#         reward_weights = [-0.01, -1., -0.01, -0.5, -0.001, -0.001]
#         ):
#         super(BicycleKin, self).__init__()
#         self.l = length
#         self.sampling_time = sampling_time
#         self.discretization_steps = discretization_steps
#         self.disc_time = self.sampling_time/self.discretization_steps
#         self.obstacle_loc = obstacle_loc
#         self.obstacle_radius  = obstacle_radius
#         self.goal_pos = goal_pos
#         self.total_steps = total_steps
#         self.init_pos = init_pos
#         self.rand_initial_pos = rand_initial_pos
#         self.v_bound = v_bound
#         self.w_bound = w_bound
#         self.reward_weights = reward_weights

#         high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
#         # print(high)
#         # low_obs = np.full(shape = (3,), fill_value = -np.inf, dtype=np.float32)
#         # high_obs = np.full(shape = (3,), fill_value = np.inf, dtype=np.float32)
#         # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
#         self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
#         self.action_low = np.array([self.v_bound[0], self.w_bound[0]], dtype=np.float32)
#         self.action_high = np.array([self.v_bound[1], self.w_bound[1]], dtype=np.float32)
#         self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.], dtype=np.float32))
#         self.time = 0
#         # if self.rand_initial_pos:
#         #     self.state = np.array([np.random.uniform(0.,20.) , np.random.uniform(-10, 10.), np.random.uniform(-0.2, 0.3)], dtype=np.float32)
#         # else:
#         #     self.state = np.array([0., 0., 0.], dtype=np.float32)
#         self.reset()

#     def action_scaling(self, action):
#         a_v, b_v = self.v_bound
#         a_w, b_w = self.w_bound
        
#         v_s, w_s = action[0], action[1]
#         v = a_v + 0.5*(v_s+1.)*(b_v-a_v)
#         w = a_w + 0.5*(w_s+1.)*(b_w-a_w)
#         return np.array([v, w], dtype=np.float32)

#     def step(self, action):
#         self.time += 1

#         action = self.action_scaling(action)

#         for _ in range(self.discretization_steps):
#             self.mini_step(action)

#         reward, done = self.reward(action)
#         return self.state, reward, done, {}

#     def mini_step(self, action):
#         x, y, th = self.state
#         v, w = action[0], action[1]
#         x += v * math.cos(th) * self.disc_time
#         y += v * math.sin(th) * self.disc_time
#         th += (v*math.tan(w)/self.l) * self.disc_time
#         self.state = np.array([x, y, th], dtype=np.float32)

#     def reward(self, action):
#         x, y, th = self.state
#         v, w = action

#         done = False
#         reward = self.reward_weights[1]

#         # for r, pos in zip(self.obstacle_radius, self.obstacle_loc):
#         #     if (pos[0]-x)**2 + (pos[1]-y)**2<=r**2:
#         #         reward += self.reward_weights[2]
#         #         done = True
#         #         break
#         if self._chk_obstacle():
#             done = True
#             reward += self.reward_weights[2]

#         dist_square = (self.goal_pos[0]-x)**2 + (self.goal_pos[1]-y)**2
#         reward += self.reward_weights[3] * dist_square
#         if dist_square<=0.01:
#             done = True
#         if self.time == self.total_steps or np.sqrt(dist_square)>20.:
#             done = True
#             reward += self.reward_weights[4]
#         reward +=  self.reward_weights[4]*(v**2)
#         reward +=  self.reward_weights[5]*(w**2)
#         return reward, done

#     def _reset_state(self):
#         return np.array([np.random.uniform(0.,10.) , np.random.uniform(-5, 5.), np.random.uniform(-0.3, 0.3)], dtype=np.float32)
    
#     def _chk_obstacle(self):
#         obstacle = False
#         x, y, th = self.state
#         for r, pos in zip(self.obstacle_radius, self.obstacle_loc):
#             if (pos[0]-x)**2 + (pos[1]-y)**2<=r**2:
#                 obstacle =  True
#                 break
#         return obstacle
    
#     def reset(self):
#         self.time = 0
#         if self.rand_initial_pos:
#             obstacle = True
#             while obstacle:
#                 self.state = self._reset_state()
#                 obstacle = self._chk_obstacle()
#         else:
#             self.state = np.array([0., 0., 0.], dtype=np.float32)
#         return self.state
    
#     def render(self, mode='console'):
#         if mode != 'console':
#             raise NotImplementedError()
#         print("not implemented")
    
#     def close(self):
#         pass



class BicycleKin(gym.Env):
    """
    Description:
            
    Time:
            discretizing_time: 
            sampling_time: 
            
    Observation*:
            Type: Box(3,)
            Num     Observation       Min     Max
            0       x                 -inf       inf
            1       y                 -inf       inf
            2       theta              -pi       +pi
            
    Actions*:
            Type: Box(2,), min=0 max=2
            Num     Action     Min     Max
            0       v          -inf       inf
            1       w      -pi         pi      
            
    Reward: is affected by 6 things
        1. per each step
        2. Hitting an obstacle
        3. reaching the goal
        4. Reaching the end and not reaching the goal?
        5. input v
        6. input w (or w)
    weight:

    Episode Termination:

    """


    metadata = {'render.modes': ['console']}


    def __init__(
        self, 
        length: float = 0.256, 
        sampling_time: float = 0.02, 
        discretization_steps: int = 10,
        obstacle_loc = [[4.0, 0.0], [7.0, 2.0]],
        obstacle_radius = [1.0, 1.0],
        goal_pos = [10., 0.],
        init_pos = [0., 0.],
        rand_initial_pos = True,
        v_bound = [0.1, 1.],
        w_bound = [-0.5, 0.5],
        total_steps: int = 10_000,
        reward_weights = [-0.0, -1., -0.1, -0.0, -0.000, -0.000]
        ):
        super(BicycleKin, self).__init__()
        self.l = length
        self.sampling_time = sampling_time
        self.discretization_steps = discretization_steps
        self.disc_time = self.sampling_time/self.discretization_steps
        self.obstacle_loc = obstacle_loc
        self.obstacle_radius  = obstacle_radius
        self.goal_pos = goal_pos
        self.total_steps = total_steps
        self.init_pos = init_pos
        self.rand_initial_pos = rand_initial_pos
        self.v_bound = v_bound
        self.w_bound = w_bound
        self.reward_weights = reward_weights

        high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        # print(high)
        # low_obs = np.full(shape = (3,), fill_value = -np.inf, dtype=np.float32)
        # high_obs = np.full(shape = (3,), fill_value = np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_low = np.array([self.v_bound[0], self.w_bound[0]], dtype=np.float32)
        self.action_high = np.array([self.v_bound[1], self.w_bound[1]], dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.], dtype=np.float32))
        self.time = 0
        # if self.rand_initial_pos:
        #     self.state = np.array([np.random.uniform(0.,20.) , np.random.uniform(-10, 10.), np.random.uniform(-0.2, 0.3)], dtype=np.float32)
        # else:
        #     self.state = np.array([0., 0., 0.], dtype=np.float32)
        self.reset()

    def action_scaling(self, action):
        a_v, b_v = self.v_bound
        a_w, b_w = self.w_bound
        
        v_s, w_s = action[0], action[1]
        v = a_v + 0.5*(v_s+1.)*(b_v-a_v)
        w = a_w + 0.5*(w_s+1.)*(b_w-a_w)
        return np.array([v, w], dtype=np.float32)

    def step(self, action):
        self.old_state = self.state
        self.time += 1

        action = self.action_scaling(action)

        for _ in range(self.discretization_steps):
            self.mini_step(action)

        reward, done = self.reward(action)
        if self.time==self.total_steps:
            done =  True
        return self.state, reward, done, {}

    def mini_step(self, action):
        x, y, th = self.state
        v, w = action[0], action[1]
        x += v * math.cos(th) * self.disc_time
        y += v * math.sin(th) * self.disc_time
        th += (v*math.tan(w)/self.l) * self.disc_time
        self.state = np.array([x, y, th], dtype=np.float32)

    def reward(self, action):
        x, y, th = self.state
        x_old, y_old, th = self.old_state
        v, w = action

        done = False
        reward = 0.
        # reward = self.reward_weights[0]

        # if self._chk_obstacle():
        #     done = True
        #     reward += self.reward_weights[1] * (self.total_steps-self.time)

        dist_new = np.sqrt((self.goal_pos[0]-x)**2 + (self.goal_pos[1]-y)**2)
        dist_old = np.sqrt((self.goal_pos[0]-x_old)**2 + (self.goal_pos[1]-y_old)**2)
        dist_net = dist_old - dist_new

        reward += dist_net
        if dist_new<=0.1:
            reward += 100.
            done = True
        if self.time == self.total_steps:
            reward += -1. * dist_new
            done = True
        reward +=  self.reward_weights[4]*(v**2)
        reward +=  self.reward_weights[5]*(w**2)
        return reward, done

    def _reset_state(self):
        return np.array([np.random.uniform(0.,10.) , np.random.uniform(-5, 5.), np.random.uniform(-0.3, 0.3)], dtype=np.float32)
    
    def _chk_obstacle(self):
        obstacle = False
        x, y, th = self.state
        for r, pos in zip(self.obstacle_radius, self.obstacle_loc):
            if (pos[0]-x)**2 + (pos[1]-y)**2<=r**2:
                obstacle =  True
                break
        return obstacle
    
    def reset(self):
        self.time = 0
        if self.rand_initial_pos:
            obstacle = True
            while obstacle:
                self.state = self._reset_state()
                obstacle = self._chk_obstacle()
        else:
            self.state = np.array([0., 0., 0.], dtype=np.float32)
        return self.state
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("not implemented")
    
    def close(self):
        pass
