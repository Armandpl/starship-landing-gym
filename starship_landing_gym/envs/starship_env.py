import gym
from gym.spaces import Box
import numpy as np
from math import cos, sin


class StarshipEnv(gym.Env):

    def __init__(self, dt=0.04):

        max_act = np.array([1, 1])
        min_act = np.array([0, -1])
        self.action_space = Box(low=max_act, high=min_act)

        max_obs = np.array([np.inf]*6)
        self.observation_space = Box(low=-max_obs, high=max_obs)

        self.dt = dt

        self.dyn = StarshipDynamics()
        self._init_state()

        # max steps 400 TODO

    def _get_obs(self):
        x, x_dot, y, y_dot, th, th_dot = self._state
        # TODO normalize obs?
        return [x, x_dot, y, y_dot, th, th_dot]

    def _init_state(self):
        # x, x_dot, y, y_dot, th, th_dot
        self._state = np.array[0, 0, 1000, -80, -np.pi/2, 0]
        pass

    def _update_state(self, a):
        x_dot, x_dotdot, y_dot, y_dotdot, theta_dot, theta_dotdot = \
            self.dyn(self._state, a)

        # update speeds from accels
        self._state[1] += self.dt * x_dotdot
        self._state[3] += self.dt * y_dotdot
        self._state[5] += self.dt * theta_dotdot

        # update pos from speeds
        self._state[0] += self.dt * self._state[1]
        self._state[2] += self.dt * self._state[3]
        self._state[4] += self.dt * self._state[5]

        return np.copy(self._state)

    def render(self, mode="rgb_array"):
        pass

    def _rwd(self, state):
        # -1 each step, 200 if solved
        # solved = abs(th) < 10 deg
        # x = 0, y = 0
        # x, y and th speeds are slow

        return 0

    def step(self, a):
        self._state = self._update_state(a)
        rwd = self._rwd(self._state, a)
        obs = self._get_obs()
        done = False
        info = {}

        return obs, rwd, done, info

    def reset(self):
        self._init_state()
        return self.step(np.array([0, 0]))[0]


class StarshipDynamics:
    def __init__(self):
        self.g = 9.8
        self.m = 100000  # kg
        # self.min_thrust = 880 * 1000  # N
        self.max_thrust = 1 * 2210 * 1000  # kN

        self.length = 50  # m
        self.width = 10

        # Inertia for a uniform density rod
        self.inertia = (1/12) * self.m * self.length**2

        deg_to_rad = 0.01745329

        self.max_gimble = 20 * deg_to_rad
        # self.min_gimble = -self.max_gimble

    def __call__(self, s, u):
        x, x_dot, y, y_dot, theta, theta_dot = s
        thrust, thrust_angle = u[0], u[1]
        thrust_angle = thrust_angle * self.max_gimble

        # Horizontal force
        F_x = self.max_thrust * thrust * sin(thrust_angle + theta)
        x_dotdot = (F_x) / self.m

        # Vertical force
        F_y = self.max_thrust * thrust * cos(thrust_angle + theta)
        y_dotdot = (F_y) / self.m - self.g

        # Torque
        T = -self.length/2 * self.max_thrust * thrust * sin(thrust_angle)
        theta_dotdot = T / self.inertia

        return [x_dot, x_dotdot, y_dot, y_dotdot, theta_dot, theta_dotdot]
