import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np


class StarshipEnv(gym.GoalEnv):
    """
    The aim of this env is to land a Starship rocket.

    ### Action Space
    0: thruster -1 no thrust, 1 full thrust
    1: thrust angle -1 left, 1 right # TODO ou l'inverse il faut check
    """

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 30}

    def __init__(self, dt=0.04, drop_h=1000, width=400):
        self.dt = dt
        self.drop_h = drop_h
        self.width = width

        self.dyn = StarshipDynamics()
        self.renderer = StarshipRenderer(self.dyn, self.width*2, self.drop_h)

        self.norm_obs = np.array([self.width, 10, self.drop_h, 10, 1, 1, 10])

        max_act = np.array([1, 1])
        self.action_space = spaces.Box(low=-max_act, high=max_act)

        # TODO put the right values there
        max_obs = np.array([np.inf]*7)
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(low=-max_obs, high=max_obs),
             "achieved_goal": spaces.Box(low=-max_obs, high=max_obs),
             "desired_goal": spaces.Box(low=-max_obs, high=max_obs)})

        # desired observation when the task is solved
        goal = np.array([0, 0, self.dyn.length/2, 0, -1, 0, 0])
        self.goal = self._normalize_obs(goal)

    def compute_reward(self, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, info: dict,
                       p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded
        We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward.
        Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        # reward_w = [1, 0.3, 0, 0, 0.02, 0.02]
        reward_w = [1, 1, 1, 1, 1, 1, 1]
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal),
                         np.array(reward_w)), p)

    def step(self, a):
        # TODO: check better way of doing this
        thrust = a[0]
        thrust = (thrust+1)/2
        # thrust = 0.4 + thrust*(1-0.4)  # enforce a minimum thrust of 0.4
        a = np.array([thrust, a[1]])

        self._state = self._update_state(a)
        info = {}

        obs = {}
        curr_obs = self._get_obs()
        obs["observation"] = curr_obs
        obs["desired_goal"] = self.goal
        obs["achieved_goal"] = curr_obs

        done = False
        x, _, y, _, _, _ = self._state
        if y-self.dyn.length <= 0:  # if at ground level
            done = True
        if abs(x) > self.width:  # if rocket outside of scope
            done = True

        self.renderer.update(self._state, a)

        rwd = self.compute_reward(obs["achieved_goal"],
                                  obs["desired_goal"], info)

        return obs, rwd, done, info

    def _normalize_obs(self, obs):
        return obs/self.norm_obs

    def _get_obs(self):
        x, x_dot, y, y_dot, th, th_dot = self._state
        obs = np.array([x, x_dot, y, y_dot, np.cos(th), np.sin(th), th_dot])
        return self._normalize_obs(obs)

    def _init_state(self):
        self._state = np.array([
            np.random.randint(-self.width/2, self.width/2),  # start x pos
            0,  # start x speed
            self.drop_h,  # start y pos
            -80,  # start y speed
            np.random.rand()*np.pi*2,  # start theta
            0,  # start theta speeed
        ])

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

        return self._state

    def render(self, mode="rgb_array"):
        return self.renderer.render(return_rgb_array=mode == "rgb_array")

    def reset(self):
        self._init_state()
        return self.step(np.zeros(self.action_space.shape))[0]


class StarshipDynamics:
    def __init__(self):
        self.g = 9.8
        self.m = 100000  # kg
        # self.min_thrust = 880 * 1000  # N  # TODO delete if not useful
        self.max_thrust = 1 * 2210 * 1000  # kN

        self.length = 50  # m
        self.width = 10

        # Inertia for a uniform density rod
        self.inertia = (1/12) * self.m * self.length**2

        self.max_gimble = np.deg2rad(20)
        # self.min_gimble = -self.max_gimble  # TODO delete if not useful

    def __call__(self, s, u):
        """
            thrust should be between 0 and 1  TODO replace by assert?
            thrust angle should be between -1 and 1
        """
        x, x_dot, y, y_dot, theta, theta_dot = s
        thrust, thrust_angle = u[0], u[1]

        # TODO: check if that's useful
        # thrust = self.min_thrust + thrust*(self.max_thrust-self.min_thrust)
        thrust = thrust*self.max_thrust
        thrust_angle = thrust_angle * self.max_gimble

        # Horizontal force
        F_x = thrust * np.sin(thrust_angle + theta)
        x_dotdot = (F_x) / self.m

        # Vertical force
        F_y = -thrust * np.cos(thrust_angle + theta)
        y_dotdot = (F_y) / self.m - self.g

        # Torque
        T = -self.length/2 * thrust * np.sin(thrust_angle)
        theta_dotdot = T / self.inertia

        return [x_dot, x_dotdot, y_dot, y_dotdot, theta_dot, theta_dotdot]


class StarshipRenderer:

    def __init__(self, dyn, width, height):
        self.dyn = dyn
        self.width = width
        self.height = height

        self.viewer = rendering.Viewer(width, height)
        self.transforms = {
            "ship": rendering.Transform(),
            "flame": rendering.Transform(translation=(0, 0)),
            "pad": rendering.Transform(),
        }

        self._init_pad(100, 10)
        self._init_flame(5, 15)
        self._init_ship(dyn.width, dyn.length)
        self.transforms["pad"].set_translation(width/2, 0)

    def _make_rectangle(self, width, height):
        lef, rig, top, bot = (
            -width / 2,
            width / 2,
            height / 2,
            -height / 2,
        )
        rect = rendering.FilledPolygon([(lef, bot), (lef, top),
                                        (rig, top), (rig, bot)])

        return rect

    def _init_ship(self, width, height):
        ship = self._make_rectangle(width, height)
        ship.add_attr(self.transforms["ship"])
        ship.set_color(164/255, 210/255, 226/255)
        self.viewer.add_geom(ship)

    def _init_pad(self, width, height):
        pad = self._make_rectangle(width, height)
        pad.set_color(0, 0, 0)
        pad.add_attr(self.transforms["pad"])
        self.viewer.add_geom(pad)

    def _init_flame(self, width, height):
        flame = self._make_rectangle(width, height)
        # TODO can i pass rgb values in 255 format. maybe edit color propertie?
        flame.set_color(243/255, 89/255, 63/255)
        flame.add_attr(self.transforms["flame"])
        flame.add_attr(self.transforms["ship"])
        self.viewer.add_geom(flame)

    def update(self, state, action):
        """Updates the positions of the objects on screen"""
        thrust, thrust_angle = action
        ship_x, ship_y, ship_th = state[0], state[2], state[4]

        self.transforms["ship"].set_translation(ship_x+self.width/2, ship_y)
        self.transforms["ship"].set_rotation(ship_th)
        self.transforms["flame"].set_rotation(thrust_angle)
        self.transforms["flame"].set_translation(0, self.dyn.length/2)
        self.transforms["flame"].set_scale(1, thrust)

    def render(self, *args, **kwargs):
        """Forwards the call to the underlying Viewer instance"""
        return self.viewer.render(*args, **kwargs)

    def close(self):
        """Closes the underlying Viewer instance"""
        self.viewer.close()
