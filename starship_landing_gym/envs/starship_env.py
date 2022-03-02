import gym
from gym import spaces
import numpy as np

# TODO: check that _private func and var are consistent


class StarshipEnv(gym.GoalEnv):
    """
    The aim of this env is to land a Starship rocket.

    ### Action Space
    0: thruster -1 no thrust, 1 full thrust
    1: thrust angle -1 left, 1 right
    """

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 30}

    def __init__(self, dt=0.04, drop_h=1000, width=400,
                 random_goal=True, random_init_state=True,
                 augment_obs=False, random_constants=False,
                 reward_args=dict(
                    distance_scale=-1/120,
                    distance_weights=[1, 0, 1, 0, 1, 0, 0],
                    crash_scale=-1.0,
                    success_scale=+0.0,
                    step_scale=-0.0,
                 )):

        self.dt = dt
        self.drop_h = drop_h
        self.width = width
        self.random_goal = random_goal
        self.random_init_state = random_init_state
        self.augment_obs = augment_obs
        self.random_constants = random_constants
        self.reward_args = reward_args

        self.dyn = StarshipDynamics()

        # Define action space
        max_act = np.array([1, 1])
        self.action_space = spaces.Box(low=-max_act, high=max_act)

        # Define observation space

        # x, x_dot, y, y_dot, cos th, sin th, th_dot
        self.max_goal = np.array([self.width, 200, self.drop_h, 200,
                                  1, 1, 20*np.pi*2])
        norm_max_goal = self._normalize_obs(self.max_goal)

        self.min_goal = np.array([-self.width, -200, 0, -200,
                                  -1, -1, -20*np.pi*2])
        norm_min_goal = self._normalize_obs(self.min_goal)

        # create a state space to sample random init state
        max_init_space = np.array([self.width/2, 100, self.drop_h,
                                   -60, np.pi*2, np.pi/5])
        min_init_space = np.array([-self.width/2, -100, self.drop_h-300,
                                   -80, -np.pi*2, -np.pi/5])

        self.state_space = spaces.Box(low=min_init_space, high=max_init_space)

        if self.augment_obs:
            # TODO put the right values for the maximum distance.
            # is 2x max_obs?
            max_obs_distance = np.array([np.inf]*self.max_goal.shape[0])
            max_obs = np.concatenate((norm_max_goal, max_obs_distance))
            min_obs = np.concatenate((norm_min_goal, -max_obs_distance))
        else:
            max_obs = norm_max_goal
            min_obs = norm_min_goal

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=min_obs, high=max_obs),
            "achieved_goal": spaces.Box(low=norm_min_goal, high=norm_max_goal),
            "desired_goal": spaces.Box(low=norm_min_goal, high=norm_max_goal)}
        )

        # how far off can the agent be
        self.tolerances = np.array([50, 10, 50, 10, 0.3, np.inf, np.inf])

        self.renderer = None

    def _is_success(self, achieved_goal: np.ndarray,
                    desired_goal: np.ndarray) -> bool:

        tolerances = self._normalize_obs(self.tolerances)
        tolerances = tolerances/2
        upper_goal_limit = desired_goal + tolerances
        lower_goal_limit = desired_goal - tolerances
        achieved_in_tolerances = (achieved_goal > lower_goal_limit) \
            & (achieved_goal < upper_goal_limit)
        goal_achieved = np.sum(achieved_in_tolerances, axis=-1)
        goal_achieved = (goal_achieved == achieved_goal.shape[-1])

        return goal_achieved

    def compute_reward(self, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, info: dict):
        """
            Modular reward.
        """
        rwd_args = self.reward_args

        reward = 0
        is_success = self._is_success(achieved_goal, desired_goal)
        not_success = np.invert(is_success)

        # if we're only processing one reward make it an array
        if achieved_goal.shape[0] == achieved_goal.shape[-1]:
            a_goal = np.expand_dims(achieved_goal, axis=0)
        else:
            a_goal = achieved_goal

        crashed = np.array([self._crashed(g) for g in a_goal])

        reward_w = np.array(rwd_args["distance_weights"])
        # Distance Penalty
        distance = np.power(
            np.dot(np.abs(achieved_goal - desired_goal), reward_w),
            0.5)

        reward += distance * rwd_args["distance_scale"]

        # Crash Penalty
        # only penalize crash if not success
        if rwd_args["crash_scale"] != 0.0:
            reward += crashed * rwd_args["crash_scale"] * not_success

        # Success Reward
        reward += rwd_args["success_scale"] * is_success

        # Step Penalty: substract a fixed number at each step
        reward += rwd_args["step_scale"] * not_success

        # if only one reward. only needed for crash computation
        if achieved_goal.shape[0] == achieved_goal.shape[-1] \
                and rwd_args["crash_scale"] != 0.0:

            reward = reward[0]

        return reward

    def _crashed(self, obs):
        x, _, y, _, cos_th, _, _ = obs*self.max_goal  # de-normalize
        touching_ground = (y - abs(cos_th)*self.dyn.length/2) <= 0
        out_of_scope = abs(x) > self.width

        return bool(touching_ground or out_of_scope)

    def step(self, a):
        a = self._norm_thrust(a)
        self._state = self._update_state(a)

        obs = {}
        curr_obs = self._get_obs()

        if self.augment_obs:  # augment obs w/ distance to goal
            distance_to_goal = self.goal - curr_obs
            aug_obs = np.concatenate((curr_obs, distance_to_goal))

        obs["observation"] = aug_obs if self.augment_obs else curr_obs
        obs["desired_goal"] = self.goal
        obs["achieved_goal"] = curr_obs

        if self.renderer is not None:
            self.renderer.update(self._state, a)

        info = {"is_success": bool(self._is_success(
            obs["achieved_goal"],
            obs["desired_goal"]
        ))}

        rwd = self.compute_reward(obs["achieved_goal"],
                                  obs["desired_goal"], info)

        done = info["is_success"] or self._crashed(curr_obs)

        return obs, rwd, done, info

    def _normalize_obs(self, obs):
        return obs/self.max_goal

    def _norm_thrust(self, act):
        thrust = act[0]
        thrust = (thrust+1)/2
        return np.array([thrust, act[1]])

    def _get_obs(self):
        x, x_dot, y, y_dot, th, th_dot = self._state
        obs = np.array([x, x_dot, y, y_dot, np.cos(th), np.sin(th), th_dot])
        return self._normalize_obs(obs)

    def _init_state(self, random=True):
        self._state = np.array([
            0,  # start x pos
            0,  # start x speed
            self.drop_h,  # start y pos
            -80,  # start y speed
            -np.pi/2,  # start theta
            0,  # start theta speeed
        ]) if not random else self.state_space.sample()

    def _init_goal(self, random=True):
        max_x = self.width/2
        min_x = -max_x

        goal = np.array([
            np.random.randint(min_x, max_x) if random else 0,  # start x pos
            0,
            self.dyn.length/2,
            0,
            -1,
            0,
            0
        ])
        self.goal = self._normalize_obs(goal)
        self.raw_goal = goal

    def _update_state(self, a):
        x_dotdot, y_dotdot, theta_dotdot = \
            self.dyn(self._state, a)

        # update speeds from accels
        self._state[1] += self.dt * x_dotdot  # x_dot
        self._state[3] += self.dt * y_dotdot  # y_dot
        self._state[5] += self.dt * theta_dotdot  # th_dot

        # update pos from speeds
        self._state[0] += self.dt * self._state[1]  # x
        self._state[2] += self.dt * self._state[3]  # y
        self._state[4] += self.dt * self._state[5]  # th

        return self._state

    def render(self, mode="rgb_array"):
        if self.renderer is None:
            self.renderer = StarshipRenderer(self.width*2, self.drop_h)
            self.renderer.reset(self.dyn, self.tolerances, self.raw_goal)
        return self.renderer.render(return_rgb_array=mode == "rgb_array")

    def reset(self):
        self._init_state(self.random_init_state)
        self._init_goal(self.random_goal)
        self.dyn._init_constants(self.random_constants)
        if self.renderer is not None:
            self.renderer.reset(self.dyn, self.tolerances, self.raw_goal)
        obs, _, _, _ = self.step(np.zeros_like(self.action_space.low))
        return obs


class StarshipDynamics:
    def __init__(self):
        self._init_constants()

    def _random_variation(self, variable, pct_variation):
        """
        Returns a random % of variable between
        -pct_variation/2 and +pct_variation/2
        """

        return variable * (np.random.rand() * 2 - 1) * pct_variation/2

    def _init_constants(self, randomize=False):
        self.g = 9.8
        self.g += self._random_variation(self.g, pct_variation=0.2) * randomize

        self.m = 100000  # kg
        self.m += self._random_variation(self.m, pct_variation=0.2) * randomize

        self.max_thrust = 1 * 2210 * 1000  # kN
        self.max_thrust += self._random_variation(
            self.max_thrust, pct_variation=0.1) * randomize

        self.length = 50  # m
        self.length += self._random_variation(
            self.length, pct_variation=0.1) * randomize

        self.width = 10
        self.width += self._random_variation(
            self.width, pct_variation=0.1) * randomize

        # Inertia for a uniform density rod
        self.inertia = (1/12) * self.m * self.length**2

        self.max_gimble = np.deg2rad(20)

    def __call__(self, s, u):
        _, _, _, _, theta, _ = s
        thrust, thrust_angle = u[0], u[1]

        assert thrust >= 0 and thrust <= 1
        assert thrust_angle >= -1 and thrust_angle <= 1

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

        return [x_dotdot, y_dotdot, theta_dotdot]


class StarshipRenderer:

    def __init__(self, width, height):
        # import fails if no screen so need to import here
        from gym.envs.classic_control import rendering

        self.width = width
        self.height = height
        self.rendering = rendering
        self.viewer = None

    def reset(self, dyn, tolerances, raw_goal):
        if self.viewer is not None:
            self.close()
        self.dyn = dyn
        self.viewer = self.rendering.Viewer(self.width, self.height)
        self.transforms = {
            "ship": self.rendering.Transform(),
            "flame": self.rendering.Transform(translation=(0, 0)),
            "pad": self.rendering.Transform(),
        }

        x_tolerance, _, _, _, _, _, _ = tolerances
        self._init_pad(x_tolerance, 10)
        self._init_flame(5, 15)
        self._init_ship(dyn.width, dyn.length)
        self._update_goal(raw_goal)

    def _make_rectangle(self, width, height):
        lef, rig, top, bot = (
            -width / 2,
            width / 2,
            height / 2,
            -height / 2,
        )
        rect = self.rendering.FilledPolygon([(lef, bot), (lef, top),
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

    def _update_goal(self, goal):
        goal_x, _, _, _, _, _, _ = goal
        self.transforms["pad"].set_translation(goal_x+self.width/2, 0)

    def render(self, *args, **kwargs):
        """Forwards the call to the underlying Viewer instance"""
        return self.viewer.render(*args, **kwargs)

    def close(self):
        """Closes the underlying Viewer instance"""
        self.viewer.close()
