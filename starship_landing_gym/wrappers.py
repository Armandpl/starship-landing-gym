import gym
from gym.spaces import Box, Dict
import numpy as np


class HistoryWrapper(gym.Wrapper):
    """
    Track history of observations for given amount of steps
    Initial steps are zero-filled
    """
    def __init__(self, env, steps):
        super(HistoryWrapper, self).__init__(env)
        self.steps = steps

        # concat obs with action
        self.step_low = np.concatenate(
            [self.observation_space["observation"].low,
             self.action_space.low])
        self.step_high = np.concatenate(
            [self.observation_space["observation"].high,
             self.action_space.high])

        # stack for each step
        obs_low = np.tile(self.step_low, (self.steps, 1))
        obs_high = np.tile(self.step_high, (self.steps, 1))

        self.observation_space = Dict(
            {"observation": Box(low=obs_low, high=obs_high),
             "achieved_goal": self.observation_space["achieved_goal"],
             "desired_goal": self.observation_space["desired_goal"]})

        self.history = self._init_history()

    def _init_history(self):
        # TODO check if it would make sense to use a dequeue instead?
        return [np.zeros_like(self.step_low) for _ in range(self.steps)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # remove step obs at the left of the buffer
        self.history.pop(0)

        # concat current obs with action
        observation = np.concatenate([obs["observation"], action])
        self.history.append(observation)
        obs["observation"] = np.array(self.history)  # update obs with history

        return obs, reward, done, info

    def reset(self):
        self.history = self._init_history()  # init empty history
        self.history.pop(0)  # make room for first obs
        obs = self.env.reset()
        zero_act = np.zeros_like(self.env.action_space.low)
        observation = np.concatenate([obs["observation"], zero_act])
        self.history.append(observation)
        obs["observation"] = np.array(self.history)

        return obs
