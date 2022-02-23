import gym
import numpy as np
import starship_landing_gym  # noqa F420
from stable_baselines3.common.env_checker import check_env
from starship_landing_gym import __version__


def test_version():
    assert __version__ == '0.1.0'


def test_env():
    env = gym.make("StarshipLanding-v0")
    check_env(env)


def test_is_success():
    env = gym.make("StarshipLanding-v0")
    goal = np.ones_like(env.observation_space["desired_goal"].low)
    achieved_goal = np.ones_like(env.observation_space["achieved_goal"].low)
    not_achieved_goal = np.zeros_like(
        env.observation_space["achieved_goal"].low)

    assert bool(env._is_success(achieved_goal, goal)) is True
    assert bool(env._is_success(not_achieved_goal, goal)) is False


def test_compute_reward():
    # TODO once reward dialed in check it computes the right values
    # check with 1D and 2D arrays
    pass
