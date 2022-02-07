__version__ = '0.1.0'
from gym.envs.registration import register

register(
    id='StarshipLanding-v0',
    entry_point='starship_landing_gym.envs:StarshipEnv',
)
