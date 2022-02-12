import gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
import wandb


import starship_landing_gym   # noqa F420

if __name__ == "__main__":
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 1000000,
        "env_name": "StarshipLanding-v0",
        # Available strategies (cf paper): future, final, episode
        "goal_selection_strategy": 'future',
        "online_sampling": False,
        "max_episode_length": 400
    }

    run = wandb.init(
        project="starship-landing",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env():
        env = gym.make(config["env_name"])
        check_env(env)
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}",
                           record_video_trigger=lambda x: x % 2000 == 0,
                           video_length=400)

    model = SAC(
        config["policy_type"],
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=config["goal_selection_strategy"],
            online_sampling=config["online_sampling"],
            max_episode_length=config["max_episode_length"],
        ),
        verbose=1,
        tensorboard_log=f"runs/{run.id}"
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()
