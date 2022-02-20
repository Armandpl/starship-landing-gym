import gym
from gym.wrappers import TimeLimit
import pyvirtualdisplay
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from wandb.integration.sb3 import WandbCallback
import wandb


import starship_landing_gym   # noqa F420

if __name__ == "__main__":
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 5000000,
        "env_name": "StarshipLanding-v0",
        # Available strategies (cf paper): future, final, episode
        "goal_selection_strategy": 'episode',
        "online_sampling": False,
        "max_episode_length": 400,
        "batch_size": 256,
        "her_k": 4
    }

    run = wandb.init(
        project="starship-landing",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    def make_env():
        env = gym.make(config["env_name"])
        check_env(env)
        env = TimeLimit(env, config["max_episode_length"])
        env = TimeFeatureWrapper(env, config["max_episode_length"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}",
                           record_video_trigger=lambda x: x % 4000 == 0,
                           video_length=400)

    model = SAC(
        config["policy_type"],
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=config["her_k"],
            goal_selection_strategy=config["goal_selection_strategy"],
            online_sampling=config["online_sampling"],
            max_episode_length=config["max_episode_length"],
            handle_timeout_termination=False
        ),
        batch_size=config["batch_size"],
        # buffer_size=int(1e6),
        # gamma=0.95, batch_size=1024, tau=0.05,
        # policy_kwargs=dict(net_arch=[512, 512, 512]),
        tensorboard_log=f"runs/{run.id}",
        verbose=1
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
