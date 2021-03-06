# Starship Landing Gym [![tests](https://github.com/Armandpl/starship-landing-gym/actions/workflows/tests.yml/badge.svg)](https://github.com/Armandpl/starship-landing-gym/actions/workflows/tests.yml)
A Gym env for propulsive rocket landing. 

<p align="center">
  <img width="400" height="500" src="https://raw.githubusercontent.com/Armandpl/starship-landing-gym/master/images/landing.gif">
  <br/>
  <i> Successfull Rocket Landing </i>
</p>

The goal is to bring the rocket above the landing pad with a speed inferior to 5m/s.  

This is inspired by and based on Thomas Godden's ["Starship Landing Trajectory Optimization" blog post.](http://thomasgodden.com/starship-trajopt.html).

I made a video about the issues I ran into while solving this env: [Je fais atterrir une fusée SpaceX avec du RL !](https://www.youtube.com/watch?v=9xX14NkrbRU) It's in french and I wrote english subtitles.

## Installation

`pip install starship-landing-gym`

## Usage

```python
import gym
import starship_landing_gym

env = gym.make("StarshipLanding-v0")

done = False
env.reset()
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```
