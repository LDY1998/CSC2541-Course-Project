import gym
from gym import register

def create_env(random_noise=5e-3, max_steps=1000):
    register(id="Rand-Hopper-v2", entry_point="gym.envs.mujoco:Hopper-v2", kwargs={"reset_noise_scale": random_noise, "max_episode_steps": max_steps})
    env = gym.make("Rand-Hopper-v2")
    return env


# only for testing
if __name__ == '__main__':
    # env = create_env()
    pass