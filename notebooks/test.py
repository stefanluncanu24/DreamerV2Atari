import sys
import os
import gymnasium as gym
import ale_py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.atari import Atari

if __name__ == "__main__":
    env = gym.make("ALE/Pong-v5", render_mode="human")
    wrapped_env = Atari(env, action_repeat=4, frame_stack=4, sticky_actions=0.25)
    obs, info = wrapped_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("Episode finished. Total reward:", total_reward)
    wrapped_env.close()