import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class Atari(gym.Wrapper):
    def __init__(self, env, action_repeat=4, frame_stack=4, sticky_actions=0.25):
        super().__init__(env)
        self._action_repeat = action_repeat
        self._frame_stack = frame_stack
        self._sticky_actions = sticky_actions
        self._last_action = 0
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(frame_stack, 84, 84), dtype=np.uint8
        )
        self._frames = deque(maxlen=frame_stack)

    def _preprocess(self, frame):
        if frame is None:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._preprocess(obs)
        for _ in range(self._frame_stack):
            self._frames.append(processed_obs)
        return self._get_obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        if np.random.uniform() < self._sticky_actions:
            action = self._last_action
        self._last_action = action

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            processed_obs = self._preprocess(obs)
            self._frames.append(processed_obs)
            if terminated or truncated:
                break
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self._frames, dtype=np.uint8)
