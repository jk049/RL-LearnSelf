"""
open ai开源的gym环境封装，本文件包含部分常用的封装函数,用于DQN算法
"""
import cv2
import gym
import gym.spaces
import numpy as np
import collections
import random
from gym.wrappers import AtariPreprocessing


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(
                np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class NoopResetEnv(gym.Wrapper):
    """
    在 reset 时执行随机次数的 NOOP（action 0）。兼容不同 gym 版本的 reset/step 返回格式。
    """
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        # 默认把 NOOP 定义为动作 0；若 env 提供 action meanings 可做检查（非必须）
        try:
            am = env.unwrapped.get_action_meanings()
            if isinstance(am, (list, tuple)) and len(am) > 0 and am[0].upper() != 'NOOP':
                # 如果 action meanings 存在且第一个不是 NOOP，这里仍默认 0
                pass
        except Exception:
            pass
        self.noop_action = 0

    def _reset_env(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}
        return obs, info, isinstance(res, tuple)

    def _step_env(self, action):
        res = self.env.step(action)
        # step can be (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        if len(res) == 4:
            obs, reward, done, info = res
        else:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        return obs, reward, done, info, res

    def reset(self, **kwargs):
        obs, info, was_tuple = self._reset_env(**kwargs)
        noops = random.randint(1, self.noop_max)
        for _ in range(noops):
            obs_, reward, done, info, step_res = self._step_env(self.noop_action)
            obs = obs_  # 更新 obs
            if done:
                # If env ended, re-reset to get a valid starting state
                obs, info, was_tuple = self._reset_env(**kwargs)
        return (obs, info) if was_tuple else obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    This helps value-based agents to treat loss of life as terminal for learning,
    while not resetting the underlying emulator until the game is actually over.
    Compatible with older/newer gym APIs (step may return 4- or 5-tuple).
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def _get_lives(self, info):
        # Try various places where life count may appear
        if isinstance(info, dict):
            if 'lives' in info:
                return info['lives']
            if 'ale.lives' in info:
                return info['ale.lives']
        # Fallback to env.unwrapped attribute (Atari environments)
        try:
            return self.env.unwrapped.ale.lives()
        except Exception:
            try:
                return self.env.unwrapped.lives()
            except Exception:
                return None

    def step(self, action):
        res = self.env.step(action)
        # support both old and new gym step signatures
        if len(res) == 4:
            obs, reward, done, info = res
            terminated = done
            truncated = False
            returned_5 = False
        else:
            obs, reward, terminated, truncated, info = res
            returned_5 = True

        done_flag = terminated or truncated

        lives = self._get_lives(info)
        life_lost = False
        if lives is not None:
            if lives < self.lives and lives > 0:
                # lost life but not game over
                life_lost = True
                done_flag = True

            self.lives = lives

        # track whether this was a real done (game over)
        self.was_real_done = done_flag and (lives is None or lives == 0)

        if returned_5:
            # return in new gym format
            # if we forced a terminal due to life loss, set terminated True, truncated False
            if life_lost:
                return obs, reward, True, False, info
            return obs, reward, terminated, truncated, info
        else:
            # old 4-tuple
            return obs, reward, done_flag, info

    def reset(self, **kwargs):
        # If the previous was a real game over, do a full reset. Otherwise, just do a no-op step to advance from the terminal state.
        if self.was_real_done:
            res = self.env.reset(**kwargs)
            # handle both return types
            if isinstance(res, tuple):
                obs, info = res
                was_tuple = True
            else:
                obs = res
                info = {}
                was_tuple = False
        else:
            # no-op step to advance from lost-life state
            try:
                # try a single NOOP action
                res = self.env.step(0)
            except Exception:
                # fall back to reset if step fails
                res = self.env.reset(**kwargs)

            if len(res) == 4:
                obs, _, _, info = res
                was_tuple = False
            else:
                obs, _, _, _, info = res
                was_tuple = True

        # update lives count after reset
        lives = None
        try:
            lives = self.env.unwrapped.ale.lives()
        except Exception:
            try:
                lives = self.env.unwrapped.lives()
            except Exception:
                # try info
                if isinstance(info, dict):
                    lives = info.get('lives', None)
        if lives is not None:
            self.lives = lives

        return (obs, info) if was_tuple else obs


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards either by sign or to a specific range.
    - If use_sign=True: reward -> sign(reward) in {-1, 0, 1}
    - Else: reward clipped to [min_reward, max_reward]
    Compatible with both old and new gym APIs.
    """
    def __init__(self, env, use_sign=True, min_reward=-1.0, max_reward=1.0):
        super(ClipRewardEnv, self).__init__(env)
        self.use_sign = use_sign
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward):
        if self.use_sign:
            # 如果想对 reward 做裁剪，可以使用下面自定义的 ClipRewardEnv：
            if reward > 0:
                return 1.0
            elif reward < 0:
                return -1.0
            else:
                return 0.0
        else:
            # 或按范围裁剪到 [-1,1]:
            return float(np.clip(reward, self.min_reward, self.max_reward))

def make_env(env_name, render_mode=None, train=True):
    env = gym.make(env_name, render_mode=render_mode)
    if train:
        env = NoopResetEnv(env)  # 环境reset时，不进行任何操作
        env = EpisodicLifeEnv(env)  # 每失去一条命，done=True，但不reset环境
        env = ClipRewardEnv(env, use_sign=True)  # 将 reward 裁剪为 sign(-1,0,1)
    env = MaxAndSkipEnv(env) # 每4帧采样一次，reward为4帧之和，state是最后两帧的最大值
    env = FireResetEnv(env) # 环境reset时，按开火键
    env = ProcessFrame84(env) # 将RGB图像转为灰度图，并裁剪、缩放为84x84
    env = ImageToPyTorch(env) # 将图像shape从(H,W,C)转为(C,H,W)，并且归一化到0-1之间
    env = BufferWrapper(env, 4) # 堆叠4帧图像作为一个state
    return ScaledFloatFrame(env) # 将图像归一化到0-1之间