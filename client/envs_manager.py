import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from vec_env import VecEnvWrapper
from vec_env.subproc_vec_env import SubprocVecEnv
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.vec_normalize import VecNormalize
from vec_env.util import dict_to_obs, obs_to_dict

from env import CarlaEnv

def make_env(obs_converter, action_converter, port, id, seed, subset,
             video_every, video_dir, reward_class_name, experiment_suite, benchmark, city_name):
    return lambda: CarlaEnv(obs_converter, action_converter, id, seed, reward_class_name=reward_class_name, port=port,
                            subset=subset, video_every=video_every, video_dir=video_dir,
                            exp_suite_name=experiment_suite,
                            benchmark=benchmark, city_name=city_name)

def make_vec_envs(obs_converter, action_converter, starting_port, seed, num_processes, gamma,
                  device, reward_class_name, num_frame_stack=1, subset=None, norm_reward=True, norm_obs=True, video_every=100, video_dir='./video', apply_her=False,
                  experiment_suite='TrainingSuite', benchmark=False, city_name='Town01'):

    ports = range(starting_port, starting_port + 3*num_processes, 3)
    envs = [make_env(obs_converter, action_converter, ports[i], i, seed + i,
                     subset, video_every, video_dir, reward_class_name,
                     experiment_suite,
                     benchmark, city_name) for i in range(num_processes)]

    if len(envs) > 1 or apply_her:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if gamma is None:
        envs = VecNormalize(envs, ret=False, ob=norm_obs)
    else:
        envs = VecNormalize(envs, gamma=gamma, ret=norm_reward, ob=norm_obs)

    envs = VecPyTorch(envs, device)

    envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data typesVecEnvWrapper

    def reset(self):
        obs = self.venv.reset()
        obs = obs_to_dict(obs)
        for k in obs:
            obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        return dict_to_obs(obs)

    def step_async(self, actions):
        if type(actions) != np.ndarray:
            actions = actions.cpu().numpy()
            # actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = obs_to_dict(obs)
        for k in obs:
            obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return dict_to_obs(obs), reward, done, info



# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):

        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        wos = obs_to_dict(wos)
        self.stacked_obs = {}
        new_observation_spaces = {}
        self.shape_dim0 = {}
        for k in wos.spaces:

            self.shape_dim0[k] = wos.spaces[k].shape[0]
            low = np.repeat(wos.spaces[k].low, self.nstack, axis=0)
            high = np.repeat(wos.spaces[k].high, self.nstack, axis=0)

            if device is None:
                device = torch.device('cpu')
            self.stacked_obs[k] = torch.zeros((venv.num_envs,) + low.shape).to(device)

            new_observation_spaces[k] = gym.spaces.Box(
                low=low, high=high, dtype=venv.observation_space.dtype)

        if set(new_observation_spaces.keys()) == {None}:
            VecEnvWrapper.__init__(self, venv, observation_space=new_observation_spaces[None])
        else:
            VecEnvWrapper.__init__(self, venv, observation_space=gym.spaces.Dict(new_observation_spaces))


    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = obs_to_dict(obs)
        for k in obs:
            self.stacked_obs[k][:, :-self.shape_dim0[k]] = \
                self.stacked_obs[k][:, self.shape_dim0[k]:]
            for (i, new) in enumerate(news):
                if new:
                    self.stacked_obs[k][i] = 0
            self.stacked_obs[k][:, -self.shape_dim0[k]:] = obs[k]
        return dict_to_obs(self.stacked_obs), rews, news, infos

    def reset(self):

        obs = self.venv.reset()
        obs = obs_to_dict(obs)
        for k in obs:
            self.stacked_obs[k].zero_()
            self.stacked_obs[k][:, -self.shape_dim0[k]:] = obs[k]
        return dict_to_obs(self.stacked_obs)

    def close(self):
        self.venv.close()
