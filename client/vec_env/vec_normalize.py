from . import VecEnvWrapper
import numpy as np
from .util import copy_obs_dict, dict_to_obs, obs_space_info, obs_to_dict


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, norm_type='FixedMaxMin', ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        obs_space = venv.observation_space
        self.obs_space = obs_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.shapes = shapes
        self.norm_type = norm_type
        if norm_type == 'RunningMean':
            self.ob_rms = {k:  RunningMeanStd(shape=shapes[k]) for k in self.keys if len(shapes[k]) == 1} if ob else None
        self.ob = ob
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ob:
            obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.norm_type == 'FixedMaxMin' and self.ob:
            for key in obs:
                if len(self.shapes[key]) == 1:
                    obs[key] = (obs[key] - self.obs_space.spaces[key].low) /(self.obs_space.spaces[key].high - self.obs_space.spaces[key].low)
            return dict_to_obs(obs)
        elif self.norm_type == 'RunningMean' and self.ob_rms:
            obs = obs_to_dict(obs)
            for key in self.ob_rms:
                if self.training: # Only update when training
                    self.ob_rms[key].update(obs[key])
                obs[key] = np.clip((obs[key] - self.ob_rms[key].mean) / np.sqrt(self.ob_rms[key].var + self.epsilon), -self.clipob, self.clipob)
            return dict_to_obs(obs)
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
