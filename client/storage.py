import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from vec_env.util import dict_to_obs, obs_to_dict

import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_space, action_space, recurrent_hidden_state_size, 
                    num_virtual_goals, rel_coord_system, obs_converter):

        self.obs_space = obs_space
        self.action_space = action_space
        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.her = num_virtual_goals
        self.rel_coord_system = rel_coord_system
        self.obs_converter = obs_converter
        
        if self.obs_space.shape:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        else: 
            # Observation space is Dict
            self.obs = {k: torch.zeros(num_steps + 1, num_processes, *v.shape) for k, v in self.obs_space.spaces.items()}
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if self.action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = self.action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if self.action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

    def to(self, device):
        self.obs = obs_to_dict(self.obs)
        for k in self.obs:
            self.obs[k] = self.obs[k].to(device)
        self.obs = dict_to_obs(self.obs)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        obs = obs_to_dict(obs)
        self.obs = obs_to_dict(self.obs)
        for k in self.obs:
            self.obs[k][self.step + 1].copy_(obs[k])
        self.obs = dict_to_obs(self.obs)
        # self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def get_obs(self, step):
        if isinstance(self.obs, dict):
            return {k: self.obs[k][step] for k in self.obs}
        else:
            return self.obs[step] 

    def after_update(self):

        self.obs = obs_to_dict(self.obs)
        
        if self.her:
            for k in self.obs:
                self.obs[k] = self.obs[k][:, :self.num_processes]
            self.masks = self.masks[:, :self.num_processes]
            self.returns = self.returns[:, :self.num_processes]
            self.value_preds = self.value_preds[:, :self.num_processes]
            self.rewards = self.rewards[:, :self.num_processes]
            self.action_log_probs = self.action_log_probs[:, :self.num_processes]
            self.actions = self.actions[:, :self.num_processes]
            self.recurrent_hidden_states = self.recurrent_hidden_states[:, :self.num_processes]
        
        for k in self.obs:
            self.obs[k][0].copy_(self.obs[k][-1])
        self.obs = dict_to_obs(self.obs)
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]


    def apply_her(self, num_virtual_goals, device, beta=1):

        # Sample examples 
        r = torch.tensor(range(self.masks.shape[0]-1), dtype=torch.float)
        # TODO: check there are no instabilities here (underflow?)
        p = (torch.nn.functional.softmax(beta*r, dim=0).view(self.masks.shape[0]-1, 1, 1).cuda() * self.masks[1:])
        p[0] = 0
        p = p.view(-1)
        global_idx = torch.multinomial(p.cpu(), num_virtual_goals, replacement=False).numpy()
        process_idx = (global_idx // (self.masks.shape[0]-1))
        step_idx = (global_idx % (self.masks.shape[0]-1))

        # Init rollouts for HER virtual goals
        # These will be stacked into the actual rollouts along the process dimension
        assert isinstance(self.obs, dict), 'Observation must be a dictionary with keys img and v in order to use HER'
        obs = {k: torch.zeros(self.num_steps + 1, num_virtual_goals, *v.shape).to(device) for k, v in self.obs_space.spaces.items()}
        rewards = torch.zeros(self.num_steps, num_virtual_goals, 1).to(device)
        value_preds = torch.zeros(self.num_steps + 1, num_virtual_goals, 1).to(device)
        returns = torch.zeros(self.num_steps + 1, num_virtual_goals, 1).to(device)
        action_log_probs = torch.zeros(self.num_steps, num_virtual_goals, 1).to(device)
        recurrent_hidden_states = torch.zeros(self.num_steps + 1, num_virtual_goals, self.recurrent_hidden_state_size).to(device)
        if self.action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = self.action_space.shape[0]
        actions = torch.zeros(self.num_steps, num_virtual_goals, action_shape).to(device)
        if self.action_space.__class__.__name__ == 'Discrete':
            actions = actions.long().to(device)
        masks = torch.zeros(self.num_steps + 1, num_virtual_goals, 1).to(device)

        # Calculate new rollouts
        for k, (i, j) in enumerate(list(zip(process_idx, step_idx))):
            # Traverse backwards in time
            for sj in range(j+1)[::-1]:
                if self.masks[sj+1, i, 0].item() == 0: 
                    break
            masks[sj+1:j+1, k].copy_(self.masks[sj+1:j+1, i])
            returns[:, k].copy_(self.returns[:, i])
            value_preds[:, k].copy_(self.value_preds[:, i])
            action_log_probs[:, k].copy_(self.action_log_probs[:, i])
            actions[:, k].copy_(self.actions[:, i])
            recurrent_hidden_states[:, k].copy_(self.recurrent_hidden_states[:, i])
            obs['img'][:, k].copy_(self.obs['img'][:, i])
            rewards[j, k] = 500
            obs['v'][:, k].copy_(self.obs['v'][:, i])
            if self.rel_coord_system:
                virtual_goal_world = self.obs['world_pos'][j+1, i]
                if ((j - sj) < 10):
                    masks[:, k] = 0
                    print('Discarding virtual goal due to short seq ({} timesteps)'.format(j - sj))
                else:
                    traversed_distance = torch.sqrt((virtual_goal_world[:2] - self.obs['world_pos'][sj+2, i, 0:2])**2).sum().item()
                    average_speed = traversed_distance * 100 * 10 / (j - sj)
                    if average_speed < 0.1: # less than 0.1 m/s
                        masks[:, k] = 0
                        print('Discarding virtual goal due to slow movement ({} m/s)'.format(average_speed))
                        continue
                    else:
                        print('Accepted virtual goal ({} m/s)'.format(average_speed))
                for t in range(j+1):
                    target_x, target_y = self.obs_converter.get_relative_location_target(
                                            self.obs['world_pos'][t+1, i, 0], 
                                            self.obs['world_pos'][t+1, i, 1], 
                                            self.obs['world_pos'][t+1, i, 2],
                                            virtual_goal_world[0], 
                                            virtual_goal_world[1])
                    obs['v'][t+1, k, -4] = target_x
                    obs['v'][t+1, k, -3] = target_y
                    target_norm = np.linalg.norm(np.array([target_x, target_y]))
                    if target_norm > 1e-6:
                        obs['v'][t+1, k, -2] = target_x / target_norm
                        obs['v'][t+1, k, -1] = target_y / target_norm
                    # We don't know what would be the directions for the HER trajectory
                    obs['v'][:, k, 3:8] = 0.0
                # print('from {} to {}'.format(obs['v'][sj+2, k, -4:-2], obs['v'][j+1, k, -4:-2]))
                obs['world_pos'][:, k].copy_(self.obs['world_pos'][:, i])
            else:
                obs['v'][:, k, -3:] = self.obs['v'][j+1:j+2, i, :3]

        # Stack rollouts
        self.masks = torch.cat([self.masks, masks], 1)
        self.returns = torch.cat([self.returns, returns], 1)
        self.value_preds = torch.cat([self.value_preds, value_preds], 1)
        self.rewards = torch.cat([self.rewards, rewards], 1)
        self.action_log_probs = torch.cat([self.action_log_probs, action_log_probs], 1)
        self.actions = torch.cat([self.actions, actions], 1)
        self.recurrent_hidden_states = torch.cat([self.recurrent_hidden_states, recurrent_hidden_states], 1)
        self.obs['img'] = torch.cat([self.obs['img'], obs['img']], 1)
        self.obs['v'] = torch.cat([self.obs['v'], obs['v']], 1)
        if self.rel_coord_system:
            self.obs['world_pos'] = torch.cat([self.obs['world_pos'], obs['world_pos']], 1)

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = {} 
            self.obs = obs_to_dict(self.obs)
            for k in self.obs: 
                obs_batch[k] = self.obs[k][:-1].view(-1, *self.obs[k].size()[2:])[indices]
            self.obs = dict_to_obs(self.obs)
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            obs_v_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            obs_v_batch = torch.stack(obs_v_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            obs_v_batch = _flatten_helper(T, N, obs_v_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, obs_v_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
