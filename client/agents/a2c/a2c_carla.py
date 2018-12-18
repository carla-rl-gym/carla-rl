import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .kfac import KFACOptimizer
from carla.agent.agent import Agent
from model import Policy


class A2CCarla(Agent):
    def __init__(self,
                 obs_converter,
                 action_converter,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 eps_greedy_start=0.0,
                 eps_greedy_decay=0.0001, # Decayed every update
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.obs_converter = obs_converter
        self.action_converter = action_converter
        self.model = Policy(self.obs_converter.get_observation_space(),
                            self.action_converter.get_action_space()).to("cuda:0")
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Epsilon Greedy Values
        self.eps_curr = eps_greedy_start
        self.eps_greedy_decay = eps_greedy_decay

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(self.model)
        else:
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr, eps=eps, alpha=alpha)


    def update(self, rollouts):
        # Update Epsilon Greedy
        self.eps_curr = max(0.0, self.eps_curr - self.eps_greedy_decay)
        obs_shape = {k: r.size()[2:] for k, r in rollouts.obs.items()}
        rollouts_flatten = {k: r[:-1].view(-1, *obs_shape[k]) for k, r in rollouts.obs.items()}
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            rollouts_flatten['img'],
            rollouts_flatten['v'],
            rollouts.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.model.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()


    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        return self.model.act(inputs['img'], inputs['v'], rnn_hxs, masks, self.eps_curr, deterministic)

    def get_value(self, inputs, rnn_hxs, masks):
        return self.model.get_value(inputs['img'], inputs['v'], rnn_hxs, masks)
