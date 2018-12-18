import torch
import numpy as np

from carla.agent.agent import Agent
from carla.client import VehicleControl

class ForwardCarla(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent agent that goes straight
    """

    def __init__(self, action_type='carla-original'):
        super(ForwardCarla, self).__init__()
        self.action_type = action_type

    def get_value(self, obs, hidden_state, mask):
        return torch.Tensor([0])

    def act(self, obs, hidden_states, masks):
        throttle = 0.9
        if self.action_type == 'carla-original':
            # value, action, log_prob, hidden states
            return torch.Tensor([0]), torch.Tensor([6]), torch.Tensor([1]), torch.zeros([20])
        else:
            return torch.Tensor([0]), torch.Tensor([throttle, 0, 0]), torch.Tensor([1]), torch.zeros([20])


    def update(self, rollouts):
        return 0, 0, 0
