
import gym
import numpy as np
from carla.client import VehicleControl


class CarlaActionsConverter(object):

    def __init__(self, action_type='continuous'):

        self.action_type = action_type

        if self.action_type == 'carla-original':
            self.discrete_actions = [[0., 0.], [-1.,0.], [-0.5,0.], [-0.25,0.], [0.25,0.], [0.5, 0.], [1.0, 0.], [0., -1.],
                                        [0., -0.5], [0., -0.25], [0., 0.25], [0., 0.5], [0.,1.]]


    def get_action_space(self):

        if self.action_type == 'carla-original':
            return gym.spaces.Discrete(len(self.discrete_actions))

        elif self.action_type == 'continuous':
            low = [0.0, 0.0, -1.0]
            high = [1.0, 1.0, 1.0]
            return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)


    def action_to_control(self, action, last_measurements=None):
        control = VehicleControl()
        if self.action_type == 'carla-original':
            if type(action) == np.ndarray:
                action = action.item()
            if type(action) != int:
                print('Unexpected action got {}'.format(type(action)))
            assert type(action) == int, 'Action should be an int'
            action = self.discrete_actions[action]
            if last_measurements is not None and last_measurements.player_measurements.forward_speed * 3.6 < 30:
                control.throttle = action[0]
            elif action[0] > 0.:
                control.throttle = 0.
            control.steer = action[1]

        elif self.action_type == 'continuous':
            control.throttle = min(1, max(0.0, action[0]))
            control.brake = min(1, max(0.0, action[1]))
            control.steer = min(1, max(-1, action[2]))

        # print('Control: {}, {}, {}'.format(control.throttle, control.brake, control.steer))
        return control
