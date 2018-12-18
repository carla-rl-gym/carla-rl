import numpy as np

class HERReward():
    '''
        A modified variant of sparse rewards for training HER. Returns 500 when successful
        and otherwise the velocity of the vehicle.
    '''

    def __init__(self):
        pass

    def get_reward(self, measurements, target, directions, action, env_state):
        v = min(25, measurements.player_measurements.forward_speed * 3.6) / 25
        if env_state['success']:
            return 500.0
        return v


    def reset_reward(self):
        return
