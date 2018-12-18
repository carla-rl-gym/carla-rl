import numpy as np

class SparseReward():
    '''
        Implict Reward Function defined by the benchmarking code. 1 is successful, 0 otherwise. 
    '''

    def __init__(self):
        pass

    def get_reward(self, measurements, target, directions, action, env_state):
        if env_state['success']:
            return 1
        return 0

    def reset_reward(self):
        return
