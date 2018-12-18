'''
    Logging for debugging our CARLA training code. Eventually, should only use it inside
    train.py since it should not be used in any code that would be open sourced as a standalone
    OpenAI gym env.
'''
import os
import logging


def get_carla_logger():
    return logging.getLogger('carla-debug')

def setup_carla_logger(save_dir, experiment_name, logger_name='carla-debug'):
    # Logging setup - we need to add a name so that we don't conflict with Tensorboard
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(logging.DEBUG)
    log_dir = os.path.join(save_dir, 'logs', experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(logging.StreamHandler()) # Write logging to STDOUT too
    return logger
