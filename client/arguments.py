import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='CARLA RL Training Script')
    parser.add_argument('--experiment-name', type=str, default='debug')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--starting-port', type=int, default=2000,
                        help='starting_port')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--video-interval', type=int, default=100,
                        help='create a visualization of the agent behavior every number of episodes')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='interval to save a checkpoint, one save per n updates (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-dir', default='./outputs', help='directory to save models, logs and videos')
    parser.add_argument('--resume-training', default=None, help='checkpoint file to resume training from')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
