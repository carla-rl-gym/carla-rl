#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import logging
import os

from carla.driving_benchmark.experiment_suites import CoRL2017
from carla.driving_benchmark.experiment_suites import BasicExperimentSuite

from utils import load_modules
from model import Policy
import agents

import torch

from carla_logger import setup_carla_logger
from train import get_config_and_checkpoint
from envs_manager import make_vec_envs
from storage import RolloutStorage
from utils import get_vec_normalize, save_modules, load_modules

import datetime
from tensorboardX import SummaryWriter

import agents
from arguments import get_args
from observation_utils import CarlaObservationConverter
from action_utils import CarlaActionsConverter
from env import CarlaEnv
from vec_env.util import dict_to_obs, obs_to_dict
from carla.driving_benchmark.recording import Recording
from carla.driving_benchmark.metrics import Metrics
from carla.driving_benchmark.results_printer import print_summary

from carla.tcp import TCPConnectionError

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-conf', '--config',
        default=None,
        help='Values to configure training at the beginning')
    argparser.add_argument(
        '--resume_training',
        default=False,
        help='path to saved model')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--corl-2017',
        action='store_true',
        help='If you want to benchmark the corl-2017 instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )
    argparser.add_argument(
        '--resume-training',
        default=None,
        help='If you want to continue the experiment with the same name'
    )
    argparser.add_argument(
	'--cuda',
	default=False,
	help='If you are using a CPU, set it to False'
    )
    argparser.add_argument(
	'--save-dir',
	default='./outputs',
	help='Directory to save model, logs and videos'
    )
    argparser.add_argument(
	'--video-interval', type=int, default=1
    )
    argparser.add_argument(
	'--save-interval', type=int, default=5
    )
    args = argparser.parse_args()
    log_level = logging.INFO

    # logging.basicConfig(filename='test.log', format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    config, checkpoint = get_config_and_checkpoint(args)
    obs_converter = CarlaObservationConverter(h=84, w=84)
    action_converter = CarlaActionsConverter(config.action_type)

    device = torch.device("cpu")
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    norm_reward = not config.no_reward_norm

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    if args.corl_2017:
        experiment_suite = CoRL2017(args.city_name)
        experiment_name = 'CoRL2017'
    else:
        print (' WARNING: running the basic driving benchmark, to run for CoRL 2017'
               ' experiment suites, you should run'
               ' python driving_benchmark_example.py --corl-2017')
        experiment_suite = BasicExperimentSuite(args.city_name)
        experiment_name = 'BasicExperimentSuite'

    logger = setup_carla_logger('output_logger/', experiment_name, 'carla-debug')
    envs = make_vec_envs(obs_converter, action_converter, args.port, config.seed, config.num_processes,
                         config.gamma, device, reward_class_name='RewardCarla', num_frame_stack=1, subset=None,
                         norm_reward=norm_reward,
                         video_every=args.video_interval,
                         video_dir=os.path.join(args.save_dir, 'video'),
                         experiment_suite=experiment_name,
                         benchmark=True,
                         city_name=args.city_name)
    experiments = experiment_suite.get_experiments()

    # Agent instantiation
    if config.agent == 'forward':
        agent = agents.ForwardCarla()

    elif config.agent == 'a2c':
        agent = agents.A2CCarla(obs_converter,
                                action_converter,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps, alpha=config.alpha,
                                max_grad_norm=config.max_grad_norm)

    elif config.agent == 'acktr':
        agent = agents.A2CCarla(obs_converter,
                                action_converter,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps, alpha=config.alpha,
                                max_grad_norm=config.max_grad_norm,
                                acktr=True)

    elif config.agent == 'ppo':
        agent = agents.PPOCarla(obs_converter,
                                action_converter,
                                config.clip_param,
                                config.ppo_epoch,
                                config.num_mini_batch,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps,
                                max_grad_norm=config.max_grad_norm)

    else:
        raise NotImplementedError

    if checkpoint is not None:
        load_modules(agent.optimizer, agent.model, checkpoint)

    vec_norm = get_vec_normalize(envs)
    #if vec_norm is not None:
    #    vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    metrics_object = Metrics(experiment_suite.metrics_parameters,
                             experiment_suite.dynamic_tasks)
    recording = Recording(name_to_save=args.save_dir,
              continue_experiment=False,
              save_images=True)

    logging.info('START')
    iter = True
    obs = None
    while iter:
        try:
            if obs is None:
                print('should be reset')
                obs = envs.reset()
            elif obs is False:
                print('end of the experiments')
                iter = False
                break
            [exp_idx, pose_idx, rep] = envs.venv.venv.venv.envs[0].benchmark_index
            experiment = experiments[exp_idx]
            pose = experiment.poses[0:][pose_idx]
            rep = rep - 1
            recording.log_start(id_experiment=experiment.task)

            poses = experiment.poses[0:]
            print('Benchmarking experiment {} out of {}, which contains {} poses'.format(exp_idx+1, len(experiments), len(poses)))

            start_index = pose[0]
            end_index = pose[1]
            logging.info('======== !!!! ==========')
            logging.info(' Start Position %d End Position %d ',
                         start_index, end_index)
            recording.log_poses(start_index, end_index,
                                experiment.Conditions.WeatherId)

            initial_distance = envs.venv.venv.venv.envs[0].last_distance_to_goal

            recurrent_hidden_states = torch.zeros(config.num_processes,
                                                       20, device=device)
            masks = torch.zeros(config.num_processes, 1, device=device)

            reward_vec = []
            control_vec = []
            print('beginning of the while loop')
            done = [False, False, False]
            while any(done) is False:
            # while (envs.venv.venv.venv.envs[0]._failure_timeout is False) and (envs.venv.venv.venv.envs[0]._failure_collision is False) and (envs.venv.venv.venv.envs[0]._success is False):
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs, recurrent_hidden_states, masks, deterministic=True)
                # Observe reward and next obs
                carla_obs, reward, done, infos = envs.step(action)
                masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                if config.action_type == 'carla-original':
                    control = action_converter.action_to_control(int(action[0][0]))
                else:
                    control = action_converter.action_to_control(action[0])
                if envs.venv.venv.venv.envs[0]._reward.state is None:
                    rw_vec = {'d_x': envs.venv.venv.venv.envs[0].last_measurements.player_measurements.transform.location.x,
                              'd_y': envs.venv.venv.venv.envs[0].last_measurements.player_measurements.transform.location.y,
                              'c_o': 0, 'c_p':0, 'c_v': 0, 'o':0, 's':0}
                else:
                    rw_vec = envs.venv.venv.venv.envs[0]._reward.state

                reward_vec.append(rw_vec)
                control_vec.append(control)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                recording.write_summary_results(
                experiments[exp_idx],
                pose,
                rep,
                initial_distance,
                envs.venv.venv.venv.envs[0].last_distance_to_goal,
                envs.venv.venv.venv.envs[0].last_measurements.game_timestamp - envs.venv.venv.venv.envs[0]._initial_timestamp,
                int(envs.venv.venv.venv.envs[0]._failure_timeout), int(envs.venv.venv.venv.envs[0]._success))

                if envs.venv.venv.venv.envs[0]._success:
                    logging.info('+++++ Target achieved in %f seconds! +++++',
                                 envs.venv.venv.venv.envs[0].last_measurements.game_timestamp - envs.venv.venv.venv.envs[0]._initial_timestamp)

                elif envs.venv.venv.venv.envs[0]._failure_timeout:
                    logging.info('----- Timeout! -----')

                else:
                    logging.info('----- Collision! -----')


            # Write the details of this episode.
            print('end of the while loop')
            recording.write_measurements_results(experiments[exp_idx], rep, pose, reward_vec, control_vec)


            print('experiment ended')
            recording.log_end()
        except:
            break
    recording.write_measurements_results(experiments[exp_idx], rep, pose, reward_vec, control_vec)

    benchmark_summary = metrics_object.compute(recording.path)

    print("")
    print("")
    print("----- Printing results for training weathers (Seen in Training) -----")
    print("")
    print("")
    print_summary(benchmark_summary, experiment_suite.train_weathers,
                                  recording.path)

    print("")
    print("")
    print("----- Printing results for test weathers (Unseen in Training) -----")
    print("")
    print("")

    print_summary(benchmark_summary, experiment_suite.test_weathers,
                                  recording.path)
