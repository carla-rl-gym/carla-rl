import copy
import os
import time
import yaml
import shutil
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from carla.tcp import TCPConnectionError
from carla_logger import setup_carla_logger

import traceback
import argparse
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

def get_config_and_checkpoint(args):
    config_dict, checkpoint = None, None
    if args.config and args.resume_training:
        print('ERROR: Should either provide --config or --resume-training but not both.')
        exit(1)

    if args.config:
        config_dict = load_config_file(args.config)

    if args.resume_training:
        print('Resuming training from: {}'.format(args.resume_training))
        assert os.path.isfile(args.resume_training), 'Checkpoint file does not exist'
        checkpoint = torch.load(args.resume_training)
        config_dict = checkpoint['config']

    if config_dict is None:
        print("ERROR: --config or --resume-training flag is required.")
        exit(1)

    config = namedtuple('Config', config_dict.keys())(*config_dict.values())
    return config, checkpoint

def load_config_file(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

        # To be careful with values like 7e-5
        config['lr'] = float(config['lr'])
        config['eps'] = float(config['eps'])
        config['alpha'] = float(config['alpha'])
        return config

def set_random_seeds(args, config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if args.cuda:
        torch.cuda.manual_seed(config.seed)
    # TODO: Set CARLA seed (or env seed)

def main():
    config = None
    args = get_args()
    config, checkpoint = get_config_and_checkpoint(args)

    set_random_seeds(args, config)
    eval_log_dir = args.save_dir + "_eval"
    try:
        os.makedirs(args.save_dir)
        os.makedirs(eval_log_dir)
    except OSError:
        pass

    now = datetime.datetime.now()
    experiment_name = args.experiment_name + '_' + now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create checkpoint file
    save_dir_model = os.path.join(args.save_dir, 'model', experiment_name)
    save_dir_config = os.path.join(args.save_dir, 'config', experiment_name)
    try:
        os.makedirs(save_dir_model)
        os.makedirs(save_dir_config)
    except OSError as e:
        logger.error(e)
        exit()

    if args.config:
        shutil.copy2(args.config, save_dir_config)

    # Tensorboard Logging
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard', experiment_name))

    # Logger that writes to STDOUT and a file in the save_dir
    logger = setup_carla_logger(args.save_dir, experiment_name)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    norm_reward = not config.no_reward_norm
    norm_obs = not config.no_obs_norm

    assert not (config.num_virtual_goals > 0) or (config.reward_class == 'SparseReward'), 'Cant use HER with dense reward'
    obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=config.rel_coord_system)
    action_converter = CarlaActionsConverter(config.action_type)
    envs = make_vec_envs(obs_converter, action_converter, args.starting_port, config.seed, config.num_processes,
                                config.gamma, device, config.reward_class, num_frame_stack=1, subset=config.experiments_subset,
                                norm_reward=norm_reward, norm_obs=norm_obs, apply_her=config.num_virtual_goals > 0,
                                video_every=args.video_interval, video_dir=os.path.join(args.save_dir, 'video', experiment_name))


    if config.agent == 'forward':
        agent = agents.ForwardCarla()

    if config.agent == 'a2c':
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

    if checkpoint is not None:
        load_modules(agent.optimizer, agent.model, checkpoint)

    rollouts = RolloutStorage(config.num_steps, config.num_processes,
                        envs.observation_space, envs.action_space, 20,
                        config.num_virtual_goals, config.rel_coord_system, obs_converter)

    obs = envs.reset()
    # Save the first observation
    obs = obs_to_dict(obs)
    rollouts.obs = obs_to_dict(rollouts.obs)
    for k in rollouts.obs:
        rollouts.obs[k][rollouts.step + 1].copy_(obs[k])
    rollouts.obs = dict_to_obs(rollouts.obs)
    rollouts.to(device)

    start = time.time()


    total_steps = 0
    total_episodes = 0
    total_reward = 0

    episode_reward = torch.zeros(config.num_processes)


    for j in range(config.num_updates):

        for step in range(config.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.act(
                        rollouts.get_obs(step),
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, info = envs.step(action)

            # For logging purposes
            carla_rewards = torch.tensor([i['carla-reward'] for i in info], dtype=torch.float)
            episode_reward += carla_rewards
            total_reward += carla_rewards.sum().item()
            total_steps += config.num_processes

            if done.any():
                total_episodes += done.sum()
                torch_done = torch.tensor(done.astype(int)).byte()
                mean_episode_reward = episode_reward[torch_done].mean().item()
                logger.info('{} episode(s) finished with reward {}'.format(done.sum(), mean_episode_reward))
                writer.add_scalar('train/mean_ep_reward_vs_steps', mean_episode_reward, total_steps)
                writer.add_scalar('train/mean_ep_reward_vs_episodes', mean_episode_reward, total_episodes)
                episode_reward[torch_done] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(1-done)

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks.unsqueeze(-1))

        if config.num_virtual_goals > 0:
            rollouts.apply_her(config.num_virtual_goals, device, beta=config.beta)

        with torch.no_grad():
            next_value = agent.get_value(rollouts.get_obs(-1), # Get last observation
                                         rollouts.recurrent_hidden_states[-1],
                                         rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, config.use_gae, config.gamma, config.tau)


        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "" and config.agent !='forward':
            save_path = os.path.join(save_dir_model, str(j) + '.pth.tar')
            save_modules(agent.optimizer, agent.model, args, config, save_path)

        total_num_steps = (j + 1) * config.num_processes * config.num_steps

        if j % args.log_interval == 0:

            # Logging to the stdout/our logs
            end = time.time()
            logger.info('------------------------------------')
            logger.info('Episodes {}, Updates {}, num timesteps {}, FPS {}'\
                .format(total_episodes, j + 1, total_num_steps, total_num_steps / (end - start)))
            logger.info('------------------------------------')


            # Logging to tensorboard
            writer.add_scalar('train/cum_reward_vs_steps', total_reward, total_steps)
            writer.add_scalar('train/cum_reward_vs_updates', total_reward, j+1)

            if config.agent in ['a2c', 'acktr', 'ppo']:
                writer.add_scalar('debug/value_loss_vs_steps', value_loss, total_steps)
                writer.add_scalar('debug/value_loss_vs_updates', value_loss, j+1)
                writer.add_scalar('debug/action_loss_vs_steps', action_loss, total_steps)
                writer.add_scalar('debug/action_loss_vs_updates', action_loss, j+1)
                writer.add_scalar('debug/dist_entropy_vs_steps', dist_entropy, total_steps)
                writer.add_scalar('debug/dist_entropy_vs_updates', dist_entropy, j+1)

            # Sample the last reward
            writer.add_scalar('debug/sampled_normalized_reward_vs_steps', reward.mean(), total_steps)
            writer.add_scalar('debug/sampled_normalized_reward_vs_updates', reward.mean(), j+1)
            writer.add_scalar('debug/sampled_carla_reward_vs_steps', carla_rewards.mean(), total_steps)
            writer.add_scalar('debug/sampled_carla_reward_vs_updates', carla_rewards.mean(), j+1)

        if (args.eval_interval is not None and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.starting_port, obs_converter, args.x + config.num_processes, config.num_processes,
                config.gamma, eval_log_dir, config.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(config.num_processes,
                            20, device=device)
            eval_masks = torch.zeros(config.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = agent.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                carla_obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            logger.info(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))



if __name__ == "__main__":
    main()
