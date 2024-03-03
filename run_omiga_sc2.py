import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
import argparse

import numpy as np
from framework.utils import set_seed, load_data, get_dim_from_space
from framework.buffer import ReplayBuffer, StateActionReturnDataset, SequentialDataset
from datetime import datetime, timedelta
import time
from tensorboardX.writer import SummaryWriter
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--algorithm', type=str, default='omiga')
parser.add_argument('--game', type=str, default='StarCraft')
parser.add_argument('--eval_epochs', type=int, default=32)
parser.add_argument('--buffer_size', type=int, default=5000)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--offline_data_dir', type=str, default='../data/6h_vs_8z/good/')
parser.add_argument('--offline_episodes', type=int, default=1000)
parser.add_argument('--offline_episode_bias', type=int, default=0)
parser.add_argument('--offline_batch_size', type=int, default=128)  # episodes used for offline train
parser.add_argument('--offline_epochs', type=int, default=55)
parser.add_argument('--offline_lr', type=float, default=5e-4)
parser.add_argument('--offline_log_dir', type=str, default='./offline_logs/')
parser.add_argument('--offline_eval_interval', type=int, default=1)
# parser.add_argument('--offline_target_interval', type=int, default=20)

# for mixing net
parser.add_argument('--hyper_hidden_dim', type=int, default=64)
parser.add_argument('--qmix_hidden_dim', type=int, default=64)

parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
parser.add_argument("--use_obs_instead_of_state", action='store_true',
                    default=False, help="Whether to use global state or concatenated obs")

# StarCraftII environment
parser.add_argument('--map_name', type=str, default='6h_vs_8z', help="Which smac map to run on")
parser.add_argument("--add_move_state", action='store_true', default=False)
parser.add_argument("--add_local_obs", action='store_true', default=False)
parser.add_argument("--add_distance_state", action='store_true', default=False)
parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
parser.add_argument("--add_agent_id", action='store_true', default=True)
parser.add_argument("--add_visible_state", action='store_true', default=False)
parser.add_argument("--add_xy_state", action='store_true', default=False)
parser.add_argument("--use_state_agent", action='store_true', default=True)
parser.add_argument("--use_mustalive", action='store_false', default=True)
parser.add_argument("--add_center_xy", action='store_true', default=True)
parser.add_argument("--stacked_frames", type=int, default=1,
                    help="Dimension of hidden layers for actor/critic networks")
parser.add_argument("--use_stacked_frames", action='store_true',
                    default=False, help="Whether to use stacked_frames")
parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for evaluating rollouts")

args = parser.parse_args()

from envs.env import Env
import torch

# import wandb
# wandb.init(project='offline-marl', entity='xxx', name = f"{args.algorithm}_{args.map_name}_{args.seed}")


if __name__ == '__main__':

    set_seed(args.seed)
    logging.basicConfig(filename='ma_dt.log', filemode='a',
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    eval_env = Env(args)
    online_train_env = Env(args)
    global_obs_dim = get_dim_from_space(online_train_env.real_env.share_observation_space)
    local_obs_dim = get_dim_from_space(online_train_env.real_env.observation_space)
    action_dim = get_dim_from_space(online_train_env.real_env.action_space)
    num_agents = online_train_env.num_agents
    block_size = args.context_length * 1

    buffer = ReplayBuffer(online_train_env.num_agents, args.buffer_size, args.context_length)
    buffer.reset()

    cur_time = datetime.now() + timedelta(hours=0)
    args.offline_log_dir += args.algorithm + "_" + args.offline_data_dir.split("/")[-2] + "_" + str(args.seed)
    args.offline_log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")
    writer = SummaryWriter(args.offline_log_dir)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    if args.algorithm == 'omiga':
        from models.omiga_model import OMIGA, OMIGAConfig, MixNet
        from framework.trainer_omiga import Trainer, TrainerConfig
        from framework.rollout_omiga import RolloutWorker

        q_mix_model = MixNet(global_obs_dim, num_agents, args.hyper_hidden_dim, action_dim)
        v_mix_model = MixNet(global_obs_dim, num_agents, args.hyper_hidden_dim, action_dim)

        mconf = OMIGAConfig(
            num_actions=action_dim,
            state_dim=global_obs_dim,
            obs_dim=local_obs_dim,
            num_agents=num_agents,
        )

        model = OMIGA(q_mix_model, v_mix_model,  mconf)
        rollout_worker = RolloutWorker(model.actor, buffer, args.context_length)
        offline_tconf = TrainerConfig(max_epochs=1, num_agents=num_agents, num_actions=action_dim, batch_size=args.offline_batch_size,
                                      learning_rate=args.offline_lr, num_workers=0,
                                      seed=args.seed, game=args.game, log_dir=args.offline_log_dir)
        offline_trainer = Trainer(model, offline_tconf)
        gc.collect()

    else:
        model = None
        rollout_worker = None
        offline_trainer = None
        offline_dataset = None
        mconf = None

    print("offline total episodes: ", buffer.cur_size)
    last_eval_steps = 0

    for i in range(args.offline_epochs):
        print("offline iter: ", i + 1)
        offline_trainer.train(
            episodes=args.offline_episodes, data_dir=args.offline_data_dir, num_agents=num_agents
        )

        # if (i + 1) % args.offline_eval_interval == 0:
        #     aver_return, aver_win_rate = rollout_worker.rollout(eval_env, 20, args.eval_epochs, train=False)
        #     print("offline return: %s, win_rate: %s" % (aver_return, aver_win_rate))
        #     wandb.log({'aver_return': aver_return, 'aver_win_rate': aver_win_rate})

    # path = "save/model.pth"
    # torch.save(model, path)

    online_train_env.real_env.close()
    eval_env.real_env.close()
