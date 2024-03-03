from .starcraft2.StarCraft2_Env import StarCraft2Env
from .starcraft2.smac_maps import get_map_params
from .config import get_config
from .env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


class Env:
    def __init__(self, config):
        # parser = get_config()
        # all_args = parser.parse_known_args()[0]
        self.real_env = make_eval_env(config)
        self.num_agents = get_map_params(config.map_name)["n_agents"]
        self.max_timestep = get_map_params(config.map_name)["limit"]
