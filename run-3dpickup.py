"""
The script to run CenRA framework on the 3DPickup environment.

"""

import argparse

from CenRA.Agents import DQNAgent
from CenRA.Algorithms import CenRA_dis

from CenRA.Networks import QNetMiniWorld
from CenRA.Networks import RAActorWorld, RAQNetMiniWorld

from CenRA.utils import miniworld_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run CenRA framework on 3DPickup environment.")

    parser.add_argument("--exp-name", type=str, default="CenRA-3DPickup")

    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--suggested-reward-scale", type=float, default=1)
    parser.add_argument("--lamb", type=float, default=0.5)

    ###### For the Reward Agent ######

    parser.add_argument("--ra-rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--ra-batch-size", type=int, default=256)

    parser.add_argument("--ra-actor-lr", type=float, default=3e-4)
    parser.add_argument("--ra-critic-lr", type=float, default=1e-3)
    parser.add_argument("--ra-alpha-lr", type=float, default=1e-4)

    parser.add_argument("--ra-policy-frequency", type=int, default=2)
    parser.add_argument("--ra-target-frequency", type=int, default=1)

    parser.add_argument("--ra-alpha", type=float, default=0.2)
    parser.add_argument("--ra-alpha-autotune", type=bool, default=True)

    ###### For the Policy Agent (DQN-based agent) ######

    parser.add_argument("--pa-learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--pa-buffer-size", type=int, default=1000000)
    parser.add_argument("--pa-rb-optimize-memory", type=bool, default=True)
    parser.add_argument("--pa-batch-size", type=int, default=128)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=10000)
    parser.add_argument("--save-folder", type=str, default="./CenRA-3DPickup/")

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--pa-learning-starts", type=int, default=1e4)
    parser.add_argument("--ra-learning-starts", type=int, default=5e3)

    args = parser.parse_args()
    return args


def run():
    import torch
    torch.cuda.empty_cache()

    args = parse_args()

    env_ids = ["MiniWorld-3DPickup-ball-red", "MiniWorld-3DPickup-cube-green",
               "MiniWorld-3DPickup-key-blue", "MiniWorld-3DPickup-healthkit"]
    # env_ids = ["MiniWorld-3DPickup-cube-yellow"]  # for the new task

    envs = [miniworld_env_maker(env_id=env_id, seed=args.seed, render=args.render) for env_id in env_ids]

    policy_agents = [
        DQNAgent(env=env, q_network_class=QNetMiniWorld, exp_name=f"{args.exp_name}-{env.unwrapped.spec.id}",
                 seed=args.seed, cuda=args.cuda, learning_rate=args.pa_learning_rate, buffer_size=args.pa_buffer_size,
                 rb_optimize_memory=args.pa_rb_optimize_memory, gamma=0.99, tau=1.0, target_network_frequency=500,
                 batch_size=args.pa_batch_size, start_e=1.0, end_e=0.05, exploration_fraction=1.0, train_frequency=10,
                 write_frequency=100, save_folder=f"{args.save_folder}/{env.unwrapped.spec.id}") for env in envs]

    agent = CenRA_dis(policy_agents=policy_agents, sample_env=envs[0], actor_class=RAActorWorld,
                      critic_class=RAQNetMiniWorld, buffer_size=args.pa_buffer_size * len(env_ids),
                      batch_size=args.ra_batch_size, policy_lr=args.ra_actor_lr, q_lr=args.ra_critic_lr,
                      alpha_lr=args.ra_alpha_lr, policy_frequency=args.ra_policy_frequency,
                      alpha=args.ra_alpha, alpha_autotune=args.ra_alpha_autotune,
                      suggested_reward_scale=args.suggested_reward_scale, lamb=args.lamb)

    agent.learn(total_timesteps=args.total_timesteps, pa_learning_starts=args.pa_learning_starts,
                ra_learning_starts=args.ra_learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
