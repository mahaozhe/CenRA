"""
The script to run CenRA framework on the MujocoCar environment.
"""

import argparse

from CenRA.Agents import SACAgent
from CenRA.Algorithms import CenRA_con

from CenRA.Networks import VectorActor, VectorQNetwork
from CenRA.Networks import RAActorVectorObs, RAQNetVectorObs

from CenRA.utils import car_navigation_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run CenRA framework on MujocoCar environment.")

    parser.add_argument("--exp-name", type=str, default="CenRA-MujocoCar")

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

    ###### For the Policy Agent (SAC-based agent) ######

    parser.add_argument("--pa-buffer-size", type=int, default=1000000)
    parser.add_argument("--pa-rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--pa-batch-size", type=int, default=256)

    parser.add_argument("--pa-actor-lr", type=float, default=3e-4)
    parser.add_argument("--pa-critic-lr", type=float, default=1e-3)
    parser.add_argument("--pa-alpha-lr", type=float, default=1e-4)

    parser.add_argument("--pa-policy-frequency", type=int, default=2)
    parser.add_argument("--pa-target-frequency", type=int, default=1)
    parser.add_argument("--pa-tau", type=float, default=0.005)

    parser.add_argument("--pa-alpha", type=float, default=0.2)
    parser.add_argument("--pa-alpha-autotune", type=bool, default=True)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=10000)
    parser.add_argument("--save-folder", type=str, default="./CenRA-MujocoCar/")

    ###### For the learning ######

    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--pa-learning-starts", type=int, default=1e4)
    parser.add_argument("--ra-learning-starts", type=int, default=5e3)

    args = parser.parse_args()
    return args


def run():
    import torch
    torch.cuda.empty_cache()

    args = parse_args()

    env_id = "SafetyRacecarGoal0-v0"

    configs = [{"agent_xy": [0, 0], "goal_xy": [2, 2]}, {"agent_xy": [1.5, 1.5], "goal_xy": [-0.5, -0.5]},
               {"agent_xy": [1.5, 0], "goal_xy": [-0.5, 0]}, {"agent_xy": [-1, 0.5], "goal_xy": [1.5, -0.5]}]
    # configs = [{"agent_xy": [0, 1.5], "goal_xy": [0, -0.5]}]  # for the new task

    env_names = ["Target1", "Target2", "Target3", "Target4"]

    envs = [car_navigation_env_maker(env_id=env_id, config=config, seed=args.seed, render=args.render) for config in
            configs]

    policy_agents = [
        SACAgent(env=envs[e], actor_class=VectorActor, critic_class=VectorQNetwork,
                 exp_name=f"{args.exp_name}-{env_names[e]}", seed=args.seed, cuda=args.cuda, gamma=0.99,
                 buffer_size=args.pa_buffer_size, rb_optimize_memory=args.pa_rb_optimize_memory,
                 batch_size=args.pa_batch_size, policy_lr=args.pa_actor_lr, q_lr=args.pa_critic_lr,
                 alpha_lr=args.pa_alpha_lr, target_network_frequency=args.pa_target_frequency, tau=args.pa_tau,
                 policy_frequency=args.pa_policy_frequency, alpha=args.pa_alpha, alpha_autotune=args.pa_alpha_autotune,
                 write_frequency=100, save_folder=f"{args.save_folder}/{env_names[e]}") for e in range(len(envs))]

    agent = CenRA_con(policy_agents=policy_agents, sample_env=envs[0], actor_class=RAActorVectorObs,
                      critic_class=RAQNetVectorObs, buffer_size=args.ra_buffer_size * len(envs),
                      batch_size=args.ra_batch_size, policy_lr=args.ra_actor_lr, q_lr=args.ra_critic_lr,
                      alpha_lr=args.ra_alpha_lr, policy_frequency=args.ra_policy_frequency,
                      alpha=args.ra_alpha, alpha_autotune=args.ra_alpha_autotune,
                      suggested_reward_scale=args.suggested_reward_scale, lamb=args.lamb)

    agent.learn(total_timesteps=args.total_timesteps, pa_learning_starts=args.pa_learning_starts,
                ra_learning_starts=args.ra_learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
