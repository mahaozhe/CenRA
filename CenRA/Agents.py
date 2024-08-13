"""
The script for distributed policy agents.
"""

import gymnasium as gym

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time


class DQNAgent:
    """
    The Deep Q-Network (DQN) agent.
    """

    def __init__(self, env, q_network_class, exp_name="CenRA_dqn", seed=1, cuda=0, learning_rate=2.5e-4,
                 buffer_size=10000, rb_optimize_memory=False, gamma=0.99, tau=1., target_network_frequency=500,
                 batch_size=128, start_e=1, end_e=0.05, exploration_fraction=0.5, train_frequency=10,
                 write_frequency=100, save_folder="./CenRA_dqn/"):
        """
        Initialize the DQN algorithm.
        :param env: the gymnasium-based environment
        :param q_network_class: the agent class
        :param exp_name: the experiment name
        :param seed: the random seed
        :param cuda: whether to use cuda
        :param learning_rate: the learning rate
        :param buffer_size: the replay memory buffer size
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param gamma: the discount factor gamma
        :param tau: the target network update rate
        :param target_network_frequency: the timesteps it takes to update the target network
        :param batch_size: the batch size of sample from the reply memory
        :param start_e: the starting epsilon for exploration
        :param end_e: the ending epsilon for exploration
        :param exploration_fraction: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param train_frequency: the frequency of training
        :param write_frequency: the frequency of writing to tensorboard
        :param save_folder: the folder to save the model
        """

        self.exp_name = exp_name

        self.seed = seed

        # set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env

        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported for DQN"

        # the networks
        self.q_network = q_network_class(self.env).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = q_network_class(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False
        )

        self.gamma = gamma

        # for the epsilon greedy exploration
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction

        # for the batch training
        self.batch_size = batch_size

        # for the target network update
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # for the training
        self.train_frequency = train_frequency

        # * for the tensorboard writer
        run_name = f"{exp_name}-{seed}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def linear_schedule(self, duration, t):
        """
        Linear interpolation between start_e and end_e
        :param duration: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param t: the current timestep
        """
        slope = (self.end_e - self.start_e) / duration
        return max(slope * t + self.start_e, self.end_e)

    def reset(self, total_timesteps, global_step):
        """
        The function to reset the agent.
        """
        obs, _ = self.env.reset(seed=self.seed)

        epsilon = self.linear_schedule(self.exploration_fraction * total_timesteps, global_step)

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            q_value = self.q_network(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
            action = torch.argmax(q_value, dim=1).cpu().numpy()

        return obs, action

    def step(self, obs, action, total_timesteps, global_step):
        """
        The function for the agent to take one step to interact with the environment.
        """

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.replay_buffer.add(obs, next_obs, action, reward, done, info)

        epsilon = self.linear_schedule(self.exploration_fraction * total_timesteps, global_step)

        if random.random() < epsilon:
            next_action = self.env.action_space.sample()
        else:
            q_value = self.q_network(torch.Tensor(np.expand_dims(next_obs, axis=0)).to(self.device))
            next_action = torch.argmax(q_value, dim=1).cpu().numpy()

        if "episode" in info:
            self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        return next_obs, next_action, reward, done, info

    def optimize(self, global_step, reward_agent, lamb=0.2):
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1)

            # + generate suggested rewards
            reward_sug, _, _ = reward_agent.get_action(data.observations, data.actions)

            td_target = data.rewards.flatten() + lamb * reward_sug.squeeze() + self.gamma * target_max * (
                    1 - data.dones.flatten())
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # * update q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # * update target network
        if global_step % self.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                             self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data)

    def save(self, indicator="best"):
        torch.save(self.target_network.state_dict(),
                   os.path.join(self.save_folder,
                                f"q_network-{self.exp_name}-{indicator}-{self.seed}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}.pth"))


class SACAgent:
    """
    The Soft Actor-Critic (SAC) agent.
    """

    def __init__(self, env, actor_class, critic_class, exp_name="sac", seed=1, cuda=0, gamma=0.99, buffer_size=1000000,
                 rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=1e-3, eps=1e-8, alpha_lr=1e-4,
                 target_network_frequency=1, tau=0.005, policy_frequency=2, alpha=0.2, alpha_autotune=True,
                 write_frequency=100, save_folder="./sac/"):
        """
        Initialize the SAC algorithm.
        :param env: the gymnasium-based environment
        :param actor_class: the actor class
        :param critic_class: the critic class
        :param exp_name: the name of the experiment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param buffer_size: the size of the replay buffer
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param batch_size: the batch size
        :param policy_lr: the learning rate of the policy network
        :param q_lr: the learning rate of the Q network
        :param eps: the epsilon for the Adam optimizer
        :param alpha_lr: the learning rate of the temperature parameter
        :param tau: the soft update coefficient
        :param policy_frequency: the policy update frequency
        :param target_network_frequency: the target network update frequency
        :param alpha: the temperature parameter
        :param alpha_autotune: whether to autotune the temperature parameter
        :param write_frequency: the write frequency
        :param save_folder: the folder to save the model
        """

        self.exp_name = exp_name

        self.seed = seed

        # set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env

        # initialize the actor and critic networks
        self.actor = actor_class(self.env.observation_space, self.env.action_space).to(self.device)
        self.qf_1 = critic_class(self.env.observation_space, self.env.action_space).to(self.device)
        self.qf_2 = critic_class(self.env.observation_space, self.env.action_space).to(self.device)
        self.qf_1_target = critic_class(self.env.observation_space, self.env.action_space).to(self.device)
        self.qf_2_target = critic_class(self.env.observation_space, self.env.action_space).to(self.device)

        # copy the parameters of the critic networks to the target networks
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        # initialize the optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr, eps=eps)
        self.q_optimizer = optim.Adam(list(self.qf_1.parameters()) + list(self.qf_2.parameters()), lr=q_lr, eps=eps)

        # initialize the temperature parameter
        self.alpha_autotune = alpha_autotune
        if alpha_autotune:
            # set the target entropy as the negative of the action space dimension
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # + modify the observation space to be float32
        self.env.observation_space.dtype = np.float32
        # initialize the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # * for the tensorboard writer
        run_name = f"{exp_name}-{seed}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def reset(self, learning_starts, global_step):
        obs, _ = self.env.reset()

        if global_step < learning_starts:
            action = self.env.action_space.sample()
        else:
            action, _, _ = self.actor.get_action(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
            action = action.detach().cpu().numpy()[0]

        return obs, action

    def step(self, obs, action, learning_starts, global_step):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.replay_buffer.add(obs, next_obs, action, reward, done, info)

        if global_step < learning_starts:
            next_action = self.env.action_space.sample()
        else:
            next_action, _, _ = self.actor.get_action(torch.Tensor(np.expand_dims(next_obs, axis=0)).to(self.device))
            next_action = next_action.detach().cpu().numpy()[0]

        if "episode" in info:
            self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        return next_obs, next_action, reward, done, info

    def optimize(self, global_step, reward_agent, lamb=0.2):
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf_1_next_target = self.qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi

            # + add the suggested reward
            obs_ra = torch.cat((data.observations, data.actions), dim=1)
            reward_sug, _, _ = reward_agent.get_action(obs_ra)

            next_q_value = data.rewards.flatten() + lamb * reward_sug.squeeze() + (
                    1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf_1_pi = self.qf_1(data.observations, pi)
                qf_2_pi = self.qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            if self.alpha_autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, f"actor-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, f"qf_1-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, f"qf_2-{self.exp_name}-{indicator}-{self.seed}.pth"))
