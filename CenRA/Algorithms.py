"""
The script for the Centralized Reward Agent.
"""

import gymnasium as gym

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer

import os
import random
import datetime
import time


class CenRA_dis:
    """
    The Centralized Reward Agent (CenRA) for discrete control, e.g., DQN problems.
    """

    def __init__(self, policy_agents, sample_env, actor_class, critic_class, exp_name="CenRA_dis", seed=1, cuda=0,
                 gamma=0.99, buffer_size=1000000, rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=1e-3,
                 eps=1e-8, alpha_lr=1e-4, target_network_frequency=1, tau=0.005, policy_frequency=2, alpha=0.2,
                 alpha_autotune=True, suggested_reward_scale=1, lamb=0.2, save_frequency=1000,
                 save_folder="./CenRA_dis/"):
        """
        Initialize the SAC algorithm.
        :param policy_agents: the policy agents with their respective environments
        :param sample_env: the sample environment
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
        :param suggested_reward_scale: the scale of the suggested reward space
        :param lamb: lambda, the shaped reward weight factor
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

        self.policy_agents = policy_agents
        self.num_pas = len(policy_agents)

        # + create the suggested reward space
        self.suggested_reward_space = gym.spaces.Box(low=-suggested_reward_scale, high=suggested_reward_scale,
                                                     shape=(1,), dtype=np.float32, seed=seed)

        # + create the observation space for the reward agent
        self.ra_obs_space = gym.spaces.Dict({
            'observation': sample_env.observation_space,
            'action': sample_env.action_space
        })

        # initialize the replay buffer
        self.replay_buffer = DictReplayBuffer(
            buffer_size,
            self.ra_obs_space,
            self.suggested_reward_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        # initialize the actor and critic networks for the Reward Agent
        self.actor = actor_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_1 = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_2 = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_1_target = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_2_target = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)

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
            self.target_entropy = -torch.prod(torch.Tensor(self.suggested_reward_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        self.lamb = lamb
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        self.save_folder = save_folder
        self.save_frequency = save_frequency
        os.makedirs(self.save_folder, exist_ok=True)

    def learn(self, total_timesteps=1000000, pa_learning_starts=10000, ra_learning_starts=5000):

        obs_ra_dict = [{'observation': obs, 'action': action} for obs, action in
                       (pa.reset(total_timesteps, 0) for pa in self.policy_agents)]

        for global_step in range(total_timesteps):
            if global_step < ra_learning_starts:
                reward_sug = [self.suggested_reward_space.sample() for _ in range(self.num_pas)]
            else:
                # + get a suggested reward from the RA, stack the obs and action as obs
                env_obs_batch = torch.stack([torch.tensor(obs['observation']) for obs in obs_ra_dict]).to(self.device)
                env_act_batch = torch.stack([torch.tensor(obs['action'].item()) for obs in obs_ra_dict]).unsqueeze(
                    1).to(self.device)
                reward_sug, _, _ = self.actor.get_action(env_obs_batch, env_act_batch)
                reward_sug = reward_sug.detach().cpu().numpy()

            for p in range(self.num_pas):
                next_obs, next_action, reward_env, done, info = self.policy_agents[p].step(
                    obs_ra_dict[p]['observation'], obs_ra_dict[p]['action'], total_timesteps, global_step)
                next_obs_ra_dict_one = {'observation': next_obs, 'action': next_action}
                # store the transition to the RA replay buffer
                self.replay_buffer.add(obs_ra_dict[p], next_obs_ra_dict_one, reward_sug[p][0], reward_env, done, info)

                if done:
                    # if the specific environment is done, reset it
                    obs_new, action_new = self.policy_agents[p].reset(total_timesteps, global_step)
                    # update the observation and action for the policy agents
                    obs_ra_dict[p] = {'observation': obs_new, 'action': action_new}
                else:
                    obs_ra_dict[p] = next_obs_ra_dict_one

            if global_step > pa_learning_starts:
                for pa in self.policy_agents:
                    pa.optimize(global_step, self.actor, self.lamb)

            if global_step > ra_learning_starts:
                self.optimize(global_step)

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations['observation'],
                                                                             data.next_observations['action'])
            qf_1_next_target = self.qf_1_target(data.next_observations['observation'], data.next_observations['action'],
                                                next_state_actions)
            qf_2_next_target = self.qf_2_target(data.next_observations['observation'], data.next_observations['action'],
                                                next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(
                -1)

        qf_1_a_values = self.qf_1(data.observations['observation'], data.observations['action'], data.actions).view(-1)
        qf_2_a_values = self.qf_2(data.observations['observation'], data.observations['action'], data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations['observation'], data.observations['action'])
                qf_1_pi = self.qf_1(data.observations['observation'], data.observations['action'], pi)
                qf_2_pi = self.qf_2(data.observations['observation'], data.observations['action'], pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations['observation'],
                                                             data.observations['action'])
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

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, f"ra-actor-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, f"ra-qf_1-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, f"ra-qf_2-{self.exp_name}-{indicator}-{self.seed}.pth"))


class CenRA_con(CenRA_dis):
    """
    The Centralized Reward Agent (CenRA) for continuous control, e.g., SAC problems.


    In this case, the observation and action can be combined as a single tensor.
    """

    def __init__(self, policy_agents, sample_env, actor_class, critic_class, exp_name="CenRA_con", seed=1, cuda=0,
                 gamma=0.99, buffer_size=1000000, rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=1e-3,
                 eps=1e-8, alpha_lr=1e-4, target_network_frequency=1, tau=0.005, policy_frequency=2, alpha=0.2,
                 alpha_autotune=True, suggested_reward_scale=1, lamb=0.2, save_frequency=1000,
                 save_folder="./CenRA_con/"):

        self.exp_name = exp_name

        self.seed = seed

        # set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.policy_agents = policy_agents
        self.num_pas = len(policy_agents)

        # + create the suggested reward space
        self.suggested_reward_space = gym.spaces.Box(low=-suggested_reward_scale, high=suggested_reward_scale,
                                                     shape=(1,), dtype=np.float32, seed=seed)

        # + create the observation space for the reward agent

        self.ra_obs_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(
                                               sample_env.observation_space.shape[0] + sample_env.action_space.shape[
                                                   0],),
                                           dtype=np.float32, seed=seed)

        # initialize the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.ra_obs_space,
            self.suggested_reward_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        # initialize the actor and critic networks for the Reward Agent
        self.actor = actor_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_1 = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_2 = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_1_target = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)
        self.qf_2_target = critic_class(self.ra_obs_space, self.suggested_reward_space).to(self.device)

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
            self.target_entropy = -torch.prod(torch.Tensor(self.suggested_reward_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        self.lamb = lamb
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        self.save_folder = save_folder
        self.save_frequency = save_frequency
        os.makedirs(self.save_folder, exist_ok=True)

    def learn(self, total_timesteps=1000000, pa_learning_starts=10000, ra_learning_starts=5000):

        obs_ra_dict = [{'observation': obs, 'action': action} for obs, action in
                       (pa.reset(pa_learning_starts, 0) for pa in self.policy_agents)]

        for global_step in range(total_timesteps):

            obs_ra_list = [np.hstack((d['observation'], d['action'])) for d in obs_ra_dict]

            if global_step < ra_learning_starts:
                reward_sug = [self.suggested_reward_space.sample() for _ in range(self.num_pas)]
            else:
                # + get a suggested reward from the RA, stack the obs and action as obs
                obs_ra_batch = torch.tensor(np.vstack(obs_ra_list)).to(self.device)
                reward_sug, _, _ = self.actor.get_action(obs_ra_batch)
                reward_sug = reward_sug.detach().cpu().numpy()

            for p in range(self.num_pas):
                next_obs, next_action, reward_env, done, info = self.policy_agents[p].step(
                    obs_ra_dict[p]['observation'], obs_ra_dict[p]['action'], pa_learning_starts, global_step)
                next_obs_ra_dict_one = {'observation': next_obs, 'action': next_action}
                # store the transition to the RA replay buffer
                self.replay_buffer.add(obs_ra_list[p], np.hstack((next_obs, next_action)), reward_sug[p][0], reward_env,
                                       done, info)

                if done:
                    # if the specific environment is done, reset it
                    obs_new, action_new = self.policy_agents[p].reset(pa_learning_starts, global_step)
                    # update the observation and action for the policy agents
                    obs_ra_dict[p] = {'observation': obs_new, 'action': action_new}
                else:
                    obs_ra_dict[p] = next_obs_ra_dict_one

            if global_step > pa_learning_starts:
                for pa in self.policy_agents:
                    pa.optimize(global_step, self.actor, self.lamb)

            if global_step > ra_learning_starts:
                self.optimize(global_step)

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf_1_next_target = self.qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(
                -1)

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
