"""
The networks for CenRA-MTRL framework.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


###### For Policy Agent Networks ######

class QNetMiniGrid(nn.Module):
    """
    The Q network for the minigrid environment.
    The observation space is usually a matrix.
    The action space is a discrete vector.

    The structure is referred to the MiniGrid Documentation.
    """

    def __init__(self, env):
        super().__init__()
        # Assume observation_space is a gym Space with shape (channels, height, width)
        n_input_channels = env.observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat features size by doing a forward pass through cnn with a dummy input
        with torch.no_grad():
            dummy_input = torch.as_tensor(env.observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.network = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        return self.network(cnn_features)


class QNetMiniWorld(nn.Module):
    """
    The Q network for the 3D miniworld environment.
    """

    def __init__(self, env):
        super(QNetMiniWorld, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(env.observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.network = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        cnn_features = self.cnn(x / 255.0)
        return self.network(cnn_features)


class VectorActor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class VectorQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


###### For Reward Agent Networks ######

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x


class RAQNetVectorObs(nn.Module):
    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RAActorVectorObs(nn.Module):

    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class RAQNetMiniGrid(nn.Module):
    def __init__(self, ra_obs_space, suggested_reward_space, block_num=2):
        super().__init__()

        # for the minigrid 3d matrix observation
        n_input_channels = ra_obs_space['observation'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(ra_obs_space['observation'].sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.obs_fc1 = nn.Linear(n_flatten, 1000)

        # for the action vector
        # self.action_fc1 = nn.Linear(ra_obs_space['action'].n, 128)
        self.action_fc1 = nn.Linear(1, 128)

        # for the suggested reward
        self.sug_reward_fc1 = nn.Linear(np.prod(suggested_reward_space.shape), 128)

        # combine the features
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256 + 1000, 256 + 1000) for _ in range(block_num)])
        self.fc2 = nn.Linear(256 + 1000, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x_obs, x_act, a):
        x_obs = self.cnn(x_obs)
        x_obs = self.obs_fc1(x_obs)
        x_act = F.relu(self.action_fc1(x_act.float()))
        a = F.relu(self.sug_reward_fc1(a))
        x = torch.cat((x_obs, x_act, a), dim=1)

        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RAActorMiniGrid(nn.Module):
    def __init__(self, ra_obs_space, suggested_reward_space, block_num=2):
        super().__init__()
        # For the minigrid 3d matrix observation
        n_input_channels = ra_obs_space['observation'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(ra_obs_space['observation'].sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.obs_fc1 = nn.Linear(n_flatten, 1000)

        # For the action vector
        # self.action_fc1 = nn.Linear(ra_obs_space['action'].n, 128)
        self.action_fc1 = nn.Linear(1, 128)

        # Combine the features
        self.hidden_blocks = nn.ModuleList([ResidualBlock(128 + 1000, 128 + 1000) for _ in range(block_num)])
        self.fc2 = nn.Linear(128 + 1000, 256)
        self.fc_mean = nn.Linear(256, np.prod(suggested_reward_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(suggested_reward_space.shape))

        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((suggested_reward_space.high - suggested_reward_space.low) / 2.0,
                                          dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((suggested_reward_space.high + suggested_reward_space.low) / 2.0,
                                          dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x_obs, x_act):
        x_obs = self.cnn(x_obs)
        x_obs = self.obs_fc1(x_obs)
        x_act = F.relu(self.action_fc1(x_act.float()))
        x = torch.cat((x_obs, x_act), dim=1)

        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x_obs, x_act):
        mean, log_std = self(x_obs, x_act)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class RAQNetMiniWorld(nn.Module):
    def __init__(self, ra_obs_space, suggested_reward_space, block_num=2):
        super().__init__()

        # for the minigrid 3d matrix observation
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(ra_obs_space['observation'].sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.obs_fc1 = nn.Linear(n_flatten, 1000)

        # for the action vector
        # self.action_fc1 = nn.Linear(ra_obs_space['action'].n, 128)
        self.action_fc1 = nn.Linear(1, 128)

        # for the suggested reward
        self.sug_reward_fc1 = nn.Linear(np.prod(suggested_reward_space.shape), 128)

        # combine the features
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256 + 1000, 256 + 1000) for _ in range(block_num)])
        self.fc2 = nn.Linear(256 + 1000, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x_obs, x_act, a):
        x_obs = self.cnn(x_obs / 255.0)
        x_obs = self.obs_fc1(x_obs)
        x_act = F.relu(self.action_fc1(x_act.float()))
        a = F.relu(self.sug_reward_fc1(a))
        x = torch.cat((x_obs, x_act, a), dim=1)

        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RAActorWorld(nn.Module):
    def __init__(self, ra_obs_space, suggested_reward_space, block_num=2):
        super().__init__()
        # For the minigrid 3d matrix observation
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(ra_obs_space['observation'].sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.obs_fc1 = nn.Linear(n_flatten, 1000)

        # For the action vector
        # self.action_fc1 = nn.Linear(ra_obs_space['action'].n, 128)
        self.action_fc1 = nn.Linear(1, 128)

        # Combine the features
        self.hidden_blocks = nn.ModuleList([ResidualBlock(128 + 1000, 128 + 1000) for _ in range(block_num)])
        self.fc2 = nn.Linear(128 + 1000, 256)
        self.fc_mean = nn.Linear(256, np.prod(suggested_reward_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(suggested_reward_space.shape))

        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((suggested_reward_space.high - suggested_reward_space.low) / 2.0,
                                          dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((suggested_reward_space.high + suggested_reward_space.low) / 2.0,
                                          dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x_obs, x_act):
        x_obs = self.cnn(x_obs / 255.0)
        x_obs = self.obs_fc1(x_obs)
        x_act = F.relu(self.action_fc1(x_act.float()))
        x = torch.cat((x_obs, x_act), dim=1)

        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x_obs, x_act):
        mean, log_std = self(x_obs, x_act)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
