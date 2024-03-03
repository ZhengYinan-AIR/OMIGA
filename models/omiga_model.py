import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, MultivariateNormal
from torch.nn import functional as F
from torch.distributions import Categorical
import copy

class OMIGAConfig:
    def __init__(
            self,
            num_actions,
            state_dim,
            obs_dim,
            num_agents,
            eval_eps=0.001
    ):

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.eval_eps = eval_eps


class OMIGA(nn.Module):
    def __init__(self, q_mix_model, v_mix_model, config):
        super(OMIGA, self).__init__()
        self.q_mix_model = q_mix_model
        self.v_mix_model = v_mix_model

        self.q = QNet(config)
        self.v = VNet(config)
        self.actor = Actor(config)


class VNet(nn.Module):
    def __init__(self, config):
        super(VNet, self).__init__()
        self.f1 = nn.Linear(config.obs_dim + config.num_agents, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, 1)

    def forward(self, obs):
        x = F.relu(self.f1(obs))
        h = F.relu(self.f2(x))
        v = self.f3(h)
        return v

class QNet(nn.Module):
    def __init__(self, config):
        super(QNet, self).__init__()
        self.f1 = nn.Linear(config.obs_dim + config.num_agents, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, config.num_actions)

    def forward(self, obs_with_id):
        x = F.relu(self.f1(obs_with_id))
        h = F.relu(self.f2(x))
        q = self.f3(h)
        return q


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(config.obs_dim + config.num_agents, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, config.num_actions)

    def forward(self, obs):
        x = F.relu(self.f1(obs))
        h = F.relu(self.f2(x))
        action_logits = self.f3(h)
        return action_logits


class MixNet(nn.Module):
    def __init__(self, state_shape, n_agents, hyper_hidden_dim, num_action):
        super(MixNet, self).__init__()
        self.state_shape = state_shape * n_agents  # concat state from agents
        self.n_agents = n_agents
        self.hyper_hidden_dim = hyper_hidden_dim
        self.num_action = num_action

        self.f_v = nn.Linear(self.state_shape, hyper_hidden_dim)
        self.w_v = nn.Linear(hyper_hidden_dim, n_agents)
        self.b_v = nn.Linear(hyper_hidden_dim, 1)


    def forward(self, states):
        batch_size = states.size(0)
        context_length = states.size(1)
        states = torch.cat([states[:, :, j, :] for j in range(self.n_agents)], dim=-1)
        states = states.reshape(-1, self.state_shape)

        x = self.f_v(states)
        w = self.w_v(F.relu(x)).reshape(batch_size, context_length, self.n_agents, 1)
        b = self.b_v(F.relu(x)).reshape(batch_size, context_length, 1, 1)

        return torch.abs(w), b


