import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network import Actor, V_critic, Q_critic, MixNet

class OMIGA(object):
    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._alpha = 10
        self._gamma = config['gamma']
        self._tau = config['tau']
        self._hidden_sizes = config['hidden_sizes']
        self._mix_hidden_sizes = config['mix_hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._grad_norm_clip = config['grad_norm_clip']
        self._num_agent = num_agent
        self._device = config['device']
        self._eval_env = eval_env
        self._iteration = 0
        self._optimizers = dict()

        # v-network
        self._v_network = V_critic(observation_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._v_target_network = copy.deepcopy(self._v_network)
        self._optimizers['v'] = torch.optim.Adam(self._v_network.parameters(), self._lr)

        # q-network and mix-network
        self._q_network = Q_critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(observation_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)

        # policy-network
        self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
    
    def q_loss(self, o_with_a_id, s, o_next_with_id, s_next, r, mask, result={}):
        q_values = self._q_network(o_with_a_id)
        w, b = self._mix_network(s)
        q_total = (w * q_values).sum(dim=-2) + b.squeeze(dim=-1)

        v_next_target_values = self._v_target_network(o_next_with_id)
        w_next_target, b_next_target = self._mix_target_network(s_next) 
        v_next_target_total = (w_next_target * v_next_target_values).sum(dim=-2) + b_next_target.squeeze(dim=-1)

        expected_q_total = r+ self._gamma * mask * v_next_target_total.detach()
        q_loss = ((q_total - expected_q_total.detach())**2).mean()

        result.update({
            'q_loss': q_loss,
            'expected_q_total': expected_q_total.mean(),
            'q_total': q_total.mean(),
            'w1': w[:,0,:].mean(),
            'w2': w[:,1,:].mean(),
            'b': b.mean(),
            'q_values1': q_values[:,0,:].mean(),
            'q_values2': q_values[:,1,:].mean(),
        })

        return result
    
    def v_loss(self, z, w_target, v_values, result={}):
        max_z = torch.max(z)
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self._device), max_z)
        max_z = max_z.detach()

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha
        v_loss = v_loss.mean()

        result.update({
            'v_loss': v_loss,
        })
        return result

    def policy_loss(self, exp_a, a, o_with_id, result={}):
        log_probs = self._policy_network.get_log_density(o_with_id, a)
        policy_loss = -(exp_a * log_probs).mean()

        result.update({
            'policy_loss': policy_loss,
        })
        return result
    
    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        # Shared network values
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((o, a, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)

        # q_loss
        loss_result = self.q_loss(o_with_a_id, s, o_next_with_id, s_next, r, mask, result={})

        # v and policy shared values
        q_target_values = self._q_target_network(o_with_a_id)
        w_target, b_target = self._mix_target_network(s)

        v_values = self._v_network(o_with_id)

        z = 1/self._alpha * (w_target.detach() * q_target_values.detach() - w_target.detach() * v_values)
        z = torch.clamp(z, min=-10.0, max=10.0)

        exp_a = torch.exp(z).detach().squeeze(-1)

        # v_loss
        loss_result = self.v_loss(z, w_target.detach(), v_values, result=loss_result)
        # policy_loss
        loss_result = self.policy_loss(exp_a, a, o_with_id, result=loss_result)

        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers['policy'].step()
        
        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers['q'].step()

        self._optimizers['v'].zero_grad()
        loss_result['v_loss'].backward()
        nn.utils.clip_grad_norm_(self._v_network.parameters(), self._grad_norm_clip)
        self._optimizers['v'].step()

        # soft update
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._v_network.parameters(), self._v_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        self._iteration += 1

        loss_result.update({
            'v_values1': v_values[:,0,:].mean(),
            'v_values2': v_values[:,0,:].mean(),
            'q_target_values1': q_target_values[:,0,:].mean(),
            'q_target_values2': q_target_values[:,1,:].mean(),
        })
        return loss_result

    def step(self, o):
        o = torch.from_numpy(o).to(self._device)
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        action = self._policy_network.get_deterministic_action(o_with_id)

        return action.detach().cpu()