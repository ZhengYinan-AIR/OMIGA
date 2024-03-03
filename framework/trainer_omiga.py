import math
import logging
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from tensorboardX.writer import SummaryWriter
from framework.utils import load_data
from framework.buffer import SequentialDataset
from torch.distributions import Categorical
import wandb

logger = logging.getLogger(__name__)


class TrainerConfig:
    gamma = 0.99
    optimizer = "Adam"
    tau = 0.005
    grad_norm_clip = 1.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.writter = SummaryWriter(config.log_dir)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

        self.v_param = list(self.raw_model.v.parameters())
        self.q_param = list(self.raw_model.q.parameters()) + list(self.raw_model.q_mix_model.parameters())
        self.actor_param = list(self.raw_model.actor.parameters())

        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=config.learning_rate)
        self.v_optimizer = torch.optim.Adam(self.v_param, lr=config.learning_rate)
        self.q_optimizer = torch.optim.Adam(self.q_param, lr=config.learning_rate)
        self.target_model = copy.deepcopy(self.raw_model)
        self.target_model.train(False)
        self.global_step = 0

        self.alpha = 10.0


    def soft_update_target(self):
        for param, target_param in zip(self.raw_model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def train(self, episodes, data_dir, num_agents,
              rollout_worker=None, eval_env=None, eval_episodes=None, eval_interval=None):

        # episodes
        raw_model, config = self.raw_model, self.config
        raw_model.train(True)

        def run_epoch(dataset):
            loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size, drop_last=True,
                                num_workers=self.config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader))
            logger.info("***** Trainging Begin ******")

            for it, (s, o, pre_a, r, t, s_next, o_next, cur_a, next_ava) in pbar:
                # [step_size, context_length, num_agents, data_dim]
                s = s.to(self.device)
                o = o.to(self.device)
                r = r.to(self.device)
                s_next = s_next.to(self.device)
                o_next = o_next.to(self.device)
                cur_a = cur_a.to(self.device)
                next_ava = next_ava.to(self.device)
                done = s.eq(s_next.data)
                done = done.min(-1)[0].unsqueeze(-1).detach().long()

                one_hot_agent_id = torch.eye(config.num_agents).expand(o.shape[0], o.shape[1], -1, -1)
                if torch.cuda.is_available():
                    one_hot_agent_id = one_hot_agent_id.cuda()
                o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)

                q_eval = torch.stack([self.raw_model.q(o_with_id[:, :, j, :]) for j in range(config.num_agents)], dim=2)
                current_q = q_eval.gather(-1, cur_a)
                w_q, b_q = self.raw_model.q_mix_model(s)
                q_total = (w_q * current_q).sum(dim=-2) + b_q.squeeze(dim=-1)

                o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
                v_next = torch.stack([self.target_model.v(o_next_with_id[:, :, j, :]) for j in range(config.num_agents)], dim=2)
                w_next, b_next = self.target_model.q_mix_model(s_next)
                v_next_total = (w_next * v_next).sum(dim=-2) + b_next.squeeze(dim=-1)
                expected_q_total = r[:, :, 0, :] + config.gamma * (1 - done.min(2)[0]) * v_next_total.detach()
                q_loss = ((q_total - expected_q_total.detach())**2).mean()

                target_q = torch.stack([self.target_model.q(o_with_id[:, :, j, :]) for j in range(config.num_agents)], dim=2)
                target_q = target_q.gather(-1, cur_a)
                target_w_q, target_b_q = self.target_model.q_mix_model(s)
                v = torch.stack([self.raw_model.v(o_with_id[:, :, j, :]) for j in range(config.num_agents)], dim=2)

                z = 1 / self.alpha * (target_w_q.detach() * target_q.detach() - target_w_q.detach() * v)
                z = torch.clamp(z, min=-10.0, max=10.0)
                max_z = torch.max(z)
                if torch.cuda.is_available():
                    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).cuda(), max_z)
                else:
                    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0), max_z)
                max_z = max_z.detach()

                v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * target_w_q.detach() * v / self.alpha
                v_loss = v_loss.mean()

                exp_a = torch.exp(z).detach().squeeze(-1)
                action_logits = torch.stack([self.raw_model.actor(o_with_id[:, :, j, :]) for j in range(config.num_agents)], dim=2)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(cur_a.squeeze(-1))
                actor_loss = -(exp_a * log_probs).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_param, config.grad_norm_clip)
                self.actor_optimizer.step()

                self.q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_param, config.grad_norm_clip)
                self.q_optimizer.step()

                self.v_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_param, config.grad_norm_clip)
                self.v_optimizer.step()

                self.global_step += 1
                self.soft_update_target()

        for epoch in range(config.max_epochs):
            bias = 0
            num_step = 20
            for i in range(num_step):
                global_states, local_obss, actions, done_idxs, rewards, time_steps, next_global_states, next_local_obss, \
                next_available_actions, _ = load_data(int(episodes/num_step), bias, data_dir, n_agents=num_agents)
                offline_dataset = SequentialDataset(1, global_states, local_obss, actions, done_idxs, rewards,
                                                    time_steps,
                                                    next_global_states, next_local_obss, next_available_actions)
                run_epoch(offline_dataset)
                bias += int(episodes/num_step)

        self.raw_model = raw_model

