import torch
import numpy as np
import torch.nn.functional as F

def sample(model, raw_model, state, agent_id, num_agent, device, available_actions=None, sample=False):
    
    model.eval()
    one_hot_agent_id = F.one_hot(torch.tensor(agent_id), num_classes=num_agent).unsqueeze(dim=0).unsqueeze(dim=0)
    if torch.cuda.is_available():
        one_hot_agent_id = one_hot_agent_id.cuda()

    state_with_id = torch.cat((state, one_hot_agent_id), dim=-1)
    q = model(state_with_id)
    if available_actions is not None:
        q[available_actions == 0] = -1e8
    next_action = q.argmax(-1, keepdim=False)
    
    return next_action


class RolloutWorker:

    def __init__(self, model, buffer, context_length=1):
        self.buffer = buffer
        self.model = model
        self.context_length = context_length
        self.device = 'cpu'
        self.raw_model = model

        if torch.cuda.is_available() and not isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).to(self.device)

    def rollout(self, env, ret, num_episodes, train=False, random_rate=1.0):
        self.model.train(False)

        T_rewards, T_wins = [], 0.
        for i in range(num_episodes):
            obs, share_obs, available_actions = env.real_env.reset()
            state = torch.from_numpy(obs).to(self.device)
            reward_sum = 0
            rtgs = np.ones((1, env.num_agents, 1)) * ret
            sampled_action = [
                sample(self.model, self.raw_model, state[:, j].unsqueeze(0),
                       agent_id=j,
                       num_agent=env.num_agents,
                       device=self.device,
                       available_actions=torch.from_numpy(available_actions)[:, j].unsqueeze(0),
                       sample=train,
                       )
                for j in range(env.num_agents)]

            actions = []
            all_states = state
            m = 0
            while True:
                action = [a.cpu().numpy()[0, -1] for a in sampled_action]
                actions += [action]

                cur_global_obs = share_obs
                cur_local_obs = obs
                cur_ava = available_actions

                obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step([action])

                if train:
                    self.buffer.insert(cur_global_obs, cur_local_obs, action, rewards, dones, cur_ava)

                reward_sum += np.mean(rewards)
                m += 1
                if np.all(dones):
                    T_rewards.append(reward_sum)
                    if infos[0][0]['won']:
                        T_wins += 1.
                    break

                state = torch.from_numpy(obs).to(self.device)
                all_states = torch.cat([all_states, state], dim=0)

                sampled_action = [
                    sample(self.model, self.raw_model, state[:, j].unsqueeze(0),
                           agent_id=j,
                           num_agent=env.num_agents,
                           device=self.device,
                           available_actions=torch.from_numpy(available_actions)[:, j].unsqueeze(0),
                           sample=train,
                           )
                    for j in range(env.num_agents)]

        aver_return = np.mean(T_rewards)
        aver_win_rate = T_wins / num_episodes
        self.model.train(True)
        return aver_return, aver_win_rate