import torch
import numpy as np
from torch.utils.data import Dataset


def list2array(input_list):
    return np.array(input_list)


class SequentialDataset(Dataset):

    def __init__(self, context_length, states, obss, actions, done_idxs, rewards, timesteps, next_states,
                 next_obss, next_available_actions):
        self.context_length = context_length
        self.states = states
        self.obss = obss
        self.next_states = next_states
        self.next_obss = next_obss
        self.actions = actions
        self.next_available_actions = next_available_actions
        # done_idx - 1 equals the last step's position
        self.done_idxs = done_idxs
        self.rewards = rewards
        self.timesteps = timesteps

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        context_length = self.context_length
        done_idx = idx + context_length
        for i in np.array(self.done_idxs)[:, 0].tolist():
            if i > idx:  
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32)
        next_states = torch.tensor(np.array(self.next_states[idx:done_idx]), dtype=torch.float32)
        obss = torch.tensor(np.array(self.obss[idx:done_idx]), dtype=torch.float32)
        next_obss = torch.tensor(np.array(self.next_obss[idx:done_idx]), dtype=torch.float32)

        if idx == 0 or idx - 1 in self.done_idxs:
            padding = list(np.zeros_like(self.actions[idx]))
            pre_actions = [padding] + self.actions[idx:done_idx - 1]
            pre_actions = torch.tensor(pre_actions, dtype=torch.int64)
        else:
            pre_actions = torch.tensor(self.actions[idx - 1:done_idx - 1], dtype=torch.int64)
        cur_actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.int64)
        next_available_actions = torch.tensor(self.next_available_actions[idx:done_idx], dtype=torch.int64)

        # actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(-1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64)

        return states, obss, pre_actions, rewards, timesteps, next_states, next_obss, cur_actions, \
               next_available_actions

class ExpertDataSet(Dataset):
    def __init__(self, expert_data):
        self.state_size = np.shape(expert_data[0])[0]
        # self.expert_data = np.array(pd.read_csv(data_set_path))
        self.state = torch.tensor(torch.from_numpy(expert_data[0]), dtype=torch.float32)
        self.action = torch.tensor(torch.from_numpy(np.array(expert_data[1])), dtype=torch.float32)
        self.next_state = torch.tensor(torch.from_numpy(expert_data[0]), dtype=torch.float32)  # as the current state
        self.length = self.state_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]


class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx]), dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.long)
        rtgs = torch.tensor(self.rtgs[idx], dtype=torch.float32).unsqueeze(-1)
        timesteps = torch.tensor(self.timesteps[idx], dtype=torch.int64)

        return states, actions, rtgs, timesteps


class ReplayBuffer:
    def __init__(self, n_agents, buffer_size, context_length):
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.context_length = context_length

        self.data = []
        self.episode = [[] for i in range(self.n_agents)]

    @property
    def cur_size(self):
        return len(self.data)

    def insert(self, global_obs, local_obs, action, reward, done, available_actions):
        for i in range(self.n_agents):
            step = [global_obs[0][i], local_obs[0][i], [action[i]], reward[0][i], done[0][i], available_actions[0][i]]
            self.episode[i].append(step)
        if np.all(done):
            if self.cur_size >= self.buffer_size:
                del_count = self.cur_size - self.buffer_size + 1
                del self.data[:del_count]
            self.data.append(self.episode.copy())
            self.episode = [[] for i in range(self.n_agents)]

    def reset(self):
        self.data = []