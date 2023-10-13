import numpy as np
import torch
import h5py

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, n_agents, env_name, data_dir, max_size=int(2e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        self.env_name = env_name
        self.data_dir = data_dir

        self.o = np.zeros((max_size, n_agents, state_dim))
        self.s = np.zeros((max_size, n_agents, state_dim))
        self.a = np.zeros((max_size, n_agents, action_dim))
        self.r = np.zeros((max_size, 1))
        self.mask = np.zeros((max_size, 1))
        self.s_next = np.zeros((max_size, n_agents, state_dim))
        self.o_next = np.zeros((max_size, n_agents, state_dim))
        self.a_next = np.zeros((max_size, n_agents, state_dim))
        self.device = torch.device(device)

    def sample(self, batch_size):
        o_size = self.o.shape[0]
        ind = np.random.randint(0, o_size, size=batch_size)  
        return (
            torch.FloatTensor(self.o[ind]).to(self.device),
            torch.FloatTensor(self.s[ind]).to(self.device),
            torch.FloatTensor(self.a[ind]).to(self.device),
            torch.FloatTensor(self.r[ind]).to(self.device),  
            torch.FloatTensor(self.mask[ind]).to(self.device),
            torch.FloatTensor(self.s_next[ind]).to(self.device),
            torch.FloatTensor(self.o_next[ind]).to(self.device),
            torch.FloatTensor(self.a_next[ind]).to(self.device)
        )
    
    def load(self):
        print('==========Data loading==========')
        data_file = self.data_dir + self.env_name + '.hdf5'
        # data_file = self.data_dir + 'test.hdf5'
        print('Loading from:', data_file)
        f = h5py.File(data_file, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        data_size = s.shape[0]
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(d[:,0]),
                np.arange(data_size) < data_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), data_size))

        self.o = o[nonterminal_steps]
        self.s = s[nonterminal_steps]
        self.a = a[nonterminal_steps]
        self.r = r[nonterminal_steps].reshape(-1, 1)
        self.mask = 1 - d[nonterminal_steps + 1].reshape(-1, 1)
        self.s_next = s[nonterminal_steps + 1]
        self.o_next = o[nonterminal_steps + 1]
        self.a_next = a[nonterminal_steps + 1]
        self.size = self.o.shape[0]