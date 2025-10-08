"""
用numpy实现简易replay buffer
"""

import torch
import collections

exp_shape = collections.namedtuple(
    'exp_shape', ['state_shape', 'action_shape', 'reward_shape', 'done_shape']
)

class RB:
    def __init__(self, capacity, state_shape, device):
        self.cur_index = 0
        self.capacity = capacity
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)


    def add(self, state, action, reward, next_state, done):
        self.states[self.cur_index] = state
        self.actions[self.cur_index] = torch.tensor(action, dtype=torch.int64, device=self.actions.device)
        self.rewards[self.cur_index] = torch.tensor(reward, dtype=torch.float32, device=self.rewards.device)
        self.next_states[self.cur_index] = next_state
        self.dones[self.cur_index] = torch.tensor(done, dtype=torch.bool, device=self.dones.device)
        self.cur_index = (self.cur_index + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = torch.randint(0, self.capacity, (batch_size,), device=self.states.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

        
        


