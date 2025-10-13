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

"""
优先级经验回放,用数组实现，采样时间复杂度是O(n)(用线段树实现较麻烦，本代码用于实验测试，故简化)
PER的要点如下：
    1. p_i: 除了s, a, r, s', done外，还需要维护每个样本的优先级p_i
    2. add(): 新加入的样本要设置优先级p_i，其值一般是当前样本中的最大优先级，第一个样本可设为1.0
    3. sample(): 
        a. 采样前先计算采样概率P(i)=p_i/sum(p);
        b. 然后random choice(N, B, P, replace=True)有放回采样, 得样本索引indices
        c. 返回值:
            I.（s,a,r,s',done):
            II. indices: 用于计算loss后，根据loss更新对应索引的优先级  
            III. weights: 重要性采样权重，w_i=(N*P_i)^(-beta)/max(w) (beta一般从0.4线性增至1.0) 
    4. update_prio(indices, loss): 根据索引更新对应样本的优先级p_i=(|loss|+eps)^alpha
        a. alpha决定了优先级对采样概率的影响程度，alpha=0时，PER退化为普通的均匀采样, 一般取值0.6
        b. eps是一个很小的正数，防止p_i=0
"""
class PrioRB(RB):
    def __init__(self, capacity, state_shape, device, alpha, beta, eps):
        super(PrioRB, self).__init__(capacity, state_shape, device)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.init_prio = 1.0
        self.prioritys = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rng = torch.Generator(device=device)
        self.full= False

    def add(self, state, action, reward, next_state, done):
        max_prio = max(self.init_prio, self.prioritys.max().item())
        self.prioritys[self.cur_index] = max_prio
        super().add(state, action, reward, next_state, done)
        self.full= True if self.cur_index == 0 else self.full

    def sample(self, batch_size):
        exp_num = self.capacity if self.full else self.cur_index
        prob = self.prioritys[:exp_num] / self.prioritys[:exp_num].sum()
        indices = torch.multinomial(prob, batch_size, replacement=True, generator=self.rng)
        weights = (self.capacity * prob[indices]) ** (-self.beta)
        weights /= weights.max()
        self.update_beta()
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                indices,
                weights)

    # 10000步线性增加到1.0
    def update_beta(self, increment=1e-4, max_beta=1.0):
        self.beta = min(max_beta, self.beta + increment)

    def update_prio(self, loss, indices):
        prios = (loss.abs() + self.eps) ** self.alpha
        self.prioritys[indices] = prios

        
        
        
        


