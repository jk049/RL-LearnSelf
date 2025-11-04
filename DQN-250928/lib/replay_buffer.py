"""
用numpy实现简易replay buffer
"""

import torch
import collections
import math
import numpy as np

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
        actions_t = torch.tensor(action, dtype=torch.int64, device=self.actions.device)
        rewards_t = torch.tensor(reward, dtype=torch.float32, device=self.rewards.device)
        dones_t = torch.tensor(done, dtype=torch.bool, device=self.dones.device)
        exp_num = actions_t.shape[0]
        idx = (self.cur_index + torch.arange(exp_num, device=self.actions.device)) % self.capacity
        self.states[idx] = state
        self.actions[idx] = actions_t
        self.rewards[idx] = rewards_t
        self.next_states[idx] = next_state
        self.dones[idx] = dones_t
        self.cur_index = (self.cur_index + exp_num) % self.capacity
    
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
存储空间优化的RB
核心设计：
    1. state和nextstate共用states空间;
    2. state按uint8存储, 采样是时转float32并归一化;
    3. 支持multi-step return;
    4. 支持多环境并行采样存储;
    5. 支持framestack采样;
    6. all tensor都存于device上，避免频繁cpu<->gpu传输开销.
"""
class OptimRB:
    def __init__(self, env_nums, capacity, image_size, device, batch_size=32, multi_step=3, framestack=4, gamma=0.99):
        if multi_step >= framestack:
            raise ValueError("multi_step must be less than framestack.")

        self.cur_index = 0
        self.batch_size = batch_size
        self.capacity = capacity
        self.multi_step = multi_step if multi_step < framestack else framestack
        self.framestack = framestack
        self.env_nums = env_nums
        self.gamma = gamma
        self.gammas = torch.tensor([gamma**i for i in range(multi_step)], dtype=torch.float32, device=device)
        self.device = device
        self.full = False
        self.tail = framestack -1  # 每个环境末尾预留的冗余空间，便于采样framestack
        self.cap_per_env = capacity // env_nums + self.tail
        self.states = torch.zeros((env_nums, self.cap_per_env, *image_size), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((env_nums, self.cap_per_env), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((env_nums, self.cap_per_env), dtype=torch.float32, device=device)
        self.dones = torch.zeros((env_nums, self.cap_per_env), dtype=torch.bool, device=device)
        self.states_stack = self.states.unfold(dimension=1, size=framestack, step=1) # [E, C, W, H, F]
        self.actions_stack = self.actions.unfold(dimension=1, size=framestack, step=1)  # [e_num,cap//e_num,f_stk]
        self.rewards_stack = self.rewards.unfold(dimension=1, size=framestack, step=1)  
        self.dones_stack = self.dones.unfold(dimension=1, size=framestack, step=1) 
        self.done_tmp = torch.ones((env_nums), dtype=torch.bool, device=device) # 用于暂存上一个exp的done标记
        self.pre_index = 0
        self.prefetch_stream = torch.cuda.Stream(device=device)
        self.preloaded_batch = None
    # 注意：以下索引辅助张量应按当前采样 batch 的实际形状动态构造，避免 batch 大小变化导致广播错误
    # 在 _state_mask 与 sample 中按需即时创建（不缓存固定 batch 形状的张量）。
    
    # s, a, r, d都在外面转成tensor再传进来，对于cule，训练和存储exp都在相同的device上以提高效率
    """
    Input:
        state: [env_nums, w, h], uint8
        action: [env_nums], int64
        reward: [env_nums], float32
        nest_state: [env_nums, w, h], uint8. 由于state共用空间，不用此参数。此处保留仅为接口统一
        done: [env_nums], bool
    """
    def add(self, state, action, reward, nest_state, done): # 
        self.dones[:, self.pre_index] = self.done_tmp # 更新上一个exp的done标记

        self.states[:, self.cur_index] = state
        self.actions[:, self.cur_index] = action
        self.rewards[:, self.cur_index] = reward
        self.dones[:, self.cur_index] = True # 先假设当前exp是episode结尾, 采样到此处时，不会把不同episode的frame包含进来
        self.done_tmp = done

        # 如果cur_index在tail范围内，还要把cap尾部对应位置也写入
        if self.cur_index < self.tail and self.full:
            self.states[:, self.cap_per_env - self.tail + self.cur_index] = state
            self.actions[:, self.cap_per_env - self.tail + self.cur_index] = action
            self.rewards[:, self.cap_per_env - self.tail + self.cur_index] = reward
            self.dones[:, self.cap_per_env - self.tail + self.cur_index] = self.dones[:, self.cur_index]
            
        if self.pre_index < self.tail and self.full:
            self.dones[:, self.cap_per_env - self.tail + self.pre_index] = self.dones[:, self.pre_index]

        self.pre_index = self.cur_index

        self.cur_index = (self.cur_index + 1) % (self.cap_per_env - self.tail)
        if self.cur_index == 0:
            self.full = True
        if self.cur_index > self.batch_size:
            self.preload()
        
    """
    [Input]:
        states: [B, w, h, framestack], uint8
        dones: [B, framestack], bool
        mast_type:
            'before': mask states before the last done, used for current states
            'after': mask states after the first done, used for next states
    [Output]:
        masked_states: [B, framestack, w, h], float32, 0~1
    """
    def _state_mask(self, states, dones, mask_type='before'):
        # 输入 states 形状为 [B, W, H, F]（unfold 后索引的常见布局），转换为 [B, F, W, H]
        states = states.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        if not dones.any().item():
            return states

        B, F, W, H = states.shape
        device = states.device
        # 按当前 B、F 动态构造索引网格，避免固定 batch 形状引发的广播或越界问题
        orin_idx = torch.arange(F, device=device, dtype=torch.long).view(1, F).expand(B, F)
        idx_low = torch.full((B, F), -1, dtype=torch.long, device=device)
        idx_high = torch.full((B, F), F, dtype=torch.long, device=device)
        if mask_type == 'before':
            done_pos = torch.where(dones.bool(), orin_idx, idx_low) 
            last_done_pos = done_pos.max(dim=1).values # [B]，∈[-1..F-1]，-1 表示“无 done”
            has_done = (last_done_pos >= 0).view(B, 1).expand(B, F) # [B]，是否存在 done
            replace_pos = (last_done_pos + 1).clamp_max(F - 1).view(B, 1).expand(B, F) # [B]
            last_done_pos = last_done_pos.view(B, 1).expand(B, F) # [B, F]
            replace_flag = has_done & (orin_idx <= last_done_pos) # [B, F]，哪些位置要替换
        else:  # 'after'
            done_pos = torch.where(dones.bool(), orin_idx, idx_high) 
            first_done_pos = done_pos.min(dim=1).values
            has_done = (first_done_pos < F).view(B, 1).expand(B, F)
            first_done_pos = first_done_pos.view(B, 1).expand(B, F)
            replace_pos = first_done_pos
            replace_flag = has_done & (orin_idx > first_done_pos)

        masked_idx = torch.where(replace_flag, replace_pos, orin_idx).view(B, F, 1, 1).expand(B, F, W, H)
        # 保护性钳制，防止异常情况下的越界（理论上不会触发）；使用非原地版本以兼容 expand 视图
        masked_idx = masked_idx.clamp(0, F - 1)
        return torch.gather(states, dim=1, index=masked_idx) 

    """
    Input: batch_size: int
    Output: (S, A, R, S', D)
        states: [B, framestack, w, h], float32, 0~1
        actions: [B], int64
        rewards: [B], float32
        next_states: [B, framestack, w, h], float32, 0~1
        dones: [B], bool
    """
    def _sample(self, batch_size):
        cur_capacity = self.capacity if self.full else self.cur_index * self.env_nums
        if cur_capacity <= self.multi_step:
            raise RuntimeError("Not enough samples in buffer to sample with the given framestack and multi-step.")

        sample_indices = torch.randint(0, cur_capacity - self.multi_step, (batch_size,), device=self.device, dtype=torch.long)
        env_indices = sample_indices % self.env_nums
        pos_indices = sample_indices // self.env_nums
        next_pos_indices = (pos_indices + self.multi_step) % (self.cap_per_env - self.tail) 

        states = self.states_stack[env_indices, pos_indices]  # [batch, w, h, framestack]
        next_states = self.states_stack[env_indices, next_pos_indices]
        dones = self.dones_stack[env_indices, pos_indices] # [B, framestack]
        next_dones = self.dones_stack[env_indices, next_pos_indices]
        states = self._state_mask(states, dones, 'before')
        next_states = self._state_mask(next_states, next_dones, 'after')

        actions_indices = (pos_indices + self.framestack -1) % (self.cap_per_env - self.tail)
        actions = self.actions[env_indices, actions_indices]  

        reward_indices = (pos_indices + self.framestack -1) % (self.cap_per_env - self.tail)
        rewards = self.rewards_stack[env_indices, reward_indices]  # [B, multi_step]
        reward_dones = self.dones_stack[env_indices, reward_indices]
        done_false = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        reward_dones = torch.cat([done_false, reward_dones[:, :-1]], dim=1)
        reward_mask = (~reward_dones).int().cumprod(dim=1).float() # cumprod: 累乘
        rewards = ((rewards[:, :self.multi_step] * reward_mask[:, :self.multi_step]) * self.gammas).sum(dim=1) 

        dones = ~(reward_mask[:, self.multi_step -1].bool())
        return (states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone())

    def preload(self):
        with torch.cuda.stream(self.prefetch_stream):
            self.preloaded_batch = self._sample(self.batch_size)
    
    def sample(self, batch_size):
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        batch = self.preloaded_batch
        self.preload()
        return batch


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
        exp_num = state.shape[0]
        idx = (self.cur_index + torch.arange(exp_num, device=self.actions.device)) % self.capacity
        self.prioritys[idx] = max_prio
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

        
class CuleRB():
    def __init__(self, args, capacity, device, num_ales=None):
        self.num_ales = num_ales if num_ales else args.env_num
        self.device = device
        self.history = args.framestack
        self.discount = args.gamma
        self.n = args.multi_step
        self.priority_weight = 0 # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = 0 
        self.priority_replay = 0

        self.full = False
        self.index = 0
        self.epoch = 0
        self.steps_per_ale = int(math.ceil(float(capacity) / self.num_ales))
        self.capacity = self.steps_per_ale * self.num_ales

        self.actions = torch.zeros(self.num_ales, self.steps_per_ale, device=self.device, dtype=torch.long)

        if self.priority_replay:
            self.priority = torch.ones(self.capacity, device=self.device, dtype=torch.float32) * float(np.finfo(np.float32).eps)
            self.priority_view = self.priority.view(self.num_ales, self.steps_per_ale)
            self.rank = torch.zeros(self.capacity, device=self.device, dtype=torch.long)

        self.section_offset = torch.zeros(0, device=self.device, dtype=torch.int32)
        self.section_size = torch.zeros(0, device=self.device, dtype=torch.int32)
        self.gammas = self.discount ** torch.FloatTensor(range(self.n)).to(self.device).unsqueeze(0)
        self.frame_offsets = torch.IntTensor(range(-(self.history - 1), 1)).to(device=self.device).unsqueeze(0)
        self.weights = torch.ones(args.batch_size, device=self.device, dtype=torch.float32)

        width, height = 84, 84
        imagesize = width * height
        num_steps = self.steps_per_ale + 2 * (self.history - 1)
        stepsize  = num_steps * imagesize

        self.observations = torch.zeros((self.num_ales, num_steps, width, height), device=self.device, dtype=torch.uint8)
        self.states_view = self.observations.as_strided(
            size=torch.Size([self.num_ales, num_steps - (self.history - 1), self.history, width, height]),
            stride=(stepsize, imagesize, imagesize, width, 1),
            storage_offset=0
        )

        self.frame_number = torch.zeros(self.num_ales, num_steps, device=self.device, dtype=torch.int32)
        self.frame_number[:, (self.history - 1) + (self.steps_per_ale - 1)] = -1
        self.frame_view = self.frame_number.as_strided(
            size=torch.Size([self.num_ales, num_steps - (self.history - 1), self.history]),
            stride=(num_steps, 1, 1),
            storage_offset=0,
        )

        self.rewards = torch.zeros(self.num_ales, num_steps, device=self.device, dtype=torch.float32)
        self.reward_view = self.rewards.as_strided(
            size=torch.Size([self.num_ales, num_steps - (self.history - 1), self.n]),
            stride=(num_steps, 1, 1),
            storage_offset=0,
        )

    def update_sections(self, batch_size):
        capacity = self.capacity if self.full else self.index * self.num_ales

        if self.section_size.size(0) != capacity:
            # initialize rank-based priority segment boundaries
            pdf = torch.FloatTensor(1.0 / np.arange(1, capacity + 1)).to(device=self.device) ** self.priority_exponent
            self.p_i_sum = pdf.sum(0)
            pdf = pdf / self.p_i_sum
            cdf = pdf.cumsum(0)

            haystack = cdf.cpu().numpy()
            needles = np.linspace(0, 1, batch_size, endpoint=False)[::-1]
            self.section_offset = np.trim_zeros(np.searchsorted(haystack, needles))
            self.section_offset = torch.from_numpy(self.section_offset).to(device=self.device)
            self.section_offset = torch.cat((self.section_offset, torch.LongTensor([0]).to(device=self.device)))
            self.section_size = self.section_offset[:-1] - self.section_offset[1:]

            mask = self.section_size != 0
            self.section_offset = self.section_offset[:-1][mask]
            self.section_offset = torch.cat(((self.section_offset, torch.LongTensor([0]).to(device=self.device))))
            self.section_size = self.section_size[mask]

        return self.section_size.size(0)

    def reset(self, observations):
        self.observations[:, self.history - 1] = observations.mul(255.0).to(device=self.device, dtype=torch.uint8)

    # Adds state and action at time t, reward and terminal at time t + 1
    def add(self, observations, actions, rewards, next_state, terminals):
        if actions is None:
            actions = torch.zeros(self.num_ales, device=self.device, dtype=torch.long)
        if rewards is None:
            rewards = torch.zeros(self.num_ales, device=self.device, dtype=torch.float32)

        curr_offset = self.index + self.history
        prev_offset = curr_offset - 1 + ((self.index == 0) * self.steps_per_ale)

        nonterminal = (terminals == 0).float()
        terminal_mask = terminals == 1

        self.actions[:, self.index] = actions
        self.rewards[:, self.index] = rewards.float() * nonterminal
        self.rewards[terminal_mask, prev_offset - (self.history - 1)] = rewards[terminal_mask].float()
        self.observations[:, curr_offset] = observations.mul(255.0).to(device=self.device, dtype=torch.uint8)
        self.frame_number[:, curr_offset] = nonterminal.int() * (self.frame_number[:, prev_offset] + 1)
        # self.frame_number[:, curr_offset] += terminals.int() * np.random.randint(sys.maxsize / 1024)

        if self.priority_replay:
            self.priority_view[:, self.index] = self.priority.max() + float(np.finfo(np.float32).eps)
            self.rank = self.priority.sort(descending=True)[1]

        if (self.epoch > 0) and (self.index == 0):
            self.observations[:, self.history - 1] = self.observations[:, self.steps_per_ale + self.history - 1]
            self.frame_number[:, self.history - 1] = self.frame_number[:, self.steps_per_ale + self.history - 1]

        self.index = (self.index + 1) % self.steps_per_ale  # Update index
        self.full |= self.index == 0
        self.epoch += int(self.index == 0)

    def sample(self, batch_size=0, indices=None):
        capacity = self.capacity if self.full else self.index * self.num_ales

        if indices is None:
            indices = torch.randint(capacity, (batch_size,), device=self.device, dtype=torch.long)

        batch_size = indices.size(0)

        if self.priority_replay:
            batch_size = self.update_sections(batch_size)

            indices = self.section_offset[:-1] + (indices[:batch_size] % self.section_size)
            p_i = (indices.float() + 1.0) ** -self.priority_exponent
            P = p_i / self.p_i_sum
            indices = self.rank[indices]

            weights = (capacity * P) ** -self.priority_weight  # Compute importance-sampling weights w
            weights /= weights.max()   # Normalize by max importance-sampling weight from batch
        else:
            weights = self.weights

        ale_indices = indices % self.num_ales
        step_indices = indices // self.num_ales

        # Create un-discretised state and nth next state
        base_frame_numbers = self.frame_number[ale_indices, step_indices + self.history - 1].unsqueeze(-1).expand(-1, self.history)
        expected_frame_numbers = base_frame_numbers + self.frame_offsets.expand(batch_size, -1)

        actual_frame_numbers = self.frame_view[ale_indices, step_indices]
        states_mask = (actual_frame_numbers == expected_frame_numbers).float().unsqueeze(-1).unsqueeze(-1)
        states = self.states_view[ale_indices, step_indices].float().div_(255.0) * states_mask

        next_actual_frame_numbers = self.frame_view[ale_indices, step_indices + self.n]
        next_states_mask = (next_actual_frame_numbers == (expected_frame_numbers + self.n)).float().unsqueeze(-1).unsqueeze(-1)
        next_states = self.states_view[ale_indices, step_indices + self.n].float().div_(255.0) * next_states_mask

        # Discrete action to be used as index
        actions = self.actions[ale_indices, step_indices]

        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = self.reward_view[ale_indices, step_indices]
        returns = torch.sum(self.gammas * rewards * next_states_mask[:, -self.n - 1:-1, 0, 0], 1)

        # Check validity of the last state
        nonterminals = next_actual_frame_numbers[:, -1] == (expected_frame_numbers[:, -1] + self.n)

        #return indices, states, actions, returns, next_states, nonterminals, weights
        return states, actions, returns, next_states, ~nonterminals

    def update_priorities(self, indices, td_error):
        if self.priority_replay:
            self.priority[indices] = td_error.abs() ** self.priority_exponent
            self.rank = self.priority.sort(descending=True)[1]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration

        ale_index = int(self.current_idx % self.num_ales)
        step_index = int(self.current_idx / self.num_ales)

        # Create un-discretised state
        base_frame_numbers = self.frame_number[ale_index, step_index + self.history - 1].expand(self.history)
        expected_frame_numbers = base_frame_numbers + self.frame_offsets.squeeze(0)

        actual_frame_numbers = self.frame_view[ale_index, step_index]
        states_mask = (actual_frame_numbers == expected_frame_numbers).float().unsqueeze(-1).unsqueeze(-1)
        states = self.states_view[ale_index, step_index].float().div_(255.0) * states_mask

        self.current_idx += 1
        return states
        
        


