"""
实现DQN中的Q网络定义
"""
import torch
import torch.nn as nn
import numpy as np
import math

class NoisyLinear(nn.Module):
    def __init__(self, feats_in, feats_out, std_init=0.5):
        super().__init__()
        self.feats_in = feats_in
        self.feats_out = feats_out
        self.std_init = std_init
        self.w_mu = nn.Parameter(torch.FloatTensor(feats_out, feats_in))
        self.w_sigma = nn.Parameter(torch.FloatTensor(feats_out, feats_in))
        self.register_buffer('w_eps', torch.FloatTensor(feats_out, feats_in))
        self.b_mu = nn.Parameter(torch.FloatTensor(feats_out))
        self.b_sigma = nn.Parameter(torch.FloatTensor(feats_out))
        self.register_buffer('b_eps', torch.FloatTensor(feats_out))
        self.para_init()

    def para_init(self):
        sqrt_fin = 1 / math.sqrt(self.feats_in)
        self.w_mu.data.uniform_(-sqrt_fin, sqrt_fin) # 按照Xavier，w取值范围与输入特征数呈反比
        self.w_sigma.data.fill_(self.std_init * sqrt_fin)
        self.w_eps.normal_() # N(0,1)
        self.b_mu.data.uniform_(-sqrt_fin, sqrt_fin)
        self.b_sigma.data.fill_(self.std_init * sqrt_fin)
        self.b_eps.normal_()
    def reset_noise(self):
        self.w_eps.normal_()
        self.b_eps.normal_()
    
    def forward(self, x):
        if self.training:
            w = self.w_mu + self.w_sigma * self.w_eps
            b = self.b_mu + self.b_sigma * self.b_eps
        else:
            w = self.w_mu
            b = self.b_mu
        return nn.functional.linear(x, w, b)

class DqnFc(nn.Module):
    def __init__(self, feats_in, feats_out, noisy=False, categorical=False, atoms=51):
        super().__init__()
        if categorical:
            feats_out = feats_out * atoms # C51中动作的类别数是动作数*原子数51
        if noisy:
            self.fc_module = nn.Sequential(
                nn.Flatten(),
                NoisyLinear(feats_in, 512),
                nn.ReLU(),
                NoisyLinear(512, feats_out)
            )
        else:
            self.fc_module = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feats_in, 512),
                nn.ReLU(),
                nn.Linear(512, feats_out)
            )
    def reset_noise(self):
        for m in self.fc_module:
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        return self.fc_module(x)


class QNet(nn.Module):
    def __init__(self, input_shape, action_num, args): # input_shape: (B, 4(frame stack), H=84, W=84)
        super().__init__()
        self.act_num = action_num
        self.dueling = args.dueling
        self.noisy = args.noisy
        self.categorical = args.categorical
        self.atoms = args.atoms
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # (84-8)/4 + 1 = 20. -> (B, 32, 20, 20) 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (20-4)/2 + 1 = 9. -> (B, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (9-3)/1 + 1 = 7. -> (B, 64, 7, 7)
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)  # 1*64*7*7 = 3136
        if self.dueling:
            self.fc_state_value = DqnFc(conv_out_size, 1, self.noisy, self.categorical, self.atoms)
            self.fc_act_advantage = DqnFc(conv_out_size, action_num, self.noisy, self.categorical, self.atoms)
        else:
            self.fc = DqnFc(conv_out_size, action_num, self.noisy, self.categorical, self.atoms)

    def reset_noise(self):
        if self.dueling:
            self.fc_state_value.reset_noise()
            self.fc_act_advantage.reset_noise()
        else:
            self.fc.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        if self.dueling:
            state_value = self.fc_state_value(conv_out) # shape: [B, 1] or [B, 1*51]
            state_value = state_value.repeat(repeats=(1, self.act_num)) # shape: [B, act_num or *51]
            act_advantage = self.fc_act_advantage(conv_out)     # shape: [B, action_num]
            q = state_value + act_advantage - act_advantage.mean(dim=1, keepdim=True) 
        else:
            q = self.fc(conv_out)

        if self.categorical:
            q = q.view(-1, self.act_num, self.atoms)  # shape: [B, action_num, atoms]
            q = q.softmax(dim=-1)  # shape: [B, action_num, atoms]
        return q

