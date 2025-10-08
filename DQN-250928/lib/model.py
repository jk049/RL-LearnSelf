"""
实现DQN中的Q网络定义
"""
import torch
import torch.nn as nn
import numpy as np

class Q_Net(nn.Module):
    def __init__(self, input_shape, action_num): # input_shape: (B, 4(frame stack), H=84, W=84)
        super(Q_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # (84-8)/4 + 1 = 20. -> (B, 32, 20, 20) 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (20-4)/2 + 1 = 9. -> (B, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (9-3)/1 + 1 = 7. -> (B, 64, 7, 7)
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)  # 1*64*7*7 = 3136
        self.fc = nn.Sequential(
            nn.Flatten(), # 展平，[B, 64, 7, 7] -> [B, 3136]
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_num)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out) 

class Dueling_Net(nn.Module):
    def __init__(self, input_shape, action_num): # input_shape: (B, 4(frame stack), H=84, W=84)
        super(Dueling_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # (84-8)/4 + 1 = 20. -> (B, 32, 20, 20) 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (20-4)/2 + 1 = 9. -> (B, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (9-3)/1 + 1 = 7. -> (B, 64, 7, 7)
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)  # 1*64*7*7 = 3136

        self.fc_value = nn.Sequential(
            nn.Flatten(), # 展平，[B, 64, 7, 7] -> [B, 3136]
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Flatten(), # 展平，[B, 64, 7, 7] -> [B, 3136]
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_num)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        value = self.fc_value(conv_out) # shape: [B, 1]
        adv = self.fc_adv(conv_out)     # shape: [B, action_num]
        q = value + adv - adv.mean(dim=1, keepdim=True) # 广播机制，q shape: [B, action_num]
        return q



