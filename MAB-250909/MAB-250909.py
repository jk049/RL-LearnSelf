"""
MAB(Multi-Armed Bandit) 多臂老虎机问题
[问题描述]：一个多拉杆的老虎机，拉每个拉杆获得奖励的概率不同。运用强化学习，求得奖励最高的拉杆策略。
[算法设计]：
    1. 环境设计：设计一个N臂老虎机，每个臂获奖的概率独立且服从伯努利分布，
               即获奖的概率为p_i,不获奖的概率为1-p_i。获奖的奖励为1，不获奖的奖励为0。
               N个杆的获奖概率p随机生成0到1的浮点数。动作空间是0到N-1的整数，代表拉第几个杆。
    2. 算法设计：
        1.探索策略：
            a. epsilon-greedy策略：以1-epsilon的概率选择当前已知的最优动作，以epsilon的概率随机选择动作；
            b. time decay epsilon-greedy策略：随着时间步的增加，epsilon逐渐减小，减少随机探索的频率；
            c. UCB(Upper Confidence Bound): 置信上界策略，按照公式
               a = argmax(Q(s,a) + c * sqrt(ln(time_step)) / N(s, a))) 选择动作。
               其中，Q(s, a)是动作a的估计奖励，sqrt(ln(t)/N(s,a))是动作a的不确定度，选二者之和最大的动作，兼顾探索和利用。
            d. thompson采样:拉杆奖励为伯努利分布时，第i个拉杆共被拉k_i次，其中a_i次获得奖励，b_i次无奖励，
                则第i个拉杆的奖励分布服从参数为(a_i+1, b_i+1)的Beta分布。
        2. 计算后悔值：
            a. 为什么要计算后悔值：
            b. 什么情况要计算后悔值： regret不参与agent优化，只是一个评估指标
            c. 如何计算后悔值: 最大理想奖励期望值 - 当前动作的理想奖励期望值
        3. 问题目标：最大化累计奖励，最小化累积奖励和理论最大奖励的差值。
[伪代码]：
    初始化MAB环境
    for t = 1 -> T:
        根据探索策略，选择动作a
        执行动作a，获得奖励rwd_a
        更新动作a的期望奖励Q(s, a) += (rwd_a - Q(s,a)) / N(s,a)
        更新后悔值, regret += max(ENV_Q(s, :)) - ENV_Q(s, a)
    end for
[实验结果]：
    1. 固定epsilon- greedy策略探索充分，但是后期利用不足，累积后悔值较高；
    2. time decay epsilon-greedy策略，探索衰减太激进，容易探索不足，需要调整衰减速度；
    3. UCB策略，探索和利用的平衡较好，前期探索多，后期利用多，累积后悔值最低；
    4. thompson采样，前期探索较多，后期探索不如UCB充分，估计期望与真实期望有一定差距，总效果介于epsilon和UCB之间。
"""

import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 1
ARM_NUM = 10
TIME_STEP = 5000
PLT_FLAG = True
FIXED_EPS_MODE_ID = 0
TIME_DECAY_EPS_MODE_ID = 1
UCB_MODE_ID = 2
THOMPSON_MODE_ID = 3
EXPLORATION_MODE_LIST = ["fixed-epsilon", "time decay epsilon", "ucb", "thompson"]
FIXED_EPS_VALUE = 0.02
DECAY_EPS_BASE = 10
UCB_COEF = 1
#EXPLORATION_MODE = UCB_MODE_ID
#EXPLORATION_MODE = TIME_DECAY_EPS_MODE_ID
#EXPLORATION_MODE = FIXED_EPS_MODE_ID
EXPLORATION_MODE = THOMPSON_MODE_ID

class MABEnv:
    def __init__(self, arm_num=10, plt_flag=True):
        self.arm_num = arm_num
        self.rwd_prbs_ary = np.random.rand(arm_num)

        if plt_flag is True:
            plt.figure()
            plt.title("Reward Probabilities of each arm")
            plt.xlabel("Arm index")
            plt.ylabel("Reward Probability")
            plt.bar(range(arm_num), self.rwd_prbs_ary)
            plt.show()
            print("Reward Probabilities of each arm: ", self.rwd_prbs_ary)

    def step(self, action):
        assert 0 <= action < self.arm_num, "action must be in [0, {})".format(self.arm_num)
        rwd_prob = self.rwd_prbs_ary[action]
        rwd = 1 if random.random() < rwd_prob else 0
        return rwd

class RlAgent:
    def __init__(self, arm_num=10, time_step=5000, plt_flag=True, exploration_mode=EXPLORATION_MODE):
        self.arm_num = arm_num
        self.time_step = time_step
        self.plt_flag = plt_flag
        self.regret = 0
        #self.q_est_ary = np.zeros(arm_num) # 初始化为0会导致一开始所有动作的估计奖励都是0，无法区分哪个动作更优，容易导致初始阶段的探索不足
        self.q_est_ary = np.ones(arm_num)
        self.act_cnt_ary = np.zeros(arm_num)
        self.env = MABEnv(arm_num, plt_flag)
        self.regret_ary = np.zeros(time_step)
        self.regret_per_step = np.zeros(time_step)
        self.est_regret_per_step = np.zeros(time_step)
        self.stable_cumulative_regret = np.zeros(time_step)
        self.epsilon_ary = np.zeros(time_step)
        self.act_ary = np.zeros(time_step, dtype=int)
        self.exploration_mode = exploration_mode
        self.ucb_ary = np.zeros(arm_num) # UCB算法中每个动作的置信上界值
        self.act_get_rwd_times = np.ones(arm_num) # thompson采样中每个动作获得奖励的次数
        self.act_no_rwd_times = np.ones(arm_num) # thompson采样中每个动作未获得奖励的次数
        
        if exploration_mode == FIXED_EPS_MODE_ID:
            avg_rwd = np.sum(self.env.rwd_prbs_ary) / arm_num
            for i in range(time_step):
                self.stable_cumulative_regret[i] = FIXED_EPS_VALUE * (max(self.env.rwd_prbs_ary) - avg_rwd) * i

    def select_action(self, cur_step):
        if self.exploration_mode <= TIME_DECAY_EPS_MODE_ID:
            epsilon = FIXED_EPS_VALUE if self.exploration_mode == FIXED_EPS_MODE_ID else DECAY_EPS_BASE / (cur_step + 1)
            self.epsilon_ary[cur_step] = epsilon
            action = random.randint(0, self.arm_num - 1) if random.random() < epsilon else np.argmax(self.q_est_ary)
        elif self.exploration_mode == UCB_MODE_ID: # max(Q(s,a) + coef * sqrt(ln(t) / N(s, a)))
            self.ucb_ary = self.q_est_ary + UCB_COEF * np.sqrt(np.log(cur_step + 1) / (self.act_cnt_ary + 1e-5))
            action = np.argmax(self.ucb_ary)
        elif self.exploration_mode == THOMPSON_MODE_ID:
            # np.random.beta(a, b) 产生一个Beta分布的随机数，a是成功次数+1，b是失败次数+1
            # samples是每个动作的一个采样值，选择采样值最大的动作。采样值表示了该动作的潜在奖励能力
            # 从beta分布采样，表示了对每个动作奖励概率的估计不确定性，采样值高的动作更可能是高奖励概率的动作

            samples = np.random.beta(self.act_get_rwd_times, self.act_no_rwd_times)
            action = np.argmax(samples)
        else:
            raise NotImplementedError("Only fixed-epsilon and decay-epsilon modes are implemented.")

        self.act_ary[cur_step] = action
        return action

    def learn(self):
        for i in range(self.time_step):
            action = self.select_action(i)
            self.act_cnt_ary[action] += 1
            rwd = self.env.step(action)
            self.act_get_rwd_times[action] += rwd
            self.act_no_rwd_times[action] += (1 - rwd)
            self.q_est_ary[action] += (rwd - self.q_est_ary[action]) / self.act_cnt_ary[action]
            #self.regret_per_step[i] = max(self.q_est_ary) - rwd # 我认为这里应该是估计奖励的最大值 - 实际奖励
            # 教程里的计算方法是对的，后悔值就是未知的期望之差，不参与agent优化，只是一个评估指标
            self.regret_per_step[i] = max(self.env.rwd_prbs_ary) - self.env.rwd_prbs_ary[action] # 动手学RL教程中的后悔值计算方式 
            self.est_regret_per_step[i] = max(self.q_est_ary) - self.q_est_ary[action] # 估计的后悔值
            self.regret += self.regret_per_step[i] 
            self.regret_ary[i] = self.regret

        if self.plt_flag is True:
            self.plot("Epsilon Decay over Time Steps", "Time Step", "Epsilon", self.epsilon_ary, "Epsilon")
            self.plot("Estimated Q values of each arm", "Arm index", "Estimated Q value", 
                      self.q_est_ary, "Estimated Q value", self.env.rwd_prbs_ary, "True Reward Probability")
            self.plot("Action per Step over Time Steps", "Time Step", "Action Index", self.act_ary, "Action Index")
            self.plot("Cumulative Regret over Time Steps", "Time Step", "Cumulative Regret", 
                      self.regret_ary, "Cumulative Regret", self.stable_cumulative_regret, "Stable Cumulative Regret")
            self.plot("Regret per Step over Time Steps", "Time Step", "Regret per Step", 
                      self.regret_per_step, "Regret per Step", self.est_regret_per_step, "Estimated Regret per Step")


    def plot(self, title, xlabel, ylabel, ydata1, label1, ydata2=None, label2=None):
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(ydata1, label=label1)
        plt.plot(ydata2, label=label2) if ydata2 is not None else None
        plt.legend()
        plt.show()
            
np.random.seed(SEED)
random.seed(SEED)

agent = RlAgent(ARM_NUM, TIME_STEP, PLT_FLAG)
agent.learn()




