"""
【问题描述】:基于动态规划的强化学习算法，本例用于解决冰湖问题（FrozenLake-v1）。包含两种解法：
    1. 策略迭代：迭代更新状态价值函数到收敛，再根据价值函数更新策略；重复此过程知道策略不再变化。
    2. 值迭代：每更新一次状态价值函数，就更新一次策略；重复此过程直到价值函数收敛。
【环境设计】：
    1. 创建环境：gym.make('FrozenLake-v1', is_slippery=True)
    2. 获取环境私有属性，以便访问状态价值函数和转移矩阵：env.unwrapped
    3. 环境渲染：env.render()
【算法设计】：
    1. 策略迭代：
        a. 初始化策略为均匀随机策略，初始化状态价值函数为0；
        b. 迭代状态价值函数至收敛；
        c. 根据状态价值函数更新策略；
        d. 重复b和c直到策略不再变化。
    2. 值迭代：
        a. 初始化状态价值函数为0，初始化策略为均匀随机策略；
        b. 迭代更新状态价值函数至收敛（以max action为状态价值）
        c. 根据新的价值函数更新策略；
        d. 重复b,c直到状态价值函数收敛。  
【伪代码】：
    创建环境 env=gym.make(...)
    环境解封 env=env.unwrapped
    初始化状态价值函数 V(s)=0, V_old(s)=V(s)
    初始化策略 π(a|s)=1/|A|
    初始化状态价值收敛阈值theta
    初始化状态价值函数最大差值 delta=inf
    if 策略迭代:
        for iter=1->MAX_ITER:
            # 策略评估
            while delta > theta:
                for s in S:
                    V(s) = sum(pi(a|s) * sum(p(s'|s,a) * (reward + gamma * V_old(s')))) 
                delta = max(delta, |V(s)-V_old(s)|)
                V_old(s) = V(s) 
            # 策略改进
            for s in S:
                best_a = argmax_a sum(p(s'|s,a) * (reward + gamma * V(s')))
                pi(a|s) = 1/best_a_num if a in best_a else 0
            if 策略不再变化:
                break
        end for
    else if 值迭代:
        for iter=1->MAX_ITER:
            while delta > theta:
                for s in S:
                    V(s) = argmax_a (sum(p(s'|s,a) * (reward + gamma * V_old(s'))))
                delta = max(delta, |V(s)-V_old(s)|)
                V_old(s) = V(s)
                
                for s in S:
                    best_a = argmax_a sum(p(s'|s,a) * (reward + gamma * V(s')))
                    pi(a|s) = 1/best_a_num if a in best_a else 0
            if 策略不再变化：
                break
        end for

【实验结果】：
    1. 值迭代和策略迭代均能收敛到最优策略，且最优策略相同；
    2. 策略迭代每次迭代需要多次更新状态价值函数，收敛速度较慢；
    3. 值迭代每次迭代只需更新一次状态价值函数，收敛速度较快。
"""

import gym
import numpy as np

TEST_EPISODE_NUM = 0
GAMMA = 0.9
THETA = 1e-6
MAX_ITER = 1000
SLIPPERY = False
POLICY_ITER = False #False表示值迭代
VALUE_ITER = not POLICY_ITER

env = gym.make('FrozenLake-v1', is_slippery=SLIPPERY)
env = env.unwrapped  # 取消环境的封装，便于访问内部属性
env.reset()
env.render()

if TEST_EPISODE_NUM > 0:
    for episode in range(TEST_EPISODE_NUM):
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()  # 随机采样一个动作
            next_state, reward, done, info = env.step(action)
            env.render()
            state = next_state
            step += 1
        print(f"Episode {episode+1} finished in {step} steps.")

class DPRL:
    def __init__(self, env, gamma=0.9, theta=1e-5, max_iter=1000):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        self.v_state = np.zeros(self.state_num)
        self.v_state_old = self.v_state.copy() # 用于存储上一次迭代的状态价值函数, 浅拷贝，数据相同，内存地址不同
        # 初始化均匀随机策略
        self.policy = [[1/self.action_num for _ in range(self.action_num)] for _ in range(self.state_num)]
        self.policy_old = [[self.policy[s][a] for a in range(self.action_num)] for s in range(self.state_num)]

    def policy_iter(self):
        for iter in range(self.max_iter):
            v_diff_max = float('inf')
            vs_iter_cnt = 0
            # qsa迭代收敛
            while v_diff_max > self.theta:
                v_diff_max = 0
                # v(s) = sum_a {pi(a|s) * sum_s' [p(s'|s, a) * (r + gamma * v_old(s'))]}
                # v(s) = sum_a {pi(a|s) * [reward + gamma * sum_s' p(s'|s, a) * v_old(s')]}
                for s in range(self.state_num):
                    v_s_tmp = 0
                    for action in range(self.action_num):
                        act_prob = self.policy[s][action]
                        for prob, next_s, reward, done in self.env.P[s][action]:
                            v_s_tmp += act_prob * prob * (reward + self.gamma * self.v_state_old[next_s]) # 此处公式待改
                            #v_s_tmp += act_prob * (reward + self.gamma * prob * (self.v_state_old[next_s])) # 此处公式待改
                    self.v_state[s] = v_s_tmp
                v_diff_max = max(v_diff_max, abs(self.v_state - self.v_state_old).max()) 
                self.v_state_old = self.v_state.copy()
                vs_iter_cnt += 1
                print(f"Iteration {iter+1}, vs iter {vs_iter_cnt} times, Value Function max change: {v_diff_max}")
                print("state array:\n", self.v_state.reshape((4, 4)))
            print(f"Policy Iteration {iter+1}, Value Function converged in {vs_iter_cnt} iterations.")

            # 策略改进
            self.policy_old = [[self.policy[s][a] for a in range(self.action_num)] for s in range(self.state_num)]
            for s in range(self.state_num):
                q_sa = np.zeros(self.action_num)
                for action in range(self.action_num):
                    for prob, next_s, reward, done in self.env.P[s][action]:
                        q_sa[action] += prob * (reward + self.gamma * self.v_state[next_s])
                best_a = np.argwhere(q_sa == np.max(q_sa)).flatten().tolist()
                best_a_num = len(best_a)
                for action in range(self.action_num):
                    self.policy[s][action] = 1 / best_a_num if action in best_a else 0

            if self.policy == self.policy_old:
                print(f"Policy Iteration converged in {iter+1} iterations.")
                break
        return self.policy, self.v_state

    def value_iter(self):
        for iter in range(self.max_iter):
            vs_diff_max = float('inf')
            vs_iter_cnt = 0
            while vs_diff_max > self.theta:
                vs_diff_max = 0
                for s in range(self.state_num):
                    vs_max = float('-inf')
                    for action in range(self.action_num):
                        vs_tmp = 0
                        for prob, next_s, reward, done in self.env.P[s][action]:
                            vs_tmp += prob * (reward + self.gamma * self.v_state_old[next_s])
                        vs_max = max(vs_max, vs_tmp)
                    self.v_state[s] = vs_max # 值迭代只计算最大action的状态价值，而不是求期望状态价值
                    vs_diff_max = max(vs_diff_max, abs(self.v_state[s] - self.v_state_old[s]))
                print("state array:\n", self.v_state.reshape((4, 4)))
                if vs_diff_max < self.theta:
                    print(f"Value Iteration {iter+1}, Value Function converged in {vs_iter_cnt} iterations.")
                    break
                vs_iter_cnt += 1
                self.v_state_old = self.v_state.copy()

            for s in range(self.state_num):
                q_sa = np.zeros(self.action_num)
                for action in range(self.action_num):
                    for prob, next_s, reward, done in self.env.P[s][action]:
                        q_sa[action] += prob * (reward + self.gamma * self.v_state[next_s])
                best_a = np.argwhere(q_sa == np.max(q_sa)).flatten().tolist()
                best_a_num = len(best_a)
                for action in range(self.action_num):
                    self.policy[s][action] = 1 / best_a_num if action in best_a else 0
            if self.policy == self.policy_old:
                print(f"Value Iteration converged in {iter+1} iterations.")
                break
            self.policy_old = [[self.policy[s][a] for a in range(self.action_num)] for s in range(self.state_num)]  
        return self.policy, self.v_state




# output
DPRL_agent = DPRL(env, GAMMA, THETA, MAX_ITER)
if POLICY_ITER is True:
    policy, v_state = DPRL_agent.policy_iter()
else:
    policy, v_state = DPRL_agent.value_iter()
print("Optimal Policy (0: Left, 1: Down, 2: Right, 3: Up):")
print(np.array(policy).argmax(axis=1).reshape((4, 4)))
policy_fig = [['←' if a==0 else '↓' if a==1 else '→' if a==2 else '↑' for a in np.array(policy).argmax(axis=1)]]
print(np.array(policy_fig).reshape((4, 4)))
print("Optimal State Value Function:")
print(v_state.reshape((4, 4)))

# test
done = False
state = env.reset()
while not done:
    action = np.array(policy).argmax(axis=1)[state]
    state, reward, done, info = env.step(action)
    env.render()
if reward == 1:
    print("Reached the goal!")
else:
    print("Fell into a hole!")
env.close()

                    

                


