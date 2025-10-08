"""
【问题描述】：用SARSA和Q-Learning算法解决FrozenLake问题

【环境】：gym FrozenLake-v1

【算法设计】：
    1. SARSA算法：Q(s,a) += alpha * [R(s,a) + gamma * Q(s', a') - Q(s,a)]
    2. Q-learning算法：Q(s,a) += alpha * [R(s,a) + gamma * max_a'Q(s', a') - Q(s,a)]
    3. dyna-q算法：每次更新Q-Learning后，进行n次模拟更新

【伪代码】
    env = gym.make
    Q(s,a) 初始化
    s = env.reset()
    a = agent.take_action(s)
    for episode in range(MAX_EPISODES):
        if sarsa_learn:
            s', R, ... = env.step(a)
            a' = agent.take_action(s')
            q(s, a) += alpha * [R + gamma * q(s', a') - q(s, a)]
            a, s = a', s' 
        else if q_learn:
            s', R, ... = env.step(a)
            q(s, a) += alpha * [R + gamma * max_a' q(s', a') - q(s, a)]
            s = s'
            a = agent.take_action(s)
        else if dyna_q:
            s', R, ... = env.step(a)
            q(s, a) += alpha * [R + gamma * max_a' q(s', a') - q(s, a)]
            model(s, a) = (s', R)
            s = s'
            a = agent.take_action(s)
            for n in range(dyna_n):
                s_sim, a_sim = random previously observed state and action
                s'_sim, R_sim = model(s_sim, a_sim)
                q(s_sim, a_sim) += alpha * [R_sim + gamma * max_a' q(s'_sim, a') - q(s_sim, a_sim)]
【实验结果】
    1. Q-learning需要的epsilon更大，以保证足够的探索，否则可能找不到最优策略
    2. SARSA的策略更保守，Q-learning的策略更激进。所以在能保证足够探索的前提下，Q-learning学习更快
    3. Dyna-Q收敛速度最慢，为什么？不是说此方法需要更少的样本，应该更快收敛呀。。。
        原因猜想：    a. q-planning更新的q-table不够准确，甚至是错误的，反而减慢了收敛速度
                    b. dyna-q的q-planning更新次数太少，无法显著提升
                    c. FrozenLake环境本身比较简单，dyna-q的优势无法体现
【改进方向】
    1. 增加dyna-q的q-planning更新次数: 由5增加到100，收敛速度提升明显,超越qlearning和SARSA
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 11
SLIPPERY = False
LR = 0.1
GAMMA = 0.9
MAX_EPISODES = 5000
EPSILON = 0.2
DYNA_N = 100

np.random.seed(SEED)
random.seed(SEED)


env = gym.make('FrozenLake-v1', is_slippery=SLIPPERY)

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_num = self.env.observation_space.n
        self.action_num = self.env.action_space.n
        self.q_table = np.zeros((self.state_num, self.action_num))
        self.epsilon = EPSILON
        self.alpha = LR
        self.gamma = GAMMA
        self.reward_per_episode = np.zeros(MAX_EPISODES)
        self.steps_per_episode = np.zeros(MAX_EPISODES)

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state, :])
        return action
    
    def sarsa_learn(self):
        for episode in range(MAX_EPISODES):
            cumulative_reward = 0
            step_cnt = 0
            state = self.env.reset()
            action = self.take_action(state)
            done = False
            while not done:
                next_state, reward, done, info = self.env.step(action)
                next_action = self.take_action(next_state)
                self.q_table[state, action] += self.alpha * (reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])
                state, action = next_state, next_action
                cumulative_reward += reward
                step_cnt += 1
            self.reward_per_episode[episode] = cumulative_reward
            self.steps_per_episode[episode] = step_cnt
            print(f"Episode {episode+1}: Reward = {cumulative_reward}, Steps = {step_cnt}")
    def q_learn(self):
        for episode in range(MAX_EPISODES):
            cumulative_reward = 0
            step_cnt = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.take_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
                state = next_state
                cumulative_reward += reward
                step_cnt += 1
            self.reward_per_episode[episode] = cumulative_reward
            self.steps_per_episode[episode] = step_cnt
            print(f"Episode {episode+1}: Reward = {cumulative_reward}, Steps = {step_cnt}")

    def dyna_q_learn(self):
        for episode in range(MAX_EPISODES):
            cumulative_reward = 0
            step_cnt = 0
            state = self.env.reset()
            done = False
            # Model for Dyna-Q
            model = {}
            while not done:
                action = self.take_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
                model[(state, action)] = (next_state, reward)
                state = next_state
                cumulative_reward += reward
                step_cnt += 1
                # q-planning
                for _ in range(DYNA_N):
                    if len(model) == 0:
                        break
                    (s_sim, a_sim) = random.choice(list(model.keys()))
                    s_next_sim, r_sim = model[(s_sim, a_sim)]
                    self.q_table[s_sim, a_sim] += self.alpha * (r_sim + self.gamma * np.max(self.q_table[s_next_sim, :]) - self.q_table[s_sim, a_sim])
            self.reward_per_episode[episode] = cumulative_reward
            self.steps_per_episode[episode] = step_cnt
            print(f"Episode {episode+1}: Reward = {cumulative_reward}, Steps = {step_cnt}")



    def test(self):
        state = self.env.reset()
        action = np.argmax(self.q_table[state, :])
        done = False

        while not done:
            self.env.render()
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            action = np.argmax(self.q_table[state, :])
        print("Q-table:\n", self.q_table)
    def plot_results(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.reward_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')

        plt.subplot(1, 2, 2)
        plt.plot(self.steps_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')

        plt.tight_layout()
        plt.show()
        


agent = Agent(env)
agent.sarsa_learn()
#agent.q_learn()
#agent.dyna_q_learn()
agent.plot_results()
agent.test()

env.close()

