"""
【问题描述】：用Monte Carlo方法估计函数f(x) = 2x + 10sqrt(abs(x)) + 3 的期望值；
          其中，x服从正态分布N(1, 2^2), 即均值为1，方差为2^2的正态分布。
【算法设计】：
    1. 从正态分布N(1, 2^2)中采样N个样本x_i；
    2. 计算每个样本的函数值f(x_i)；
    3. 计算函数值的均值，作为期望值的估计。
【伪代码】：
    初始化样本数量N
    采样x = np.random.normal(1, 2, N)
    计算函数值f_x = 2*x + 10*sqrt(abs(x)) + 3
    计算期望值估计E_f = np.mean(f_x)
"""

import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 11
SAMPLE_NUM = 500
PLT_FLAG = True

random.seed(SEED)
np.random.seed(SEED)

cumulative_expectation = np.zeros(SAMPLE_NUM)
sampled_x = np.random.normal(1, 2, SAMPLE_NUM) # np.ramdom.noramal(均值, 标准差, 样本数量), 标准差是方差的平方根
f_x = 2 * sampled_x + 10 * np.sqrt(np.abs(sampled_x)) + 3
est_expectation = np.mean(f_x)

cumulative_expectation[0] = f_x[0]
for i in range(1, SAMPLE_NUM):
    cumulative_expectation[i] = (cumulative_expectation[i-1] * i + f_x[i]) / (i + 1)

print(f"Estimated Expectation: {est_expectation}")
if PLT_FLAG is True:
    plt.figure()
    plt.title("Cumulative Expectation Estimation")
    plt.xlabel("Sample Number")
    plt.ylabel("Estimated Expectation")
    plt.plot(cumulative_expectation, label="Estimated Expectation")
    plt.hlines(est_expectation, 0, SAMPLE_NUM-1, colors='r', linestyles='dashed', label="Final Estimated Expectation")
    plt.legend()
    plt.show()
