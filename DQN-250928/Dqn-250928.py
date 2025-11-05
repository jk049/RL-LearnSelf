#!/usr/bin/env python3
"""
DQN算法
【问题描述】：用DQN解决Pong问题
【算法原理】：用NN近似Q(s,a), 辅以经验回放和目标网络
【伪代码】：
    定义并初始化replay buffer: replay_buffer<-np.array(shape=10000, dtype=(s, a, r, s', done))  
    定义并初始化QNet
    定义并初始化dqn agent: take_action(), learn()
    for e = 1->MAX_EPISODES:
        环境复位：s <- env.reset
        while not done:
            agent选择动作：a <- agent.take_action(s)
            环境执行动作：s', r, done, info <- env.step(a)
            replay buffer 添加新样本(s, a, r, s', done)
            s <- s'
            if replay buffer样本数大于阈值：
                随机采样训练样本：batch_sample <- replay_buffer.sample()
                agent 学习：dqn agent learn(
                    训练网络选择动作：Q_sa <- agent.train_net(s, a)
                    目标网络选择动作：target_Qsa <- R + gamma * max_a' agent.target_net(s')
                    计算损失：loss = MSE(Q_sa, target_Qsa)
                    反向传播，更新训练网络：backprop loss
                    每隔C步，target_net<-train_net参数 
                )

        end while
    end for
【实验结果】：
    1. RB容量影响重大，大容量的训练速度快很多
    2. PER use less samples to achieve same performance, but each step is slower

"""
 
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gym
try: 
    import torchcule
    from torchcule.atari import Env
    CULE_AVAILABLE = True
except ImportError:
    CULE_AVAILABLE = False

from lib import replay_buffer
from lib import model
from lib import env_wrappers

class DqnAgent:
    def __init__(self, img_size, env, action_space, args):
        self.epsilon = args.epsilon_start
        self.env_num = args.env_num
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.gamma = args.gamma
        self.n_step = args.multi_step
        self.eps_decay_slopes = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_steps
        self.PER = args.PER
        self.double = args.double
        self.noisy = args.noisy
        self.categorical = args.categorical
        self.atoms = args.atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atom_values = torch.linspace(self.vmin, self.vmax, self.atoms).to(self.device)  # shape: [atoms,]
        suffixes = [
            '_double' if args.double else '',
            '_dueling' if args.dueling else '',
            '_PER' if args.PER else '',
            '_noisy' if args.noisy else '',
            '_categorical' if args.categorical else ''
        ]
        self.model_save_name = f"{args.env}{''.join(suffixes)}.pth"
        self.train_net = model.QNet(img_size, action_space, args).to(self.device)
        self.target_net = model.QNet(img_size, action_space, args).to(self.device)
        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=args.lr)
        self.learn_times = 1 if args.env_num <= args.batch_size else args.env_num // args.batch_size
        self.cule = args.cule
        self.env = env
        self.sync_interval = args.sync_target_steps
        self.learn_times = 0
        print(self.train_net)
        if args.resume:
            resume_path = args.resume_path if args.resume_path is not None else args.save_path + self.model_save_name
            self.train_net.load_state_dict(torch.load(resume_path, map_location=self.device, weights_only=True))
            print(f"[INFO] Resumed model from {resume_path}")
        self.target_net.load_state_dict(self.train_net.state_dict())
    
    def take_action(self, state):
        if rng.random() < self.epsilon:
            if self.cule:
                action = self.env.sample_random_actions()
            else:
                action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                if self.categorical:
                    qsa_probs = self.train_net(state)
                    atoms = self.atom_values.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, atoms]
                    qsa = (qsa_probs * atoms).sum(-1)  # shape: [env_num, action_space]
                    action = qsa.argmax(dim=1) # [env_num]
                else:
                    action = self.train_net(state).argmax(dim=1) 

            if not self.cule:
                action = action.cpu().numpy()

        self.epsilon = max(args.epsilon_end, self.epsilon - self.eps_decay_slopes)


        return action
                                           
    def dist_projection(self, dist, rewards, dones):
        delta = (self.vmax - self.vmin) / (self.atoms - 1)
        rewards = rewards.unsqueeze(1).expand(-1, self.atoms)      # [B, atoms]
        dones = dones.unsqueeze(1).expand(-1, self.atoms)          # [B, atoms]
        atom_values = self.atom_values.unsqueeze(0)                # [1, atoms]
        discount_rewards = rewards + (self.gamma ** self.n_step) * (~dones).float() * atom_values  # [B, atoms]
        discount_rewards = discount_rewards.clamp(self.vmin, self.vmax)
        b = (discount_rewards - self.vmin) / delta                 # [B, atoms]
        l = b.floor().long()                                       # [B, atoms]
        u = b.ceil().long()                                        # [B, atoms]
        proj_dist = torch.zeros_like(dist)                         # [B, atoms]

        dist_l = (u - b + (l == b).float()) * dist      # l==b考虑了整点的情况
        dist_u = (b - l) * dist

        # scatter_add_: self[i, index[i,j]] += src[i,j] dim=1
        proj_dist.scatter_add_(1, l, dist_l)
        proj_dist.scatter_add_(1, u, dist_u)
        return proj_dist

    def get_train_qsa(self, state_batch, action_batch):
        train_q_values = self.train_net(state_batch) # shape: [B, action_space]
        if self.categorical:
            index = action_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.atoms) # shape: [B, 1, atoms]
            train_q_sa = train_q_values.gather(dim=1, index=index).squeeze(1) # shape: [B, atoms]
        else:
            train_q_sa = train_q_values.gather(dim=1, index=action_batch.unsqueeze(-1)).squeeze(-1) # [B,]
        return train_q_sa
    
    def get_tgt_qsa(self, next_state_batch, reward_batch, done_batch):
        with torch.no_grad():
            if self.categorical:
                if self.double:
                    next_qsa_probs = self.train_net(next_state_batch) # [B, act_num, atoms]]
                else:
                    next_qsa_probs = self.target_net(next_state_batch) # [B, act_num, atoms]]

                atoms = self.atom_values.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, atoms]
                next_action = (next_qsa_probs * atoms).sum(-1).argmax(1)  # [B,]
                index = next_action.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.atoms)  # [B, 1, atoms]
                target_q_next = self.target_net(next_state_batch).gather(dim=1, index=index).squeeze(1)  # [B, atoms]
                target_q_sa = self.dist_projection(target_q_next, reward_batch, done_batch)  # [B, atoms]
            else:
                if self.double:
                    next_action = self.train_net(next_state_batch).argmax(1)  # [B,]
                    target_q_next = self.target_net(next_state_batch).gather(dim=1, index=next_action.unsqueeze(-1)).squeeze(-1)  # [B,]
                else:
                    target_q_next = self.target_net(next_state_batch).max(1)[0]  # [B,]
                target_q_next *= ~done_batch  # if done, Q(s')=0
                target_q_sa = reward_batch + (self.gamma ** self.n_step) * target_q_next  # [B,]
        return target_q_sa

    def loss(self, tgt_qsa, train_qsa, weights=None):
        if self.categorical:
            # loss = -sum(tgt_p * log(train_p))
            log_train_qsa = torch.log(train_qsa.clamp(1e-5, 1 - 1e-5))  # shape: [B, atoms]
            loss = -(tgt_qsa * log_train_qsa).sum(dim=1) # shape: [B,]
        else:
            loss = 0.5 * (tgt_qsa - train_qsa).pow(2) # shape: [B,]
        if self.PER:
            loss *= weights
        return loss

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch[:5]
        weights = batch[-1] if self.PER else None
        train_q_sa = self.get_train_qsa(states, actions)
        target_q_sa = self.get_tgt_qsa(next_states, rewards, dones)
        loss = self.loss(target_q_sa, train_q_sa, weights)
        loss_mean = loss.mean()

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return loss.detach()

def logger(reward, avg_reward, loss, avg_loss, epsilon, avg_fps, time, episode):
    pass

def make_env_fn(rank):
    def _thunk():
        env = env_wrappers.make_env(args.env, train=True, orin_pic=(args.optim_rb or args.cule_rb))
        try:
            env.seed(args.seed + rank)
        except Exception:
            pass
        return env
    return _thunk

def make_env(args):
    if args.cule and CULE_AVAILABLE:
        # cule example创建train env时候frameskip是1，test env的frameskip是4. why?
        # obs: [env_num, h, w, c]
        env = Env(args.env, args.env_num, device = 'cuda:0', color_mode='gray', 
                  rescale=True, episodic_life=True, frameskip=4, repeat_prob=0.0)
        env.train() # 设置为训练模式，reset时执行50//frameskip次随机动作
        action_space = env.action_space.n
        seeds = [args.seed + i for i in range(args.env_num)]
        seeds = torch.tensor(seeds, device='cuda:0', dtype=torch.int32)
        # cule example reset入参initial_steps=4000
        state = env.reset(seeds=seeds, initial_steps=4000).clone() # 如果是cule env，reset会默认执行50步随机动作, 可传参指定随机步数
        image_size = env.observation_space.shape[1:3]
    else:
        # sync: 主进程依此调用各环境，适合step快的简单环境；async: 每个环境独立进程，适合step慢的复杂环境
        env = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(args.env_num)])
        action_space = env.single_action_space.n
        state = env.reset() 
        state = torch.from_numpy(np.array(state)).to(device=device) # shape:(env_num,4,84,84)
        if args.optim_rb or args.cule_rb:
            image_size = env.single_observation_space.shape[0:2] 
        else:
            image_size = env.single_observation_space.shape[1:]
        if args.cule and not CULE_AVAILABLE:
            print("[WARNING] CULE is not available. Falling back to Gym.")
            args.cule = False
    
    if args.optim_rb or args.cule_rb:
        state_stack = torch.zeros((args.env_num, args.framestack, *image_size), device=device, dtype=torch.float32)
    else:
        state_stack = None
    return env, action_space, image_size, state, state_stack

def make_rb(image_size, obs_shape, device, args):
    if args.cule_rb:
        rb = replay_buffer.CuleRB(args, args.rb_capacity, device)
    elif args.cule or args.optim_rb:
        rb = replay_buffer.OptimRB(args.env_num, args.rb_capacity, image_size, device, 
                                  batch_size=args.batch_size, multi_step=args.multi_step, 
                                  framestack=args.framestack, gamma=args.gamma)
    elif args.PER:
        rb = replay_buffer.PrioRB(args.rb_capacity, obs_shape, device, 
                                  alpha=args.rb_alpha, beta=args.rb_beta, eps=args.rb_eps)
    else:
        rb = replay_buffer.RB(args.rb_capacity, obs_shape, device)
    
    return rb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17, help='random seed, default 42')
    parser.add_argument('--cuda', default=True, action='store_true', help='enable cuda')
    parser.add_argument('--cule', default=False, action='store_true', help='use cule environment if available')
    parser.add_argument('--optim_rb', default=False, action='store_true', help='enable optimized replay buffer for cule env')
    parser.add_argument('--cule_rb', default=False, action='store_true', help='use cule optimized replay buffer')
    parser.add_argument('--multi_step', type=int, default=1, help='number of steps for multi-step return, default 3')
    parser.add_argument('--lr_times_per_step', type=int, default=3, help='number of learning updates per environment step, default 1')
    parser.add_argument('--env', default='PongNoFrameskip-v4', help='gym environment name, default PongNoFrameskip-v4')
    parser.add_argument('--env_num', type=int, default=3, help='number of parallel environments, default 1')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for reward, default 0.99')
    parser.add_argument('--batch_size', type=int, default=32, help='number of transitions sampled from replay buffer, default 32')
    parser.add_argument('--framestack', type=int, default=4, help='number of frames stacked as input to the network, default 4')
    parser.add_argument('--rb_capacity', type=int, default=10000, help='capacity of replay buffer, default 10000')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default 1e-4')
    parser.add_argument('--sync_target_steps', type=int, default=1000, help='number of frames between target network sync, default 1000 steps')
    parser.add_argument('--replay_start_size', type=int, default=3000, help='number of transitions that must be in the replay buffer before starting training, default 10000')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='starting value of epsilon for epsilon-greedy action selection, default 1.0')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='ending value of epsilon for epsilon-greedy action selection, default 0.01')
    parser.add_argument('--epsilon_decay_steps', type=int, default=10000, help='number of steps over which to decay epsilon, default 150000')
    parser.add_argument('--rwd_bound', type=float, default=19.0, help='reward boundary for stopping criterion, default 19.0')
    parser.add_argument('--max_timestep', type=int, default=1000000000, help='maximum number of training timestep, default 1000000')
    parser.add_argument('--save_path', type=str, default='./out/', help='path to save the trained model, default ./out/')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from saved model')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load the saved model')
    parser.add_argument('--double', default=False, action='store_true', help='enable double dqn')
    parser.add_argument('--dueling', default=False, action='store_true', help='enable dueling dqn')
    parser.add_argument('--PER', default=False, action='store_true', help='enable prioritized experience replay')
    parser.add_argument('--rb_alpha', type=float, default=0.6, help='alpha parameter for prioritized experience replay, default 0.6')
    parser.add_argument('--rb_beta', type=float, default=0.4, help='beta parameter for prioritized experience replay, default 0.4')
    parser.add_argument('--rb_eps', type=float, default=1e-6, help='epsilon parameter for prioritized experience replay, default 1e-6')
    parser.add_argument('--noisy', default=False, action='store_true', help='enable noisy dqn')
    parser.add_argument('--categorical', default=False, action='store_true', help='enable categorical dqn')
    parser.add_argument('--atoms', type=int, default=51, help='number of atoms for categorical dqn, default 51')
    parser.add_argument('--vmin', type=float, default=-10.0, help='minimum value for categorical dqn, default -10.0')
    parser.add_argument('--vmax', type=float, default=10.0, help='maximum value for categorical dqn, default 10.0')
    parser.add_argument('--pbar_interval', type=int, default=1, help='interval steps for progress bar update, default 1')
    parser.add_argument('--debug', default=False, action='store_true', help='enable debug mode with smaller network and buffer for quick testing')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    env, action_space, image_size, state, state_stack = make_env(args)
    obs_shape = env.observation_space.shape[-3:]  # gym:(4, 84, 84); cule:(84, 84, 1)
    rb = make_rb(image_size, obs_shape, device, args)
    agent = DqnAgent(image_size, env, action_space, args)

    # 对基于 CuLE 的重放缓冲区进行初始观测填充，避免最早期状态全零影响采样掩码
    if args.cule_rb:
        # RB.reset 期望 [E, H, W] 且为 0~1 float；
        # - 如果用的是 CuLE env（rescale=True），state 已是 0~1 float；
        # - 如果用的是 Gym env（orin_pic=True），state 为 0~255，需要除以 255。
        with torch.no_grad():
            s0 = state.squeeze(-1).float()
            rb.reset(s0)

    default_stream = torch.cuda.default_stream() # default stream上执行env.step等操作
    train_stream = torch.cuda.Stream()  # agent学习在train_stream上执行
    learn_done = torch.cuda.Event() # agent完成当前step的learn的信号

    best_reward = -float('inf')
    rwd_mean = -float('inf')
    episode_rewards = []
    episode_reward = np.zeros(args.env_num)
    pre_lr_step = 0
    start_time = time.time()
    done_mask = torch.ones(args.env_num, dtype=torch.float32, device=device)
    with tqdm(total=args.rwd_bound, initial=0, desc='Train Progress') as pbar:
        for step in range(args.max_timestep):

            with torch.cuda.stream(train_stream):
                if step >= args.replay_start_size and (step % 4 == 0):
                    for _ in range(args.lr_times_per_step):
                        agent.train_net.reset_noise()
                        batch = rb.sample(args.batch_size)
                        """
                        start_time = time.time()
                        for i in range(20):
                            loss = agent.learn(batch)
                        end_time = time.time()
                        lps = 20 / (end_time - start_time)
                        print("LPS:", f'{lps:.2f}')
                        """
                        loss = agent.learn(batch)
                        agent.learn_times += 1
                        if agent.learn_times % agent.sync_interval == 0:
                            agent.target_net.load_state_dict(agent.train_net.state_dict())
                        if args.PER: # 如果训练用非默认stream，此处需要考虑同步问题
                            rb.update_prio(loss, batch[5]) # batch = (s, a, r, s', done, indices, weights)
                    #continue
                learn_done.record(train_stream)  # 记录learn完成事件

            # 这里的state_stack要用done掩码
            if args.cule_rb:
                # RB 期望输入 0~1 float
                state_float = state.squeeze(-1).float()
                state_stack[:, :-1] = state_stack[:, 1:].clone()
                state_stack *= done_mask.view(-1, 1, 1, 1)  # 用 done 掩码清零
                state_stack[:, -1] = state_float
                state_to_net = state_stack
                state_for_rb = state_float  # CuleRB 使用 0~1 float
                # 注意：不要把 state 转成 uint8，否则 RB 内部再次 *255 会导致像素溢出
            elif args.optim_rb:
                # 优化 RB：网络用 0~1 float，RB 存储用 uint8（先 *255 再转）
                # 考虑用copy_赋值
                state_float = state.squeeze(-1).float().div(255.0)
                state_stack[:, :-1].copy_(state_stack[:, 1:].clone())
                state_stack *= done_mask.view(-1, 1, 1, 1)  # 用 done 掩码清零
                state_stack[:, -1].copy_(state_float)
                state_to_net = state_stack.clone()
                state_for_rb = state.squeeze(-1).to(dtype=torch.uint8)  # OptimRB 使用 uint8
            else:
                state_to_net = state

            if agent.noisy:
                agent.train_net.reset_noise()
            action = agent.take_action(state_to_net) 
            next_state, reward, done, info = env.step(action)


            next_state = next_state.clone()
            done_mask = (~done).float().clone()

            if not args.cule:
                next_state = torch.from_numpy(np.array(next_state)).to(device=device, dtype=torch.float32)
                if args.optim_rb or args.cule_rb:
                    action = torch.from_numpy(action).to(device=device, dtype=torch.int64)
                    reward = torch.from_numpy(np.array(reward)).to(device=device, dtype=torch.float32)
                    done = torch.from_numpy(np.array(done)).to(device=device, dtype=torch.bool)

            if args.optim_rb or args.cule_rb:
                reward_c = reward.cpu().numpy()
                done_c = done.cpu().numpy()
            episode_reward += reward_c
            done_indices = np.where(done_c)[0]
            if done_indices.size > 0:
                episode_rewards.extend(episode_reward[done_indices].tolist())
                episode_reward[done_indices] = 0.0
                rwd_mean = np.mean(episode_rewards[-10:]) 
                if rwd_mean > best_reward:
                    best_reward = rwd_mean 
                    torch.save(agent.train_net.state_dict(), args.save_path + agent.model_save_name)
                if rwd_mean >= args.rwd_bound:
                    pbar.update(round(max(0.0, rwd_mean), 2) - pbar.n) 
                    pbar.set_postfix({'Rwd': f'{rwd_mean:.2f}', 'Best Rwd': f'{best_reward:.2f}', 'FPS': f'{fps:.1f}', 
                                      'LPS': f'{lps:.1f}', 'Eps': f'{agent.epsilon:.2f}'})
                    print(f"Solved in {step*args.env_num} steps!")
                    break


                if args.debug and (step % 1000 == 0): # 显示state_for_rb和next_state的图片
                    fig, axs = plt.subplots(2, args.framestack)
                    for i in range(args.framestack):
                        axs[0][i].imshow(batch[0][0,i].cpu().numpy(), cmap='gray')
                        axs[0][i].set_title(f's{i}')
                        axs[1][i].imshow(batch[3][0,i].cpu().numpy(), cmap='gray')
                        axs[1][i].set_title(f'n_s{i}')
                    plt.show()

            if step % args.pbar_interval == 0:
                end_time = time.time()
                lps = (agent.learn_times - pre_lr_step) / (end_time - start_time)
                fps = (args.pbar_interval * args.env_num) / (end_time - start_time)
                start_time = end_time
                pre_lr_step = agent.learn_times
                pbar.set_description(f'Steps{step}')
                pbar.set_postfix({'Rwd': f'{rwd_mean:.2f}', 'Best Rwd': f'{best_reward:.2f}', 'FPS': f'{fps:.1f}', 
                                  'LPS': f'{lps:.1f}', 'Eps': f'{agent.epsilon:.2f}'})
                pbar.update(round(max(0.0, rwd_mean), 2) - pbar.n) # 显示值保底为0，用整数更新pbar，避免 NaN/负值

            default_stream.wait_event(learn_done)
            if args.cule_rb or args.optim_rb:
                rb.add(state_for_rb.clone(), action.clone(), reward.clone(), next_state, done.clone())
            else:
                rb.add(state, action, reward, next_state, done)
            state = next_state



    # plot mean reward
    if len(episode_rewards) == 0:
        print("[INFO] No episodes were completed.")
    else:
        plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'))
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward (10 episodes)')
        plt.title('DQN on ' + args.env)
        plt.show()

    if not args.cule:
        env.close() # cule env不需要close
