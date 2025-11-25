from torchcule.atari import Env
import time
import gym
import atari_py
import os
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from lib import replay_buffer
from lib import agent

color = 'gray'
rescale = True
frameskip = 4
train = True
episodic_life = False
record_video = True
track = True 
test_step = 200
num_envs = 4
device = 'cuda:0'
env_name = 'BreakoutNoFrameskip-v4'
run_name = f"{env_name}_{os.path.basename(__file__)[:-len('.py')]} \
                _{int(time.time())}"
"""
[实验结果]
CULE CUDA Env FPS Max: 220000, env_num~6000
CULE CPU Env FPS Max: 比cuda慢约10倍，比gym快很多
Gym vector Env FPS Max: 3000, env_num~5
"""


def speed_test():

    # env construct
    cule_cuda_envs = Env(env_name, num_envs, device=device, color_mode='gray', 
                        rescale=True, episodic_life=False)
    obs = cule_cuda_envs.reset()

    cule_cpu_envs = Env(env_name, num_envs, device='cpu')
    obs = cule_cpu_envs.reset()

    def make_env_fn(rank):
        def _thunk():
            env = gym.make(env_name)
            return env
        return _thunk
    gym_envs = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    obs = gym_envs.reset()

    print("   ENV                  FPS              TOTAL TIME")
    print("=========================================================")

    # gpu cule test
    gpu_start = time.time()
    for _ in range(test_step):
        acts = cule_cuda_envs.sample_random_actions()
        obs, rewards, dones, infos = cule_cuda_envs.step(acts)
    gpu_end = time.time()
    print(f'CULE CUDA \
            {(num_envs * test_step / (gpu_end - gpu_start)):.2f}, \
            {(gpu_end - gpu_start):.2f}')
    # cpu cule test
    cpu_start = time.time()
    for _ in range(test_step):
        acts = cule_cpu_envs.sample_random_actions()
        obs, rewards, dones, infos = cule_cpu_envs.step(acts)
    cpu_end = time.time()
    print(f'CULE CPU  \
            {(num_envs * test_step / (cpu_end - cpu_start)):.2f}, \
             {(cpu_end - cpu_start):.2f}')

    # vector gym test
    gym_start = time.time()
    for _ in range(test_step):
        acts = gym_envs.action_space.sample()
        step_out = gym_envs.step(acts)
    gym_end = time.time()
    gym_envs.close()
    print(f'GYM VECTOR \
           {(num_envs * test_step / (gym_end - gym_start)):.2f}, \
             {(gym_end - gym_start):.2f}')






def step_test():
    if track:
        wandb.init(
            project="Cule Test",
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            config={
                "frameskip": frameskip,
                "train": train,
                "record_video": record_video,
                "color_mode": color,
                "rescale": rescale,
                "episodic_life": True
            }
        )
    writer = SummaryWriter(f"runs/{run_name}")
    envs = Env(env_name, num_envs, device=device, color_mode=color, 
               rescale=rescale, episodic_life=episodic_life, frameskip=frameskip)
    if train:
        envs.train() # train会默认把frameskip设置为4，如果构造时传参的是1，此时会被覆盖为4
    obs = envs.reset()
    check_id = torch.tensor([0], device='cpu', dtype=torch.int32)
    end = False
    record_interval = 100
    video_buf = []
    step = 0
    episode_cnt = 0

    while not end:
        action = envs.sample_random_actions()
        obs, rwd, dones, infos = envs.step(action)
        video_buf.append(obs[0].permute(2,0,1))
        #check_states = envs.get_states(check_id)[0]
        end = dones[0]
        step += 1

    if record_video:
        video = torch.stack(video_buf, dim=0).unsqueeze(0).to('cpu')
        if video.shape[2] == 1: # 如果是灰度图，重复3次构建3通道
            video = video.repeat(1, 1, 3, 1, 1)
        writer.add_video("videos/obs", video, episode_cnt, fps=30)
        video_buf.clear()
    episode_cnt += 1
    writer.close()


def wrapper_test():
    if track:
        wandb.init(
            project="Cule Test",
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            config={
                "frameskip": frameskip,
                "train": train,
                "record_video": record_video,
                "color_mode": color,
                "rescale": rescale,
                "episodic_life": True
            }
        )
    writer = SummaryWriter(f"runs/{run_name}")
    envs = Env(env_name, num_envs, device=device, color_mode=color, 
               rescale=rescale, episodic_life=episodic_life, frameskip=frameskip)
    if train:
        envs.train() # train会默认把frameskip设置为4，如果构造时传参的是1，此时会被覆盖为4

    envs = gym.wrappers.FrameStack(envs, 4)
    obs = envs.reset()

    gym_env = gym.make(env_name)
    gym_env = gym.wrappers.FrameStack(gym_env, 4)
    obs_g = gym_env.reset()
    done = False
    episode_cnt = 0

    while not done:
        action = gym_env.action_space.sample()
        obs, r, done, info = gym_env.step(action)
        obs_np = np.array(obs)

    end = False
    video_buf = []
    while not end:
        action = envs.sample_random_actions()
        obs, rwd, dones, infos = envs.step(action)
        obs_list = [obs[i] for i in range(4)]
        obs_rcd = obs_list[0][0]
        video_buf.append(obs_rcd.permute(2, 0, 1).cpu())
        #check_states = envs.get_states(check_id)[0]
        end = dones[0]

    if record_video:
        video = torch.stack(video_buf, dim=0).unsqueeze(0).to('cpu')
        if video.shape[2] == 1: # 如果是灰度图，重复3次构建3通道
            video = video.repeat(1, 1, 3, 1, 1)
        writer.add_video("videos/framestack0", video, episode_cnt, fps=30)
        video_buf.clear()
    episode_cnt += 1
    writer.close()

def rb_test():
    if track:
        wandb.init(
            project="Cule Test",
            group="rb",
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            config={
                "frameskip": frameskip,
                "train": train,
                "record_video": record_video,
                "color_mode": color,
                "rescale": rescale,
                "episodic_life": True
            }
        )
    writer = SummaryWriter(f"runs/{run_name}")
    envs = Env(env_name, num_envs, device=device, color_mode=color, 
               rescale=rescale, episodic_life=episodic_life, frameskip=frameskip)
    if train:
        envs.train() # train会默认把frameskip设置为4，如果构造时传参的是1，此时会被覆盖为4
    obs = envs.reset()
    args = SimpleNamespace(batch_size=32,
                           rb_capacity=150000,
                           multi_step=3,
                           framestack=4,
                           env_num=num_envs,
                           gamma=0.99,
                           seed=1111)
    rb = replay_buffer.OptimRB(image_size=(84,84), device=device, args=args)

    episode_cnt = 0
    total_steps = 1000
    for i in range(total_steps):
        action = envs.sample_random_actions()
        nxt_obs, rwd, dones, infos = envs.step(action)
        rb.add(obs.squeeze(), action, rwd, nxt_obs.squeeze(), dones)
        obs = nxt_obs
        if i >= 800:
            sample_batch = rb.sample(32)

    if record_video:
        video_buf = rb.states[0, :rb.cur_index].unsqueeze(1) # 取第一个环境，加gray通道[l,1,84,84]
        video = video_buf.unsqueeze(0).to('cpu') #[1,l,1,84,84]
        if video.shape[2] == 1: # 如果是灰度图，重复3次构建3通道
            video = video.repeat(1, 1, 3, 1, 1)
        writer.add_video("videos/rbstate0", video, episode_cnt, fps=30)

    writer.close()

def train_test():
    if track:
        wandb.init(
            project="Cule Test",
            group="train",
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            config={
                "frameskip": frameskip,
                "train": train,
                "record_video": record_video,
                "color_mode": color,
                "rescale": rescale,
                "episodic_life": episodic_life
            }
        )
    writer = SummaryWriter(f"runs/{run_name}")
    args = SimpleNamespace(batch=32,
                           rb_capacity=2000000,
                           multi_step=3,
                           framestack=4,
                           env_num=4000,
                           gamma=0.99,
                           seed=17,
                           total_steps=100000000,
                           epsilon_start=1.0,
                           epsilon_end=0.01,
                           epsilon_decay_steps=1000000,
                           cuda=True,
                           PER=False,
                           double=False,
                           dueling=False,
                           noisy=False,
                           categorical=False,
                           atoms=1,
                           vmin=0,
                           vmax=0,
                           env=env_name,
                           cule=True,
                           sync_target_steps=250,
                           resume=False,
                           lr=0.0001,
                           replay_start=80000,
                           lr_times = 1000
                           )
    envs = Env(args.env, args.env_num, device=device, color_mode=color, 
               rescale=rescale, episodic_life=True, frameskip=4)
    if train:
        envs.train(4)  # train会默认把frameskip设置为4，如果构造时传参的是1，此时会被覆盖为4
    obs = envs.reset(initial_steps=4000)
    rb = replay_buffer.OptimRB(image_size=(84, 84), device=device, args=args)
    dqn_agent = agent.DqnAgent(img_size=(84, 84), env=envs, args=args)

    state_stack = torch.zeros((args.env_num, 4, 84, 84), device=device, dtype=torch.uint8)
    state_stack[:, 0] = obs.squeeze().clone()
    state_stack[:, 1] = obs.squeeze().clone()
    state_stack[:, 2] = obs.squeeze().clone()
    state_stack[:, 3] = obs.squeeze().clone()
    video_buf = []
    rwd_buf = []
    current_avg_reward = -float('inf')
    # 新增: 为每个环境维护累计奖励张量
    episode_rewards = torch.zeros(args.env_num, device=device, dtype=torch.float32)
    best_rwd = -float('inf')
    # 新增: 记录平均奖励提升阈值
    best_avg_reward = -float('inf')
    eval_reward = -float('inf')
    start_time = time.time()
    pre_step = 0

    # 时间统计累积变量
    time_action = 0.0
    time_env_step = 0.0
    time_rb_add = 0.0
    time_stack_update = 0.0
    time_learn = 0.0
    time_iter_total = 0.0
    iter_count = 0  # 自上次 episode 结束以来的迭代次数

    episode_cnt = 0
    from time import perf_counter
    with tqdm(total=args.total_steps, initial=0, desc='Train Progress') as pbar:
        for i in range(0, args.total_steps, args.env_num):
            iter_start = perf_counter()
            iter_count += 1
            #writer.add_images("images/env0_obs", state_stack[0].float().div(255.0).unsqueeze(1), i)
            if record_video:
                video_buf.append(obs[0].permute(2,0,1).cpu())
            # Action 选择计时
            t0 = perf_counter()
            dqn_agent.target_net.reset_noise()
            action = dqn_agent.take_action(state_stack.float().div(255.0).clone())
            time_action += (perf_counter() - t0)
            # 环境 step 计时
            t0 = perf_counter()
            nxt_obs, rwd, dones, infos = envs.step(action)
            time_env_step += (perf_counter() - t0)
            # Replay Buffer 写入计时
            t0 = perf_counter()
            rb.add(obs.squeeze().clone(), action.clone(), rwd.clone(), 
                   nxt_obs.squeeze().clone(), dones.clone())
            time_rb_add += (perf_counter() - t0)
            # 奖励累加: 所有环境本步奖励累加到 episode_rewards
            episode_rewards += rwd.float()
            # 训练阶段不再记录 env0 的视频
            # 状态栈更新计时
            t0 = perf_counter()
            obs = nxt_obs.clone()
            state_stack[:, :-1].copy_(state_stack[:, 1:].clone())
            state_stack[:, -1].copy_(obs.squeeze().clone())
            if dones.any():  # 将 done 的环境的stack清0
                mask = dones.view(-1)
                obs[mask, :] *= 0
                state_stack[mask, :] *= 0
            time_stack_update += (perf_counter() - t0)
            # 学习阶段计时
            if i >= args.replay_start:
                t0 = perf_counter()
                for _ in range(args.lr_times):
                    sample_batch = rb.sample(args.batch)
                    dqn_agent.target_net.reset_noise()
                    loss = dqn_agent.learn(sample_batch)
                    dqn_agent.learn_times += 1
                    if dqn_agent.learn_times % args.sync_target_steps == 0:
                        dqn_agent.target_net.load_state_dict(dqn_agent.train_net.state_dict())
                    writer.add_scalar("charts/loss", loss.mean(), i)
                time_learn += (perf_counter() - t0)
            # 本迭代总时间
            time_iter_total += (perf_counter() - iter_start)
            # 任意环境结束 episode 时处理其累计奖励
            if dones.any():
                done_mask = dones.view(-1)
                done_indices = torch.where(done_mask)[0]
                for idx in done_indices.tolist():
                    ep_r = episode_rewards[idx].item()
                    rwd_buf.append(ep_r)
                    best_rwd = ep_r if ep_r > best_rwd else best_rwd
                    if idx == 0 and record_video:
                        video_env0 = torch.stack(video_buf, dim=0).unsqueeze(0)  # [1,T,C,H,W]
                        if video_env0.shape[2] == 1:  # 灰度复制到3通道
                            video_env0 = video_env0.repeat(1, 1, 3, 1, 1)
                        writer.add_video("videos/env0", video_env0, i, fps=30)
                        video_buf.clear()

                # 计算最近窗口平均奖励
                window = 100
                if len(rwd_buf) > 0:
                    if len(rwd_buf) < window:
                        current_avg_reward = torch.tensor(rwd_buf, device=device).float().mean().item()
                    else:
                        current_avg_reward = torch.tensor(rwd_buf[-window:], device=device).float().mean().item()
                    writer.add_scalar("charts/avg_reward", current_avg_reward, i)
                # 清零已结束环境的累计奖励
                episode_rewards[done_mask] = 0.0

            end_time = time.time()
            sps = (i - pre_step) / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
            start_time = end_time
            pre_step = i
            writer.add_scalar("charts/sps", sps, i)
            pbar.set_description(f'Steps{i}')
            pbar.set_postfix({'Avg Rwd': f'{current_avg_reward:.2f}', 'Best Rwd': f'{best_rwd:.2f}',\
                              'Eval Rwd': f'{eval_reward:.2f}', 'SPS': f'{sps:.1f}', 'Eps': f'{dqn_agent.epsilon:.2f}'})
            pbar.update(i - pbar.n)

    writer.close()



#speed_test()
#step_test()
#wrapper_test()
#rb_test()
train_test()