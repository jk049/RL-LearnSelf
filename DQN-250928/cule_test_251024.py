from torchcule.atari import Env
import time
import gym
import atari_py

"""
[实验结果]
CULE CUDA Env FPS Max: 220000, env_num~6000
CULE CPU Env FPS Max: 比cuda慢约10倍，比gym快很多
Gym vector Env FPS Max: 3000, env_num~5
"""
test_step = 200
num_envs = 4
device = 'cuda:0'
env_name = 'PongNoFrameskip-v4'
#print(gym.envs.registry.keys())

cule_cuda_envs = Env(env_name, num_envs, device=device)
obs = cule_cuda_envs.reset()

gpu_start = time.time()
for _ in range(test_step):
    acts = cule_cuda_envs.sample_random_actions()
    obs, rewards, dones, infos = cule_cuda_envs.step(acts)
gpu_end = time.time()
print(f'[INFO] CULE CUDA Env FPS: {(num_envs * test_step / (gpu_end - gpu_start)):.2f}, Total Time: {(gpu_end - gpu_start):.2f}')


cule_cpu_envs = Env(env_name, num_envs, device='cpu')
obs = cule_cpu_envs.reset()
cpu_start = time.time()
for _ in range(test_step):
    acts = cule_cpu_envs.sample_random_actions()
    obs, rewards, dones, infos = cule_cpu_envs.step(acts)
cpu_end = time.time()
print(f'[INFO] CULE CPU Env FPS: {(num_envs * test_step / (cpu_end - cpu_start)):.2f}, Total Time: {(cpu_end - cpu_start):.2f}')

def make_env_fn(rank):
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk
gym_envs = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
obs = gym_envs.reset()
gym_start = time.time()
for _ in range(test_step):
    acts = gym_envs.action_space.sample()
    obs, rewards, dones, _, infos = gym_envs.step(acts)
gym_end = time.time()
gym_envs.close()
print(f'[INFO] Gym VectorEnv FPS: {(num_envs * test_step / (gym_end - gym_start)):.2f}, Total Time: {(gym_end - gym_start):.2f}')