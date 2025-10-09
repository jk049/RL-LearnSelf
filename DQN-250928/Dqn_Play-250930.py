"""
测试训练所得模型的效果
"""

import argparse 
import gym
import torch

from lib import env_wrappers
from lib import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='dqn.pth', help='path to the trained model, default dqn.pth')
    parser.add_argument('-e', '--env', type=str, default='PongNoFrameskip-v4', help='environment name, default PongNoFrameskip-v4')
    parser.add_argument('-r', '--record', default='./record_video/', help='directory to save the video, default not to record')
    parser.add_argument('-v', '--visible', action='store_true', help='whether to render the environment')

    args = parser.parse_args()
    env = env_wrappers.make_env(args.env, render_mode="human" if args.visible else None)
    if args.record:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record+args.env, episode_trigger=lambda x: True)
        #env = gym.wrappers.Monitor(env, args.record)
    
    if 'dueling' in args.model:
        net = model.Dueling_Net(env.observation_space.shape, env.action_space.n)
    else:
        net = model.Q_Net(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage, weights_only=True))
    net.eval()
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32)
        with torch.no_grad():
            q_values = net(state)
            action = torch.argmax(q_values, dim=1).item()
            state, reward, done, info = env.step(action)
            total_reward += reward
            
    print(f"Total reward: {total_reward}")
    env.close()


