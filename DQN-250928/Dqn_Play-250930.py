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
    parser.add_argument('-d', '--model_dir', type=str, default='./out/', help='directory to save the model, default ./out/')
    parser.add_argument('-m', '--model', type=str, default='dqn.pth', help='path to the trained model, default dqn.pth')
    parser.add_argument('-r', '--record', default='./record_video/', help='directory to save the video, default not to record')
    parser.add_argument('-v', '--visible', action='store_true', help='whether to render the environment')

    args = parser.parse_args()
    env = env_wrappers.make_env(args.model.split('_')[0], render_mode="human" if args.visible else None)
    if args.record:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record+args.model.split('.')[0], episode_trigger=lambda x: True)
    
    noisy = True if 'noisy' in args.model else False
    if 'dueling' in args.model:
        net = model.DuelingNet(env.observation_space.shape, env.action_space.n, noisy)
    else:
        net = model.QNet(env.observation_space.shape, env.action_space.n, noisy)
    net.load_state_dict(torch.load(args.model_dir+args.model, map_location=lambda storage, loc: storage, weights_only=True))
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


