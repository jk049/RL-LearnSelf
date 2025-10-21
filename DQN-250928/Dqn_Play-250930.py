"""
测试训练所得模型的效果
"""

import argparse 
import gym
import torch
import os

from lib import env_wrappers
from lib import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_dir', type=str, default='./out/', help='directory to save the model, default ./out/')
    parser.add_argument('-m', '--model', type=str, default='dqn.pth', help='path to the trained model, default dqn.pth')
    parser.add_argument('-r', '--record', default='./record_video/', help='directory to save the video, default not to record')
    parser.add_argument('-v', '--visible', action='store_true', help='whether to render the environment')

    args = parser.parse_args()
    env = env_wrappers.make_env(args.model.split('_')[0], render_mode="human" if args.visible else None, train=False)
    if args.record:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record+args.model.split('.')[0], episode_trigger=lambda x: True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.noisy = True if 'noisy' in args.model else False
    args.dueling = True if 'dueling' in args.model else False
    args.categorical = True if 'categorical' in args.model else False
    args.atoms = 51 if 'categorical' in args.model else 1
    atom_values = torch.linspace(-10, 10, args.atoms).to(device)  # support
    atom_values = atom_values.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, atoms]

    net = model.QNet(env.observation_space.shape, env.action_space.n, args)
    net = net.to(device)
    checkpoint = torch.load(os.path.join(args.model_dir, args.model), map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {missing}, Unexpected keys: {unexpected}") 

    net.eval()
    state = env.reset()
    done = False
    total_reward = 0.0

    try:
        while not done:
            # make sure state and model are on the same device
            state = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                q_values = net(state)
                if args.categorical:
                    q_values = (q_values * atom_values).sum(-1)  # shape: [1, action_num]
                    action = q_values.argmax(dim=1).item()
                else:
                    action = torch.argmax(q_values, dim=1).item()
                state, reward, done, info = env.step(action)
                total_reward += reward
    finally:
        print(f"Total reward: {total_reward}")
        try:
            env.close()
        except Exception:
            # avoid exceptions from destructor during interpreter shutdown
            pass


