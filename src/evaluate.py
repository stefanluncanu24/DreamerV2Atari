import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from tqdm import tqdm

from .models.actor_critic import Actor
from .models.rssm import RSSM
from .models.vision import Encoder
from .envs.atari import Atari

def evaluate(config, checkpoint_path, num_episodes=10):
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_env = gym.make(config['env']['name'])
    env = Atari(
        env=base_env,
        frame_stack=config['env']['frame_stack'],
        action_repeat=config['env']['action_repeat']
    )
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
    deter_size = config['model']['rssm']['deter_size']
    embed_dim = config['model']['embed_dim'] 

    encoder = Encoder(embed_dim=embed_dim, in_channels=obs_shape[0]).to(device)
    rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim, device=device, **config['model']['rssm']).to(device)
    actor = Actor(stoch_size=stoch_size, deter_size=deter_size, action_dim=action_dim, **config['model']['actor_critic']).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    actor.load_state_dict(checkpoint['actor'])
    
    encoder.eval()
    rssm.eval()
    actor.eval()

    total_rewards = []
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        (prev_h, prev_z) = rssm.initial_state(1)

        while not done:
            with torch.no_grad():
                latent_state = torch.cat([prev_h, prev_z], dim=-1)
                action = actor(latent_state.detach()).sample()
                action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            obs = next_obs
            total_reward += reward

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
                obs_embed = encoder(obs_tensor)
                action_one_hot = F.one_hot(action, num_classes=action_dim).float()
                prev_h, prev_z, _, _, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)
        
        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Return: {avg_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.checkpoint, args.episodes)
