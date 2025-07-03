import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import wandb
import gymnasium as gym
from tqdm import tqdm
import torchvision

from .models.actor_critic import Actor, Critic
from .models.rssm import RSSM
from .models.vision import Decoder, Encoder
from .models.heads import RewardPredictor, DiscountPredictor
from .models.behavior import ImagBehavior
from .replay.replay_buffer import ReplayBuffer
from .envs.atari import Atari
from .utils.tools import lambda_return
from .utils.image_saver import save_reconstruction_predictions
from .utils.perceptual_loss import VGGPerceptualLoss
import torchvision

import ale_py

def main(config):
    if config['logging']['wandb']:
        wandb.init(project="dreamerv2-pytorch-refactored", config=config)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_env = gym.make(config['env']['name'])
    env = Atari(
        env=base_env,
        frame_stack=config['env']['frame_stack'],
        action_repeat=config['env']['action_repeat']
    )
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=config['replay']['capacity'],
        sequence_length=config['replay']['sequence_length'],
        batch_size=config['training']['batch_size'],
        observation_shape=obs_shape,
        action_dim=1,  
        device=device
    )

    stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
    deter_size = config['model']['rssm']['deter_size']

    encoder = Encoder(embed_dim=config['model']['embed_dim'], in_channels=obs_shape[0]).to(device)
    rssm = RSSM(action_dim=action_dim, embed_dim=config['model']['embed_dim'], device=device, **config['model']['rssm']).to(device)
    decoder = Decoder(stoch_size=stoch_size, deter_size=deter_size, out_channels=obs_shape[0]).to(device)
    reward_predictor = RewardPredictor(stoch_size=stoch_size, deter_size=deter_size, hidden_size=config['model']['actor_critic']['hidden_size']).to(device)
    discount_predictor = DiscountPredictor(stoch_size=stoch_size, deter_size=deter_size, hidden_size=config['model']['actor_critic']['hidden_size']).to(device)
    actor = Actor(stoch_size=stoch_size, deter_size=deter_size, action_dim=action_dim, **config['model']['actor_critic']).to(device)
    critic = Critic(stoch_size=stoch_size, deter_size=deter_size, **config['model']['actor_critic']).to(device)
    
    # perceptual_loss_fn = VGGPerceptualLoss().to(device)  # Using VGG for perceptual loss


    world_model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters()) + list(reward_predictor.parameters()) + list(discount_predictor.parameters())

    behavior = ImagBehavior(config, rssm, actor, critic, device)

    world_model_optimizer = torch.optim.Adam(world_model_params, lr=float(config['training']['world_model_lr']), weight_decay=float(config['training']['weight_decay']))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config['training']['actor_lr']), weight_decay=float(config['training']['weight_decay']))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config['training']['critic_lr']), weight_decay=float(config['training']['weight_decay']))
    scaler = torch.amp.GradScaler(enabled=config['logging']['amp'])

    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_count = 0
    (prev_h, prev_z) = rssm.initial_state(1)

    for frame in tqdm(range(1, config['training']['total_env_frames'] + 1), desc="Training Progress"):
        with torch.no_grad():
            latent_state = torch.cat([prev_h, prev_z], dim=-1)
            action = actor(latent_state.detach()).sample()
            action_np = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        replay_buffer.add(obs, action_np, reward, done)
        obs = next_obs
        total_reward += reward

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
            obs_embed = encoder(obs_tensor)
            action_one_hot = F.one_hot(action, num_classes=action_dim).float()
            prev_h, prev_z, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)

        if done:
            episode_count += 1
            if config['logging']['wandb']:
                wandb.log({'episode_reward': total_reward, 'episode': episode_count, 'frame': frame})
            obs, _ = env.reset()
            done = False
            total_reward = 0
            (prev_h, prev_z) = rssm.initial_state(1)

        # --- Train World Model and Actor-Critic ---
        if frame > config['training']['burn_in_frames'] and frame % config['training']['world_model_update_interval'] == 0:
            batch = replay_buffer.sample()
            if batch is not None:
                world_model_optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=config['logging']['amp']):
                    obs_batch = batch['observations']
                    action_batch = F.one_hot(batch['actions'].long().squeeze(-1), num_classes=action_dim).float()
                    reward_batch = batch['rewards'].unsqueeze(-1)
                    if config['env'].get('clip_rewards') == 'tanh':
                        reward_batch = torch.tanh(reward_batch)
                    done_batch = batch['dones'].unsqueeze(-1)

                    embedded_obs = encoder(obs_batch.view(-1, *obs_shape)).view(
                        config['training']['batch_size'], config['replay']['sequence_length'], -1
                    )

                    (h, z) = rssm.initial_state(config['training']['batch_size'])
                    h_states, z_states, kl_losses = [], [], []

                    for t in range(config['replay']['sequence_length']):
                        h, z, kl_loss = rssm(embedded_obs[:, t], action_batch[:, t], h, z)
                        h_states.append(h)
                        z_states.append(z)
                        kl_losses.append(kl_loss)

                    h_states = torch.stack(h_states, dim=1)
                    z_states = torch.stack(z_states, dim=1)
                    kl_loss = torch.stack(kl_losses).mean()
                    
                    latent_states = torch.cat([h_states, z_states], dim=-1)
                    
                    recon_pred = decoder(latent_states.view(-1, latent_states.shape[-1])) 
                    # recon_loss = perceptual_loss_fn(recon_pred, obs_batch.view(-1, *obs_shape))  # Using VGG perceptual loss
                    recon_loss = F.mse_loss(recon_pred, obs_batch.view(-1, *obs_shape))

                    if config['logging'].get('save_recon_images', False) and frame >= config['logging'].get('save_recon_after_frames', 0) and frame <= config['logging'].get('stop_save_recon_after_frames', np.inf):
                        save_reconstruction_predictions(
                            recon_pred.detach()[:8],
                            config['logging']['recon_image_dir'],
                            frame,
                            0 # take batch 0
                        ) 
                    
                    reward_pred = reward_predictor(latent_states)  
                    reward_loss = F.mse_loss(reward_pred, reward_batch)
                    
                    discount_dist = discount_predictor(latent_states)
                    discount_target = (1.0 - done_batch.float())
                    discount_loss = -discount_dist.log_prob(discount_target).mean()

                    kl_free_nats = config['model']['rssm'].get('kl_free', 0.0)
                    kl_loss = torch.max(torch.tensor(kl_free_nats, device=device), kl_loss)
                    world_model_loss = recon_loss * config['model']['recon_scale'] + reward_loss + discount_loss + config['model']['rssm']['kl_beta'] * kl_loss

                scaler.scale(world_model_loss).backward()
                scaler.unscale_(world_model_optimizer)
                torch.nn.utils.clip_grad_norm_(world_model_params, config['training']['model_grad_clip'])
                scaler.step(world_model_optimizer)
                scaler.update()

                
                # # --- Actor-Critic Update ---
                # actor_optimizer.zero_grad()
                # critic_optimizer.zero_grad()
                # with torch.amp.autocast(device_type=device.type, enabled=config['logging']['amp']):
                #     # --- Imagination and Actor-Critic Training ---
                #     # Select a random start state from the sequence
                #     start_indices = np.random.randint(0, h_states.shape[1] - config['training']['imagination_horizon'])
                #     h_start_states = h_states[:, start_indices].detach()
                #     z_start_states = z_states[:, start_indices].detach()

                #     actor_loss, critic_loss, entropy = behavior.train_step(h_start_states, z_start_states, reward_predictor, discount_predictor, frame)

                # scaler.scale(actor_loss).backward()
                # scaler.scale(critic_loss).backward()
                # scaler.unscale_(actor_optimizer)
                # scaler.unscale_(critic_optimizer)
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), config['training']['actor_grad_clip'])
                # torch.nn.utils.clip_grad_norm_(critic.parameters(), config['training']['value_grad_clip'])
                # scaler.step(actor_optimizer)
                # scaler.step(critic_optimizer)
                # scaler.update()

                if config['logging']['wandb']:
                    wandb.log({
                        'world_model_loss': world_model_loss.item(),
                        'recon_loss': recon_loss.item(),
                        'kl_loss': kl_loss.item(),
                        'reward_loss': reward_loss.item(),
                        'discount_loss': discount_loss.item(),
                        # 'actor_loss': actor_loss.item(),
                        # 'critic_loss': critic_loss.item(),
                        # 'entropy': entropy,
                        'frame': frame
                    })

        # # --- Checkpointing ---
        # if frame % config['training']['checkpoint_interval_steps'] == 0 and frame > 0:
        #     checkpoint_dir = pathlib.Path(config['logging']['checkpoint_dir'])
        #     checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #     checkpoint_path = checkpoint_dir / f"model_{frame}.pth"
        #     torch.save({
        #         'encoder': encoder.state_dict(),
        #         'rssm': rssm.state_dict(),
        #         'decoder': decoder.state_dict(),
        #         'reward_predictor': reward_predictor.state_dict(),
        #         'discount_predictor': discount_predictor.state_dict(),
        #         'actor': actor.state_dict(),
        #         'critic': critic.state_dict(),
        #         'world_model_optimizer': world_model_optimizer.state_dict(),
        #         'actor_optimizer': actor_optimizer.state_dict(),
        #         'critic_optimizer': critic_optimizer.state_dict(),
        #         'scaler': scaler.state_dict(),
        #         'frame': frame,
        #     }, checkpoint_path)
        #     print(f"\nSaved checkpoint to {checkpoint_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/breakout.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['seed'] = config.get('seed', 42)
    config['env']['name'] = config['env'].get('name', 'ALE/Breakout-v5')
    main(config)

