import torch
import torch.nn as nn
import torch.nn.functional as F
import copy # Import copy for deepcopy

from .actor_critic import Actor, Critic
from ..utils.tools import lambda_return, schedule

class ImagBehavior(nn.Module):
    def __init__(self, config, rssm, actor, critic, device):
        super().__init__()
        self.config = config
        self.rssm = rssm
        self.actor = actor
        self.critic = critic
        self.device = device
        self.stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
        self.deter_size = config['model']['rssm']['deter_size']
        self.action_dim = actor.action_dim

        # Target network for critic stabilization
        self.critic_target = copy.deepcopy(critic)
        self.updates = 0 # Counter for tracking update steps

    def train_step(self, h_states, z_states, reward_predictor, discount_predictor, step):
        # Update the critic target network
        self._update_critic_target()

        # Detach the initial states for imagination
        imag_h_states = h_states.detach().reshape(-1, self.deter_size)
        imag_z_states = z_states.detach().reshape(-1, self.stoch_size)

        # Imagine trajectories
        imagined_h, imagined_z, imagined_actions = self._imagine_trajectories(
            imag_h_states, imag_z_states
        )

        imagined_latent_states = torch.cat([imagined_h, imagined_z], dim=-1)

        # Predict rewards, discounts, and values for the imagined trajectories
        imagined_rewards = reward_predictor(imagined_latent_states).mean
        imagined_discounts = discount_predictor(imagined_latent_states).mean
        # Use the critic_target for imagined_values
        imagined_values = self.critic_target(imagined_latent_states).mean
        # critic_dist = self.critic(imagined_latent_states[:-1])

        # Compute lambda-returns
        lambda_returns = lambda_return(
            imagined_rewards.squeeze(-1),
            imagined_values.squeeze(-1),
            imagined_discounts.squeeze(-1),
            bootstrap=imagined_values[-1].squeeze(-1),
            lambda_=self.config['training']['lambda_return']
        )

        # --- Actor Loss ---
        actor_latent_states = imagined_latent_states[:-1].detach()
        actor_dist = self.actor(actor_latent_states)
        
        # Calculate advantage using lambda returns and imagined values
        advantage = lambda_returns[:-1].detach().unsqueeze(-1) - imagined_values[:-1].detach() 
        actor_loss = -(advantage * actor_dist.log_prob(imagined_actions.detach()).unsqueeze(-1)).mean()
        
        # Add entropy reg 
        entropy_coeff = schedule(self.config['model']['actor_critic']['entropy_coeff'], step)
        actor_loss -= entropy_coeff * actor_dist.entropy().mean()

        # --- Critic Loss ---
        # Use lambda_returns[:-1] to match the critic predictions
        critic_latent_states = imagined_latent_states[:-1].detach()
        critic_dist = self.critic(critic_latent_states)
        critic_loss = -critic_dist.log_prob(lambda_returns[:-1].detach().unsqueeze(-1)).mean()

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': actor_dist.entropy().mean().item(),
            'actor_logits_std': actor_dist.logits.std().item(),
            'critic_value': critic_dist.mean.mean().item(),
            'critic_stddev': critic_dist.stddev.mean().item(),
        }

    def _imagine_trajectories(self, initial_h, initial_z):
        '''Generates imagined trajectories using the world model.'''
        imagined_h, imagined_z, imagined_actions = [initial_h], [initial_z], []

        for _ in range(self.config['training']['imagination_horizon']):
            latent_state = torch.cat([imagined_h[-1], imagined_z[-1]], dim=-1)
            action_dist = self.actor(latent_state.detach())
            action = action_dist.sample()
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
            
            imagined_actions.append(action)
            
            h, z = self.rssm.imagine(action_one_hot, imagined_h[-1], imagined_z[-1])
            imagined_h.append(h)
            imagined_z.append(z)

        # Convert lists to tensors
        imagined_h = torch.stack(imagined_h, dim=0)
        imagined_z = torch.stack(imagined_z, dim=0)
        imagined_actions = torch.stack(imagined_actions, dim=0)

        return imagined_h, imagined_z, imagined_actions

    def _update_critic_target(self):
        # Check if slow_value_target is enabled in config
        if self.config['training'].get('slow_value_target', False):
            # Update every 'slow_target_update' steps
            if self.updates % self.config['training']['slow_target_update'] == 0:
                mix = self.config['training'].get('slow_target_fraction', 1.0)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
            self.updates += 1
