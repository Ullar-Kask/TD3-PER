import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from PER import PER

import torch
import torch.nn.functional as F
import torch.optim as optim

##### HYPERPARAMETERS #####
# Replay buffer size
BUFFER_SIZE = 2**20
# Minibatch size
BATCH_SIZE = 1024
# Discount factor
GAMMA = 0.99
# Target parameters soft update factor
TAU = 4e-3
# Learning rate of the actor network, often 1e-4
LR_ACTOR = 5e-4
# Learning rate of the critic network, often 1e-3
LR_CRITIC = 5e-4
# L2 weight decay
WEIGHT_DECAY = 0.0
# How many times networks are updated in one go
LEARN_BATCH = 10
# The actor is updated after every so many times the critic is updated (Delayed Policy Updates)
UPDATE_ACTOR_EVERY = 2
# Std dev of Gaussian noise added to action policy (Target Policy Smoothing Regularization)
POLICY_NOISE = 0.2
# Clip boundaries of the noise added to action policy
POLICY_NOISE_CLIP = 0.5


actor_weights_file = 'weights_actor.pt'
critic1_weights_file = 'weights_critic1.pt'
critic2_weights_file = 'weights_critic2.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network (w/ Target Network)
        self.critic1_local = Critic(state_size, action_size).to(device)
        self.critic1_target = Critic(state_size, action_size).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.critic2_local = Critic(state_size, action_size).to(device)
        self.critic2_target = Critic(state_size, action_size).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(action_size)
        
        # Replay memory
        self.memory = PER(BUFFER_SIZE)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory."""
        # Set reward as initial priority, see:
        #   https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
        self.memory.add((state, action, reward, next_state, done), reward)
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1., 1.)
    
    def reset(self):
        self.noise.reset()
    
    def mse(self, expected, targets, is_weights):
        """Custom loss function that takes into account the importance-sampling weights."""
        td_error = expected - targets
        weighted_squared_error = is_weights * td_error * td_error
        return torch.sum(weighted_squared_error) / torch.numel(weighted_squared_error)
    
    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        for i in range(1, LEARN_BATCH+1):
            idxs, experiences, is_weights = self.memory.sample(BATCH_SIZE)
            
            states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            
            is_weights =  torch.from_numpy(is_weights).float().to(device)
            
            # ---------------------------- update critic ---------------------------- #
            # Target Policy Smoothing Regularization: add a small amount of clipped random noises to the selected action
            if POLICY_NOISE > 0.0:
                noise = torch.empty_like(actions).data.normal_(0, POLICY_NOISE).to(device)
                noise = noise.clamp(-POLICY_NOISE_CLIP, POLICY_NOISE_CLIP)
                # Get predicted next-state actions and Q values from target models
                actions_next = (self.actor_target(next_states) + noise).clamp (-1., 1.)
            else:
                # Get predicted next-state actions and Q values from target models
                actions_next = self.actor_target(next_states)
            
            # Error Mitigation
            Q_targets_next = torch.min(\
                self.critic1_target(next_states, actions_next), \
                self.critic2_target(next_states, actions_next)).detach()
            
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
            
            # Compute critic1 loss
            Q_expected = self.critic1_local(states, actions)
            errors1 = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
            critic1_loss = self.mse(Q_expected, Q_targets, is_weights)
            # Minimize the loss
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1_local.parameters(), 1)
            self.critic1_optimizer.step()
            
            # Update priorities in the replay buffer            
            self.memory.batch_update(idxs, errors1)
            
            # Compute critic2 loss
            Q_expected = self.critic2_local(states, actions)
            critic2_loss = self.mse(Q_expected, Q_targets, is_weights)
            # Minimize the loss
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2_local.parameters(), 1)
            self.critic2_optimizer.step()
            
            # Delayed Policy Updates
            if i % UPDATE_ACTOR_EVERY == 0:
                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = self.actor_local(states)
                actor_loss = -self.critic1_local(states, actions_pred).mean()
                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # ----------------------- update target networks ----------------------- #
                self.soft_update(self.critic1_local, self.critic1_target, TAU)
                self.soft_update(self.critic2_local, self.critic2_target, TAU)
                self.soft_update(self.actor_local, self.actor_target, TAU)                     
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save_weights(self):
        torch.save(self.actor_local.state_dict(), actor_weights_file)
        torch.save(self.critic1_local.state_dict(), critic1_weights_file)
        torch.save(self.critic2_local.state_dict(), critic2_weights_file)
    
    def load_weights(self):
        self.actor_local.load_state_dict(torch.load(actor_weights_file))
        self.critic1_local.load_state_dict(torch.load(critic1_weights_file))
        self.critic2_local.load_state_dict(torch.load(critic2_weights_file))

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
