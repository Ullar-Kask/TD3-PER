import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

##### HYPERPARAMETERS #####
# Number of units in the first hidden layer
fc1_units=400
# Number of units in the second hidden layer
fc2_units=300


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
        #self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        #x = self.bn1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
        #self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #x = self.bn1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)