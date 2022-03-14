import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, std = 0.0):
        super(PPO, self).__init__()

        self.critic = nn.Sequential(
            # nn.Linear(num_inputs, 256),
            # layer_init(nn.Linear(num_inputs, 64)),
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            # layer_init(nn.Linear(64, 64)),
            nn.Linear(64, 64),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(64, 1)
            # layer_init(nn.Linear(64, 1), std = 1.0)
        )

        self.actor = nn.Sequential(
            # nn.Linear(num_inputs, 256),
            # layer_init(nn.Linear(num_inputs, 64)),
            nn.Linear(num_inputs, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(64, 64),
            # layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(64, num_outputs)
            # layer_init(nn.Linear(64, num_outputs), std = 0.01)
        )
        self.std = nn.Parameter(torch.zeros(1, num_outputs) * std)

    def forward(self, x):
        mu = self.actor(x)
        std = self.std.expand_as(mu)
        action_std = torch.exp(std)
        dist = Normal(mu, action_std)
        value = self.critic(x)
        return dist, value

