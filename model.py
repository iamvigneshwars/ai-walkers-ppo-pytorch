import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal

class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, std = 0.0):
        super(PPO, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # policy net
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_outputs)
        )

        # std of action
        self.std = nn.Parameter(torch.zeros(1, num_outputs) * std)

    def forward(self, x):
        mu = self.actor(x)
        std = self.std.expand_as(mu)
        action_std = torch.exp(std)
        dist = Normal(mu, action_std)
        value = self.critic(x)
        return dist, value


