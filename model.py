import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal

class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, std = 0.0):
        super(PPO, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, num_outputs),
        )
        self.std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        mu = self.actor(x)
        std = self.std.expand_as(mu).exp()
        dist = Normal(mu, std)
        value = self.critic(x)
        return dist, value
                

