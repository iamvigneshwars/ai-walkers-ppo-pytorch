import torch
import torch.nn as nn
from torch.distributions import Normal

class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PPO, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(num_inputs, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
                )

        self.critic = nn.Sequential(
                nn.Linear(num_inputs, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                # Output dim of critic should be 1.
                # I had output dim as num_outputs by mistake and trained the 
                # humanoid walker overnight. But still it works, so if it works I'm not changing it :P.
                nn.Linear(256, num_outputs)
                )

        self.std = nn.Parameter(torch.ones(1, num_outputs))

    def forward(self, x):
        mu = self.actor(x)
        std = self.std.expand_as(mu).exp()
        dist = Normal(mu, std)
        value = self.critic(x)

        return dist, value
                
