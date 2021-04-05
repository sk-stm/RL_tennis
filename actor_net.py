import torch
from torch import nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorNet, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, 64)
        self.actor_fc2 = nn.Linear(64, 32)
        self.actor_fc3 = nn.Linear(32, action_size)

    def forward(self, state):
        # get mean of action distribution
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        return torch.tanh(self.actor_fc3(x_a))
