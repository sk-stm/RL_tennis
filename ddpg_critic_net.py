import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, num_agents):
        super(CriticNet, self).__init__()

        self.critic_fc1 = nn.Linear(state_size*num_agents, 512)
        self.critic_fc2 = nn.Linear(512 + action_size, 256)
        self.critic_fc3 = nn.Linear(256, 1)

    def forward(self, state, actions):
        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(state))
        state_and_actions = torch.cat((x_c, actions), 1)
        x_c = F.relu(self.critic_fc2(state_and_actions))
        return self.critic_fc3(x_c)
