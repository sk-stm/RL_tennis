import torch
from torch import nn
import torch.nn.functional as F


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, num_agents, fcs1_units=128, fcs2_units=64):
        super(CriticNet, self).__init__()

        self.critic_fc1 = nn.Linear(state_size*num_agents, fcs1_units)
        self.critic_fc2 = nn.Linear(fcs1_units+action_size*num_agents, fcs2_units)
        self.critic_fc3 = nn.Linear(64, 1)

    # TODO maybe reset parameters uniformly

    def forward(self, state, actions):

        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(state))
        x_c = torch.cat((x_c, actions), dim=1)
        x_c = F.relu(self.critic_fc2(x_c))
        return self.critic_fc3(x_c)