import torch
from torch import nn
import torch.nn.functional as F


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, num_agents):
        super(CriticNet, self).__init__()

        self.critic_fc1 = nn.Linear(state_size*num_agents + action_size*num_agents, 256)
        self.critic_fc2 = nn.Linear(256, 128)
        self.critic_fc3 = nn.Linear(128, 1)

    # TODO maybe reset parameters uniformly

    def forward(self, state, actions):

        # get critics opinion on the state
        state_and_actions = torch.cat((state, actions), 1)
        x_c = F.relu(self.critic_fc1(state_and_actions))
        x_c = F.relu(self.critic_fc2(x_c))
        return self.critic_fc3(x_c)