import os
import random
import shutil

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

from maddpg_actor_net import ActorNet
from maddpg_critic_net import CriticNet
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer
from PARAMETERS import LR_ACTOR, LR_CRITIC, device, WEIGHT_DECAY, BUFFER_SIZE, BATCH_SIZE, NOISE_REDUCTION_FACTOR, \
    UPDATE_EVERY, GAMMA, TAU, INITIAL_NOISE_WEIGHT
from save_and_plot import create_folder_structure_according_to_agent, save_score_plot


class MADDPGAgent:

    def __init__(self, state_size, action_size, agent_index):
        """
        Initializes the agent with local and target networks for the actor and the critic.

        :param state_size:  dimensionality of the state space
        :param action_size: dimensionality of the action space
        """
        # define actor networks
        self.local_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=LR_ACTOR)

        # define critic networks
        self.local_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.target_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size*2, BUFFER_SIZE, BATCH_SIZE, random.seed(0))

        self.agent_index = agent_index
        self.state_size = state_size
        self.oup = OrnsteinUhlenbeckProcess(action_size=action_size)
        self.t_step = 0
        self.noise_weight = INITIAL_NOISE_WEIGHT
        self.applied_noise = 0

    def act(self, state, add_noise: bool):
        """
        Retrieves an action from the local actor network given the current state.

        :param state:       state to get an action for
        :param add_noise:   if True, will add noise to an action given by the Ornstein-Uhlenbeck-Process
        :return:            chosen action
        """
        state_tensor = torch.from_numpy(state[self.agent_index]).float().to(device)

        # important to NOT create a gradient here because it's done later during learning and doing it twice corrupts
        # the gradients
        self.local_actor_network.eval()
        with torch.no_grad():

            # get the action
            action = self.local_actor_network(state_tensor)

            action = action.cpu().detach().numpy()

            if add_noise:
                self.noise_weight *= NOISE_REDUCTION_FACTOR
                self.applied_noise = np.mean(self.oup.sample()*self.noise_weight)
                action += self.oup.sample()*self.noise_weight

                # clip the action after noise adding to the boundaries for the environment
                # TODO make this a parameter of the environment that is chosen.
                action = np.clip(action, -1, 1)

        # change the network back to training mode to train it during the learning step
        self.local_actor_network.train()

        return action

    def step(self, other_agent, state, actions, next_observed_state, observed_reward, done):
        """
        Adds the current state, the taken action, next state, reward and done to the replay memory. Performs
        learning with the actor and critic networks.

        :param state:               Currently perceived state of the environment
        :param actions:              Action performed in the environment
        :param next_observed_state: Next state observed in the environment
        :param observed_reward:     Observed reward after taken the action
        :param done:                Indicator if the episode is done or not
        """

        # TODO this done thingy only works beacause both agents spawn and die together in the tennis env
        done_value = int(done[0] == True)
        #TODO only works for 2 agents!
        concat_states_1d = np.concatenate((state[0], state[1]))
        next_states_1d = np.concatenate((next_observed_state[0], next_observed_state[1]))

        self.memory.add(concat_states_1d, actions, observed_reward, next_states_1d, done_value)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, other_agent)

    def learn(self, experiences, other_agent):
        """
        Perform learning of the agent.

        :param experiences: Sample of size: batch size from the replay buffer.
        """
        state, actions, rewards, next_state, dones = experiences

        # run target actor to get next best action for the next_state
        if self.agent_index == 0:
            agent_next_states = next_state[:, :self.state_size]
            other_agent_next_state = next_state[:, self.state_size:]
        else:
            agent_next_states = next_state[:, self.state_size:]
            other_agent_next_state = next_state[:, :self.state_size]

        next_target_action = self.target_actor_network(agent_next_states).detach()
        other_agent_next_target_action = other_agent.target_actor_network(other_agent_next_state).detach()

        combined_next_target_actions = torch.cat((next_target_action, other_agent_next_target_action), 1).detach()

        ############## CRITIC  #########################
        # evaluate the chosen next_action by the critic
        next_state_value = self.target_critic_network(next_state, combined_next_target_actions).detach()
        target_critic_value_for_next_state = rewards[:, self.agent_index] + GAMMA * (1-dones.squeeze()) * next_state_value.squeeze()

        # obtain the local critics evaluation of the state
        local_value_current_state = self.local_critic_network(state, actions).squeeze()

        # formulate a loss to drive the local critics estimation more towards the target critics evaluation including
        # the received reward
        critic_loss = F.mse_loss(local_value_current_state, target_critic_value_for_next_state)

        # train the critic
        self.local_critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ################# ACTOR #####################
        if self.agent_index == 0:
            agent_states = state[:, :self.state_size]
            other_agent_state = state[:, self.state_size:]
        else:
            agent_states = state[:, self.state_size:]
            other_agent_state = state[:, :self.state_size]

        # get forward pass for local actor on the current state to create an action.
        action = self.local_actor_network(agent_states)
        other_agent_action = other_agent.local_actor_network(other_agent_state).detach()
        # evaluate that action with the local critic to formulate a loss in the action according to its evaluation
        # important is the '-' in front because pytorch performs gradient decent but we want to maximize the value
        # of the critic

        # TODO check if I have to detach the combined_state from the graph to prevent gradient flow to the critic
        combined_actions = torch.cat((action, other_agent_action), 1)

        policy_loss = -self.local_critic_network(state, combined_actions).mean()

        # learn the actor network
        self.local_actor_network.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # perform soft updates from the local networks to the targets to converge towards better evaluation values.
        self.soft_update(self.local_actor_network, self.target_actor_network, TAU)
        self.soft_update(self.local_critic_network, self.target_critic_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_model_into_DDPG_agent(self, model_path):
        """
        Loads a pretrained network into the created agent.
        """
        self.local_actor_network.load_state_dict(torch.load(model_path))

    def save_current_agent(self, score_max, scores, score_mean_list, i_episode):
        """
        Saves the current agent.

        :param agent:       agent to saved
        :param score_max:   max_score reached by the agent so far
        :param scores:      all scores of the agent reached so far
        :param i_episode:   number of current episode
        """
        new_folder_path = create_folder_structure_according_to_agent()

        os.makedirs(new_folder_path, exist_ok=True)
        torch.save(self.local_actor_network.state_dict(),
                   os.path.join(new_folder_path, f'checkpoint_{self.agent_index}_{np.round(score_max, 2)}.pth'))
        save_score_plot(scores, score_mean_list, i_episode, path=new_folder_path)
        shutil.copyfile("PARAMETERS.py", os.path.join(new_folder_path, "PARAMETERS.py"))