import os
import random
import shutil

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

from ddpg_actor_net import ActorNet
from ddpg_critic_net import CriticNet
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer
from PARAMETERS import LR_ACTOR, LR_CRITIC, device, WEIGHT_DECAY, BUFFER_SIZE, BATCH_SIZE, NOISE_REDUCTION_FACTOR, \
    UPDATE_EVERY, GAMMA, TAU, INITIAL_NOISE_WEIGHT
from save_and_plot import create_folder_structure_according_to_agent, save_score_plot


class DPGAgent:

    def __init__(self, state_size, action_size, agent_index):
        """
        Initializes the agent with local and target networks for the actor and the critic.

        :param state_size:  dimensionality of the state space
        :param action_size: dimensionality of the action space
        """
        self.local_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=LR_ACTOR)

        self.local_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.target_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size*2, BUFFER_SIZE, BATCH_SIZE)

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
        concat_state = np.concatenate(state, 0)
        state_tensor = torch.from_numpy(concat_state).float().to(device)

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

    def step(self, state, action, next_observed_state, observed_reward, done):
        """
        Adds the current state, the taken action, next state, reward and done to the replay memory. Performs
        learning with the actor and critic networks.

        :param state:               Currently perceived state of the environment
        :param actions:              Action performed in the environment
        :param next_observed_state: Next state observed in the environment
        :param observed_reward:     Observed reward after taken the action
        :param done:                Indicator if the episode is done or not
        """

        # TODO test if this holds the correct value
        done_value = int(done == True)
        concat_states_1d = np.concatenate((state[0], state[1]))
        next_states_1d = np.concatenate((next_observed_state[0], next_observed_state[1]))

        self.memory.add(concat_states_1d, action, observed_reward, next_states_1d, done_value)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """
        Perform learning of the agent.

        :param experiences: Sample of size: batch size from the replay buffer.
        """
        state, action, reward, next_state, dones = experiences
        next_target_action = self.target_actor_network(next_state).detach()

        self._learn_critic(action, dones, next_state, next_target_action, reward, state)
        self._learn_actor(state)

        # perform soft updates from the local networks to the targets to converge towards better evaluation values.
        self._soft_update(self.local_actor_network, self.target_actor_network, TAU)
        self._soft_update(self.local_critic_network, self.target_critic_network, TAU)

    def _learn_actor(self, state):
        """
        Calculates the loss and performs gradient decent on the actor network.

        :param state: states to train on (sampled from the experience buffer)
        """
        # get forward pass for local actor on the current state to create an action.
        action = self.local_actor_network(state)
        # evaluate that action with the local critic to formulate a loss in the action according to its evaluation
        # important is the '-' in front because pytorch performs gradient decent but we want to maximize the value
        # of the critic
        policy_loss = -self.local_critic_network(state, action).mean()
        # learn the actor network
        self.local_actor_network.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def _learn_critic(self, action, dones, next_state, next_target_action, reward, state):
        """
        Calculates the loss and performs gradient decent on the critic network. Each parameter is taken from the
        experience buffer and therefore is a list of values for each entry in the experience buffer. The length
        of the vector is determined by the batch size.

        :param action:              actions taken by the actor according to the sampled states from the experience buffer
        :param dones:               Vector indicating if the agent terminated or not
        :param next_state:          Next state of the agent after taking "action" in the current state
        :param next_target_action:  best action according to the target network
        :param reward:              received rewards after taking the action
        :param state:               current state of the environment
        """
        # evaluate the chosen next_action by the critic
        next_state_value = self.target_critic_network(next_state, next_target_action).detach()
        target_critic_value_for_next_state = reward + GAMMA * (1 - dones.squeeze()) * next_state_value.squeeze()
        # obtain the local critics evaluation of the state
        local_value_current_state = self.local_critic_network(state, action).squeeze()
        # formulate a loss to drive the local critics estimation more towards the target critics evaluation including
        # the received reward
        critic_loss = F.mse_loss(local_value_current_state, target_critic_value_for_next_state)
        # train the critic
        self.local_critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _soft_update(self, local_model, target_model, tau):
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