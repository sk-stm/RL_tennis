import os
import random
import shutil
from collections import deque

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

from actor_net import ActorNet
from multi_agent_critic import CriticNet
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer
from PARAMETERS import LR_ACTOR, LR_CRITIC, device, WEIGHT_DECAY, BUFFER_SIZE, BATCH_SIZE, NOISE_REDUCTION_FACTOR, \
    UPDATE_EVERY, GAMMA, TAU, NOISE_INITIAL_WEIGHT
from save_and_plot import create_folder_structure_according_to_agent, save_score_plot

class DDPGAgent:

    def __init__(self, state_size, action_size):
        """
        Initializes the agent with local and target networks for the actor and the critic.

        :param state_size:  dimensionality of the state space
        :param action_size: dimensionality of the action space
        """
        # define actor networks
        self.local_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=LR_ACTOR)
        self.hard_update(self.target_actor_network, self.local_actor_network)

        # define actor networks
        self.local_actor2_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor2_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor2_optimizer = optim.Adam(self.local_actor2_network.parameters(), lr=LR_ACTOR)
        self.hard_update(self.target_actor2_network, self.local_actor2_network)

        # define critic networks
        self.local_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.target_critic_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.hard_update(self.target_critic_network, self.local_critic_network)

        # define critic networks
        self.local_critic2_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.target_critic2_network = CriticNet(state_size=state_size, action_size=action_size, num_agents=2).to(device)
        self.critic2_optimizer = optim.Adam(self.local_critic2_network.parameters(), lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)
        self.hard_update(self.target_critic2_network, self.local_critic2_network)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random.seed(0))

        self.oup = OrnsteinUhlenbeckProcess(action_size=action_size)
        self.t_step = 0
        self.noise_weight = NOISE_INITIAL_WEIGHT

    def act(self, state, add_noise: bool):
        """
        Retrieves an action from the local actor network given the current state.

        :param state:       state to get an action for
        :param add_noise:   if True, will add noise to an action given by the Ornstein-Uhlenbeck-Process
        :return:            chosen action
        """
        state_tensor = torch.from_numpy(state[0]).float().to(device)
        state_tensor2 = torch.from_numpy(state[1]).float().to(device)


        # important to NOT create a gradient here because it's done later during learning and doing it twice corrupts
        # the gradients
        self.local_actor_network.eval()
        self.local_actor2_network.eval()
        with torch.no_grad():

            # get the action
            action = self.local_actor_network(state_tensor)
            action2 = self.local_actor2_network(state_tensor2)

            action = action.cpu().detach().numpy()
            action2 = action2.cpu().detach().numpy()

            if add_noise:
                self.noise_weight *= NOISE_REDUCTION_FACTOR
                action += self.oup.sample()*self.noise_weight
                action2 += self.oup.sample() * self.noise_weight

                # clip the action after noise adding to the boundaries for the environment
                # TODO make this a parameter of the environment that is chosen.
                action = np.clip(action, -1, 1)
                action2 = np.clip(action2, -1, 1)

        # change the network back to training mode to train it during the learning step
        self.local_actor_network.train()
        self.local_actor2_network.train()

        return np.concatenate((action, action2))

    def step(self, states, actions, next_observed_state, observed_reward, done):
        """
        Adds the current state, the taken action, next state, reward and done to the replay memory. Performs
        learning with the actor and critic networks.

        :param state:               Currently perceived state of the environment
        :param actions:              Action performed in the environment
        :param next_observed_state: Next state observed in the environment
        :param observed_reward:     Observed reward after taken the action
        :param done:                Indicator if the episode is done or not
        """

        # TODO this done thingy only works because both agents spawn and die together in the tennis env
        done_value = int(done[0] == True)
        self.memory.add(states, actions, observed_reward, next_observed_state, done_value)

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
        state, actions, rewards, next_state, dones = experiences

        # run target actor to get next best action for the next_state
        next_target_action = self.target_actor_network(next_state[:BATCH_SIZE])
        next_target_action2 = self.target_actor2_network(next_state[BATCH_SIZE:])

        combined_next_target_actions = torch.cat((next_target_action, next_target_action2), 1).detach()
        combined_next_state = torch.cat((next_state[:BATCH_SIZE], next_state[BATCH_SIZE:]), 1).detach()
        combined_state = torch.cat((state[:BATCH_SIZE], state[BATCH_SIZE:]), 1).detach()

        # the data compartition is not too straight forward?
        combined_actions = actions.detach()

        ############## CRITIC 1 #########################
        # evaluate the chosen next_action by the critic
        with torch.no_grad():
            Q_targets_next = self.target_critic_network(combined_next_state, combined_next_target_actions)
        Q_targets = rewards[:, 0] + GAMMA * (1-dones.squeeze()) * Q_targets_next.squeeze()

        # obtain the local critics evaluation of the state
        Q_expected = self.local_critic_network(combined_state, combined_actions).squeeze()

        # formulate a loss to drive the local critics estimation more towards the target critics evaluation including
        # the received reward
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        # train the critic
        self.local_critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        ################## CRITIC 2 #############################
        # evaluate the chosen next_action by the critic
        with torch.no_grad():
            Q_targets_next2 = self.target_critic2_network(combined_next_state, combined_next_target_actions)
        Q_targets2 = rewards[:, 1] + GAMMA * (1 - dones.squeeze()) * Q_targets_next2.squeeze()

        # obtain the local critics evaluation of the state
        Q_expected2 = self.local_critic2_network(combined_next_state, combined_actions).squeeze()

        # formulate a loss to drive the local critics estimation more towards the target critics evaluation including
        # the received reward
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets2.detach())

        # train the critic
        self.local_critic2_network.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        ################# ACTORS #####################
        # get forward pass for local actor on the current state to create an action.
        action = self.local_actor_network(state[:BATCH_SIZE])
        action2 = self.local_actor2_network(state[BATCH_SIZE:])
        # evaluate that action with the local critic to formulate a loss in the action according to its evaluation
        # important is the '-' in front because pytorch performs gradient decent but we want to maximize the value
        # of the critic

        combined_actions = torch.cat((action, action2.detach()), 1)
        combined_actions2 = torch.cat((action.detach(), action2), 1)

        policy_loss = -self.local_critic_network(combined_state, combined_actions).mean()
        policy_loss2 = -self.local_critic2_network(combined_state, combined_actions2).mean()

        # learn the actor network
        self.local_actor_network.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.local_actor2_network.zero_grad()
        policy_loss2.backward()
        self.actor2_optimizer.step()

        # perform soft updates from the local networks to the targets to converge towards better evaluation values.
        self.soft_update(self.local_actor_network, self.target_actor_network, TAU)
        self.soft_update(self.local_actor2_network, self.target_actor2_network, TAU)
        self.soft_update(self.local_critic_network, self.target_critic_network, TAU)
        self.soft_update(self.local_critic2_network, self.target_critic2_network, TAU)

    def soft_update(self, source_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param source_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, source_model, target_model):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        return self.soft_update(source_model, target_model, tau=1.)

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
                   os.path.join(new_folder_path, f'checkpoint_{np.round(score_max, 2)}.pth'))
        torch.save(self.local_actor2_network.state_dict(),
                   os.path.join(new_folder_path, f'checkpoint_agent2_{np.round(score_max, 2)}.pth'))
        save_score_plot(scores, score_mean_list, i_episode, path=new_folder_path)
        shutil.copyfile("PARAMETERS.py", os.path.join(new_folder_path, "PARAMETERS.py"))