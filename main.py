import datetime
import os
from collections import deque
from typing import List
import shutil

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch
from PARAMETERS import device
from actor_net import ActorNet

from dual_maddpg_agent import DDPGAgent
from environment import ENV_PATH
from save_and_plot import save_score_plot

env = UnityEnvironment(file_name=ENV_PATH)
TRAIN_MODE = False
MODEL1_TO_LOAD = 'DDPG_TENNIS/2021_04_21_20_24_54/checkpoint_0.01.pth'
MODEL2_TO_LOAD = 'DDPG_TENNIS/2021_04_21_20_24_54/checkpoint_agent2_0.01.pth'


def main():
    brain_name, num_agents, agent_states, state_size, action_size = init_env()

    # agent = A2CAgent(env=env, brain_name=brain_name, state_size=state_size, action_size=action_size)
    agent = DDPGAgent(state_size=state_size, action_size=action_size)

    if not TRAIN_MODE:
        load_model_into_agent(agent, state_size, action_size)

    run_environment(brain_name, agent)

    env.close()


def load_model_into_agent(agent, state_size, action_size):
    """
    Loads a pretrained network into the created agent.
    """
    actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
    actor_network.load_state_dict(torch.load(MODEL1_TO_LOAD))
    agent.local_actor_network = actor_network

    actor2_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
    actor2_network.load_state_dict(torch.load(MODEL2_TO_LOAD))
    agent.local_actor2_network = actor_network


def init_env():
    """
    Initialized the environment.

    :return: brain_name, num_agents, agent_states, state_size, action_size
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    agent_states = env_info.vector_observations
    state_size = agent_states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(agent_states.shape[0], state_size))
    print('The state for the first agent looks like:', agent_states[0])

    agent_states = env_info.vector_observations                  # get the current state (for each agent)

    return brain_name, num_agents, agent_states, state_size, action_size


num_episodes = 100000


def run_environment(brain_name, agent):
    """
    Runs the environment and the agent.

    :param brain_name:  name of the brain of the environment
    :param agent:       the agent to act in this environment
    """
    # lists containing scores from each episode
    scores_window = deque(maxlen=100)
    score_max = 0
    scores = []
    score_mean_list = []
    saved_earliest_agent = False

    for i_episode in range(1, num_episodes + 1):
        episode_scores = np.zeros(2)
        # get first state of environment
        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

        state = env_info.vector_observations

        agent.oup.reset_process()

        # TODO make this a variable depending on the environment (episode length)
        for i_times in range(1000):
            # take action in environment and get its response
            actions = agent.act(state, add_noise=TRAIN_MODE)

            env_info = env.step(actions)[brain_name]
            next_observed_state = env_info.vector_observations
            observed_reward = env_info.rewards
            done = env_info.local_done

            if TRAIN_MODE:
                agent.step(state, actions, next_observed_state, observed_reward, done)

            state = next_observed_state

            episode_scores += np.array(observed_reward)
            if done[0]:
                break

        # save the obtained scores
        max_episode_score = np.max(episode_scores)
        scores_window.append(max_episode_score)
        scores.append(max_episode_score)
        score_mean = np.mean(scores_window)
        score_mean_list.append(score_mean)

        plot_and_save_agent(agent, i_episode, score_max, scores, score_mean, score_mean_list, max_episode_score)

        if score_mean > score_max:
            score_max = score_mean


def plot_and_save_agent(agent, i_episode, score_max, scores, scores_mean, score_mean_list, score):
    """
    Plots and saves the agent each 100th episode.
    Saves the agent, the current scores, the episode number, the trained parameters of the NN model and the hyper
    parameters of the agent to a folder with the current date and time if the mean average of the last 100 scores
    are > 13 and if a new maximum average was reached.

    :param agent:           agent to saved
    :param eps:             current value of epsilon
    :param i_episode:       number of current episode
    :param score_max:       max_score reached by the agent so far
    :param scores:          all scores of the agent reached so far
    :param scores_mean:     mean of the last 100 scores
    :param score:           score of the running episode
    """

    # evaluate every so often
    if i_episode % 100 == 0 and TRAIN_MODE:
        print('\rEpisode {}\tAverage Score: {:.4f}\t Noise weight: {:.7f}'.format(i_episode, scores_mean, agent.noise_weight))
    else:
        print('\rEpisode {}\tScore for this episode {:.4f}:'.format(i_episode, score), end="")
    if i_episode % 400 == 0 and TRAIN_MODE:
        save_score_plot(scores, score_mean_list, i_episode)
    if i_episode > 100 and scores_mean >= 0.0 and scores_mean >= score_max and TRAIN_MODE:
        agent.save_current_agent(score_max=score_max, scores=scores, score_mean_list=score_mean_list, i_episode=i_episode)
        # TODO save replay buffer parameters as well if prioritized replay buffer was used
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f} '.format(i_episode, scores_mean))


if __name__ == "__main__":
    main()
