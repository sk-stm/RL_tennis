from collections import deque

from unityagents import UnityEnvironment
import numpy as np
import torch
from PARAMETERS import device, SAVE_DISTANCE, NUM_EPISODES
from ddpg_actor_net import ActorNet

from ddpg_agent import DPGAgent
from environment import ENV_PATH
from save_and_plot import save_score_plot

env = UnityEnvironment(file_name=ENV_PATH)
TRAIN_MODE = False
MODEL1_TO_LOAD = 'DDPG_TENNIS/best_solve_agent_1/checkpoint_0_0.57.pth'
MODEL2_TO_LOAD = 'DDPG_TENNIS/best_solve_agent_2/checkpoint_1_0.57.pth'


def main():
    brain_name, num_agents, agent_states, state_size, action_size = init_env()

    agent_kournikova = DPGAgent(state_size=state_size, action_size=action_size, agent_index=0)
    agent_agassi = DPGAgent(state_size=state_size, action_size=action_size, agent_index=1)

    agents = [agent_kournikova, agent_agassi]

    if not TRAIN_MODE:
        load_model_into_agent(agent_kournikova, state_size, action_size, MODEL1_TO_LOAD)
        load_model_into_agent(agent_agassi, state_size, action_size, MODEL2_TO_LOAD)

    run_environment(brain_name, agents)

    env.close()


def load_model_into_agent(agent, state_size, action_size, model_path):
    """
    Loads a pretrained network into the created agent.
    """
    actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
    actor_network.load_state_dict(torch.load(model_path))
    agent.local_actor_network = actor_network


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


def run_environment(brain_name, agents):
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

    for i_episode in range(1, NUM_EPISODES + 1):
        score = 0
        score1 = 0
        score2 = 0

        # get first state of environment
        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]
        state = env_info.vector_observations

        for agent in agents:
            agent.oup.reset_process()

        # TODO make this a variable depending on the environment (episode length)
        for i_times in range(1000):

            action1, \
            action2, \
            done, \
            next_observed_state, \
            observed_reward = act_in_environment(agents, brain_name, state)

            if TRAIN_MODE:
                agents[0].step(state, action1, next_observed_state, observed_reward[0], done[0])
                agents[1].step(state, action2, next_observed_state, observed_reward[1], done[1])

            state = next_observed_state

            score1 += observed_reward[0]
            score2 += observed_reward[1]
            score = max((score1, score2))
            if any(done):
                break

        score_mean = save_optained_scores_to_list(score, score_mean_list, scores, scores_window)

        plot_and_save_agent(agents, i_episode, score_max, scores, score_mean, score_mean_list, score)

        if score_mean > score_max + SAVE_DISTANCE:
            score_max = score_mean


def save_optained_scores_to_list(score, score_mean_list, scores, scores_window):
    scores_window.append(score)
    scores.append(score)
    score_mean = np.mean(scores_window)
    score_mean_list.append(score_mean)
    return score_mean


def act_in_environment(agents, brain_name, state):
    """
    Let agent act in the environment

    :param agents:      all agents to act
    :param brain_name:  brain name for unity environment
    :param state:       state of the environment
    :return: actions for each agent, vector if agents finished, next_state for each agent, reward for each agent
    """
    action1 = agents[0].act(state, add_noise=TRAIN_MODE)
    action2 = agents[1].act(state, add_noise=TRAIN_MODE)
    concat_actions = np.array([action1, action2])
    concat_actions = np.concatenate(concat_actions, 0)

    env_info = env.step(concat_actions)[brain_name]
    next_observed_state = env_info.vector_observations
    observed_reward = env_info.rewards
    done = env_info.local_done

    return action1, action2, done, next_observed_state, observed_reward


def plot_and_save_agent(agents, i_episode, score_max, scores, scores_mean, score_mean_list, score):
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
        print('\rEpisode {}\tAverage Score: {:.4f}\t Noise weight: {:.7f}'.format(i_episode, scores_mean, agents[0].noise_weight))
    else:
        print('\rEpisode {}\tScore for this episode {:.4f}\t Applied noise: {:.7f}:'.format(i_episode, score, agents[0].applied_noise), end="")
    if i_episode % 400 == 0 and TRAIN_MODE:
        save_score_plot(scores, score_mean_list, i_episode)
    if scores_mean >= 0.0 and scores_mean >= score_max + SAVE_DISTANCE and TRAIN_MODE:
        for agent in agents:
            agent.save_current_agent(score_max=score_max, scores=scores, score_mean_list=score_mean_list, i_episode=i_episode)
            # TODO save replay buffer parameters as well if prioritized replay buffer was used
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f} '.format(i_episode - 100, scores_mean))


if __name__ == "__main__":
    main()
