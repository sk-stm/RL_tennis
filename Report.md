# Learning algorithm
## Problem statement:
Unlike single agent environments, this environment contains 2 agents which shall cooperate towards a
common goal. The incentive is to play tennis and keep the ball up in the air as long as possible.
Since there are multiple agents in the same environment, the environment is more stochastic w.r.t. one
agent because it observes the other agent as well. And the other agent changes behavior over time, thus
the estimation of the next state is very noisy. Learning to deal with this noise is very important for
a stable learning.
This partucular environment contains 2 agents in the left and right half of a 2D tennis cord. A ball
is dropped in one of the fields. Each agent controls a tennis racket to hit the ball of. The task of
the agents is to play the ball back and forth between them and not letting it fall down to the floor.
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the
 ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is
 to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and
racket. Each agent receives its own, local observation. Two continuous actions are available,
corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of
+0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a
score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2
scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## MADDPG
MADDGP (multi agent deep deterministic policy gradient) extends single agent DDPG for the multi agent
purpose. It works as follows:
Each agent has it's own actor and critic and acts according to the action that the actor creates. The
agents actor only perceives it's own local state of the environment and acts accordingly. On the contrairy
 each critic perceives all states and all the actions of each agent. It uses this information to evaluate
 the state of each agent with respect to all other agents. The experience that all agents create is stored
 in a shared experience buffer during training. That way each drawn sample from the replay buffer will
 contain a full example of the actions and local states of each agent. Each critic will use this
 information to optimize the value function for it's own agent.
Besides this the algorithm works just like single agent DDPG which is described in
[my other repository](https://github.com/sk-stm/RL_robot_arm_control/blob/main/Report.md).

# TODO network parameters:
## ACOTR
## CRITIC



# Best performace parameters for DDPG:
### LEARNING PARAMETERS


### NOISE PARAMETERS
