# REINFORCEMENT LEARNING FOR UNITY TENNIS PROJECT

## Project description

![Earlies solution to the environment](DDPG_TENNIS/tennis_big.gif)

Two agent play a cooperative match of tennis. That is, in a 2D environment they play the ball over a net and
receive points if they succeed.

The environment is written in UNITY which can be observed through a continuous 48 dimensional statespace.
The statespace contains information about the position of the agents and the position and velocity of the ball
in the environment. Thus the world is completely observed by both agents.

The agent can perform 2 continuous actions, that are moving towards and away from the net and jumping.

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or
hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

## Project goal
The environment is considered solved if the agent get an **average reward of >=0.5** in 100 consecutive runs.

## Installation

1. Create virtual python environment and source it:
    - `python3.6 -m venv p3_env`
    - `source p3_env/bin/activate`
    - `pip install -U pip`
2. Clone repository:
    - `git clone https://github.com/sk-stm/RL_robot_arm_control.git`
    - `cd RL_robot_arm_control`
    - `pip install -r requirements.txt`
3. Follow the instructions to perform a minimal install of the environment for you system in https://github.com/openai/gym#installation
4. Select the environment according to your system:
    - Set the `ENV_PATH` variable in `main.py` to the path to the file that holds the appropriate environment for your system.
    You can download your specific environment:
        - Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Max OS: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows 32Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows 64Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Run training:
1. Set the `TRAIN_MODE` variable in `main.py` to `True`
2. Run main.py and the agent start training in the environment.

## Run inference:
1. Set the `ENV_PATH` variable to the environment you just downloaded.
2. Set the `TRAIN_MODE` variable in `main.py` to `False`.
3. Run `python main.py` and the agent start observing in the environment.
