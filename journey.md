# Start of project

Try out [IQL](https://arxiv.org/pdf/1908.03963.pdf) (independedn Q-learning) since it's naive,
straight forward and works for small problems.

However, in the case of function approximation, especiallydeep neural network (DNN) it fails.
One of the main reasons for this failure is the need for the replay memory to stabilize the training
with DNNs, which does not work because the environment is non stationary for an agent because the other
agent is an undeterministic part of the environment.

