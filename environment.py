import torch

ENV_PATH = '../Tennis_Linux/Tennis.x86'
MODEL1_TO_LOAD = ''
MODEL2_TO_LOAD = ''
AGENT_TYPE = 'MADDPG'
ENV_NAME = 'TENNIS'
NEEDED_REWARD_FOR_SOLVING_ENV = 0.5


# DEVICE PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_EACH_NEXT_BEST_REWARD = 10
