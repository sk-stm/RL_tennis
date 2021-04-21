import numpy as np
from PARAMETERS import NOISE_THETA, NOISE_MU, NOISE_SIGMA


class OrnsteinUhlenbeckProcess:

    def __init__(self, action_size):
        self.action_size = action_size
        self.reset_process()

    def sample(self):
        x = self.state
        dx = NOISE_THETA * (NOISE_MU - self.state) + NOISE_SIGMA * np.random.randn(len(self.state))
        self.state = x + dx
        return np.copy(self.state)

    def reset_process(self):
        self.state = np.ones(self.action_size) * NOISE_MU