import numpy as np
from PARAMETERS import THETA, MU, SIGMA


class OrnsteinUhlenbeckProcess:
    """
    Defined a stochastic value following the Ornstein-Uhlenbeck process:
    https://encyclopediaofmath.org/wiki/Ornstein-Uhlenbeck_process
    """

    def __init__(self, action_size):
        self.prev_x = np.zeros(action_size)
        self.action_size = action_size
        self.reset_process()

    def sample(self):
        """
        Samples next value in the process.

        :return: next value of the process
        """
        x = self.prev_x + THETA * (MU - self.prev_x) + SIGMA * np.random.randn(*(self.action_size,))
        self.prev_x = x
        return x

    def reset_process(self):
        """
        Resets the process.
        """
        self.prev_x = np.zeros(self.action_size)
