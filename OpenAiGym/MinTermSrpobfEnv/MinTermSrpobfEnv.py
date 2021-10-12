import gym
from gym import spaces

import numpy

from SigmaPiFrameworkPython.boolean_function_generator import boolean_function_generator
from SigmaPiFrameworkPython.monsetup import monsetup, q_matrix_generator

class MinTermSrpobfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension, q_matrix_representation):
        super(MinTermSrpobfEnv, self).__init__()
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.function = function

        self.function_vector = boolean_function_generator(function, dimension)
        self.d_matrix = monsetup(dimension)
        self.q_matrix = q_matrix_generator(function, self.dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)
        self.k_vector_size = self.two_to_power_dimension
        self.k_vector = numpy.ones(self.two_to_power_dimension)
        self.q_matrix_representation = q_matrix_representation

        if q_matrix_representation:
            self.function_representation_size = self.two_to_power_dimension ** 2
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        else:
            self.function_representation_size = self.two_to_power_dimension
            self.function_representation = self.walsh_spectrum

        self.state_size = self.function_representation_size + self.k_vector_size
        self.action_size = self.two_to_power_dimension + 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(self.action_size)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
        return observation, reward, done, info

    def reset(self):
        return observation  # reward, done, info can't be included

    def close(self):
        pass
