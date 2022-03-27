import gym
from gym import spaces
import numpy

from SigmaPiFrameworkPython.BooleanFunctionGenerator import boolean_function_generator
from SigmaPiFrameworkPython.MonomialSetup import monomial_setup, q_matrix_generator

from enum import Enum


class FunctionMode(Enum):
    SINGLE = 1
    LIST = 2
    RANDOM = 3


class MinTermSrpobfEnvBase(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension,
                 q_matrix_representation, episodic_reward):
        super(MinTermSrpobfEnvBase, self).__init__()
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.total_number_of_functions = 2 ** self.two_to_power_dimension

        if isinstance(function, int):
            if function < self.total_number_of_functions:
                self.function_mode = FunctionMode.SINGLE
                self.function = function % self.total_number_of_functions
            else:
                self.function_mode = FunctionMode.RANDOM
                self.function = numpy.random.randint(self.total_number_of_functions)
        elif isinstance(function, list):
            self.function_mode = FunctionMode.LIST
            self.function_list = (numpy.array(function) % self.total_number_of_functions).tolist()
            self.function_list_len = len(self.function_list)
            self.function = self.function_list[numpy.random.randint(self.function_list_len)]
        else:
            raise Exception("Unsupported function type")

        self.function_vector = boolean_function_generator(self.function, self.dimension)
        self.d_matrix = monomial_setup(dimension)
        self.q_matrix = q_matrix_generator(self.function, self.dimension, self.d_matrix)
        self.walsh_spectrum = self.q_matrix.sum(1)
        self.q_matrix_representation = q_matrix_representation

        if self.q_matrix_representation:
            self.function_representation_size = self.two_to_power_dimension ** 2
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        else:
            self.function_representation_size = self.two_to_power_dimension
            self.function_representation = self.walsh_spectrum

        self.key_name = ""
        self.key_size = 0
        self.key = None

        self.current_step = 0

        self.max_reward = -1
        self.max_reward_in_the_episode = -1
        self.max_rewards_in_the_episodes = []
        self.max_reward_dict = {}

        self.max_reward_key = None
        self.max_reward_key_dict = {}

        self.cumulative_reward_in_the_episode = 0
        self.cumulative_rewards_in_the_episodes = []

        self.function_each_episode = []

        self.episodic_reward = episodic_reward

    def _create_action_and_observation_space(self):
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, [self.state_size])

    def update_episode_reward_statistics(self, reward):
        self.cumulative_reward_in_the_episode += reward

        if reward > self.max_reward_in_the_episode:
            self.max_reward_in_the_episode = reward

            if self.function_mode is FunctionMode.SINGLE:
                if reward > self.max_reward:
                    self.max_reward = reward
                    self.max_reward_key = self.key.copy()
            else:
                if self.function in self.max_reward_dict.keys():
                    if reward > self.max_reward_dict[self.function]:
                        self.max_reward_dict[self.function] = reward
                        self.max_reward_key_dict[self.function] = self.key.tolist().copy()
                else:
                    self.max_reward_dict[self.function] = reward
                    self.max_reward_key_dict[self.function] = self.key.tolist().copy()

    def close(self):
        pass

    def set_function(self, function):
        self.function = function % self.total_number_of_functions
        self.function_vector = boolean_function_generator(self.function, self.dimension)
        self.q_matrix = q_matrix_generator(self.function, self.dimension, self.d_matrix)
        self.walsh_spectrum = self.q_matrix.sum(1)

        if self.q_matrix_representation:
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        else:
            self.function_representation = self.walsh_spectrum

    def create_observation(self):
        observation = numpy.ones([self.state_size])
        observation[0:self.function_representation_size] = self.function_representation
        observation[self.function_representation_size:self.state_size] = self.key
        return observation

    def reset_key(self):
        pass

    def reset(self):
        self.current_step = 0

        self.max_rewards_in_the_episodes.append(self.max_reward_in_the_episode)
        self.max_reward_in_the_episode = -1

        if not self.episodic_reward:
            self.cumulative_rewards_in_the_episodes.append(self.cumulative_reward_in_the_episode)
            self.cumulative_reward_in_the_episode = 0

        if self.function_mode is not FunctionMode.SINGLE:
            self.function_each_episode.append(self.function)

            if self.function_mode is FunctionMode.RANDOM:
                self.function = numpy.random.randint(self.total_number_of_functions)
            elif self.function_mode is FunctionMode.LIST:
                self.function = self.function_list[numpy.random.randint(self.function_list_len)]

            self.set_function(self.function)

        self.reset_key()

        observation = self.create_observation()
        return observation

    def check_episode_end(self):
        return self.current_step > self.steps_in_each_epoch

    def switch_to_single_mode(self):
        self.function_mode = FunctionMode.SINGLE
