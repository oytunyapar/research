import gym
from gym import spaces
import numpy

from SigmaPiFrameworkPython.BooleanFunctionGenerator import boolean_function_generator
from SigmaPiFrameworkPython.MonomialSetup import monomial_setup, q_matrix_generator

from OpenAiGym.Utils.DumpOutputs import dump_outputs, dump_json

from enum import Enum


class FunctionMode(Enum):
    SINGLE = 1
    LIST = 2
    RANDOM = 3


class FunctionRepresentationType(Enum):
    Q_MATRIX = 1
    WALSH_SPECTRUM = 2
    SPECTRUM = 3


class MinTermSrpobfEnvBase(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension,
                 function_representation_type, episodic_reward):
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
        self.spectrum = self.walsh_spectrum * (2 ** -self.dimension)
        self.function_representation_type = function_representation_type
        self.function_representation = None

        if self.function_representation_type == FunctionRepresentationType.Q_MATRIX:
            self.function_representation_size = self.two_to_power_dimension ** 2
        else:
            self.function_representation_size = self.two_to_power_dimension

        self.update_represented_function()

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

    def _create_observation_space(self, limit_low=-numpy.inf, limit_high=numpy.inf, dtype=numpy.float32):
        self.observation_space = spaces.Box(limit_low, limit_high, [self.state_size], dtype)

    def _create_action_space(self, callback=None):
        if callback is None:
            self.action_space = spaces.Discrete(self.action_size)
        else:
            self.action_space = spaces.Discrete(self.action_size, callback)

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

    def update_represented_function(self):
        if self.function_representation_type == FunctionRepresentationType.Q_MATRIX:
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        elif self.function_representation_type == FunctionRepresentationType.WALSH_SPECTRUM:
            self.function_representation = self.walsh_spectrum
        elif self.function_representation_type == FunctionRepresentationType.SPECTRUM:
            self.function_representation = self.spectrum
        else:
            raise Exception("Undefined function representation type.")

    def set_function(self, function):
        self.function = function % self.total_number_of_functions
        self.function_vector = boolean_function_generator(self.function, self.dimension)
        self.q_matrix = q_matrix_generator(self.function, self.dimension, self.d_matrix)
        self.walsh_spectrum = self.q_matrix.sum(1)
        self.spectrum = self.walsh_spectrum * (2 ** -self.dimension)

        self.update_represented_function()

    def create_observation(self):
        observation = numpy.ones([self.state_size])
        observation[0:self.function_representation_size] = self.function_representation
        observation[self.function_representation_size:self.state_size] = self.key
        return observation

    def reset_internal(self):
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

        self.reset_internal()

        observation = self.create_observation()
        return observation

    def check_episode_end(self):
        return self.current_step > self.steps_in_each_epoch

    def switch_to_single_mode(self):
        self.function_mode = FunctionMode.SINGLE

    def env_specific_configuration(self):
        return {"dimension": self.dimension, "function_representation_type": str(self.function_representation_type),
                "function_mode": str(self.function_mode), "key_name": self.key_name,
                "episodic_reward": str(self.episodic_reward)}

    def dump_env_statistics(self, output_directory):
        dump_outputs(self.max_rewards_in_the_episodes, output_directory, "max_rewards_in_the_episodes")
        dump_outputs(self.cumulative_rewards_in_the_episodes, output_directory, "cumulative_rewards_in_the_episodes")
        dump_json(self.function_each_episode, output_directory, "function_each_episode")
        dump_json(self.max_reward_dict, output_directory, "max_reward_dict")
        dump_json(self.max_reward_key_dict, output_directory, "max_reward_" + self.key_name + "_dict")

    def get_key_from_state(self, state):
        return state[self.function_representation_size:self.state_size]
