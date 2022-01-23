import gym
from gym import spaces

import functools
import numpy

from SigmaPiFrameworkPython.boolean_function_generator import boolean_function_generator
from SigmaPiFrameworkPython.monsetup import monsetup, q_matrix_generator

from enum import Enum


class ActionType(Enum):
    INCREASE = 1
    INCREASE_DECREASE = 2


class FunctionMode(Enum):
    SINGLE = 1
    LIST = 2
    RANDOM = 3


def reward_to_number_of_zeros(reward):
    return int(numpy.sqrt(reward))


class MinTermSrpobfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension,
                 q_matrix_representation,
                 action_type,
                 no_action_episode_end,
                 episodic_reward):
        super(MinTermSrpobfEnv, self).__init__()
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
        self.d_matrix = monsetup(dimension)
        self.q_matrix = q_matrix_generator(self.function, self.dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)
        self.k_vector_size = self.two_to_power_dimension
        self.k_vector = numpy.ones(self.two_to_power_dimension)
        self.k_vector_element_max_value = 2 ** (dimension - 1)
        self.q_matrix_representation = q_matrix_representation

        if self.q_matrix_representation:
            self.function_representation_size = self.two_to_power_dimension ** 2
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        else:
            self.function_representation_size = self.two_to_power_dimension
            self.function_representation = self.walsh_spectrum

        self.state_size = self.function_representation_size + self.k_vector_size

        # Example for using image as input:
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, [self.state_size])

        self.current_step = 0
        self.steps_in_each_epoch = ((self.two_to_power_dimension / 2) - 1) * self.two_to_power_dimension

        if action_type == ActionType.INCREASE_DECREASE:
            self.steps_in_each_epoch = 3 * self.steps_in_each_epoch
            self.action_size = 2 * self.two_to_power_dimension + 1
            self.increase_last_index = self.two_to_power_dimension - 1
            self.action_function = self.act_increase_decrease
        elif action_type == ActionType.INCREASE:
            self.action_size = self.two_to_power_dimension + 1
            self.action_function = self.act_increase
        else:
            raise Exception("Invalid action type")

        self.no_action_index = self.action_size - 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(self.action_size)

        self.no_action_episode_end = no_action_episode_end

        self.episodic_reward = episodic_reward

        self.max_reward_in_the_episode = 0
        self.max_rewards_in_the_episodes = []

        self.cumulative_reward_in_the_episode = 0
        self.cumulative_rewards_in_the_episodes = []

        self.max_reward = 0
        self.max_reward_k_vector = self.k_vector.copy()

        self.function_each_episode = []
        self.max_reward_dict = {}
        self.max_reward_k_vector_dict = {}

    def step(self, action):
        self.current_step = self.current_step + 1

        if self.no_action_index == action:
            if self.no_action_episode_end:
                reward = self.reward(self.k_vector)
                done = True
            else:
                reward = 0
                done = self.check_episode_end()
        else:
            valid_action = self.action_function(action)

            if not valid_action:
                reward = -self.two_to_power_dimension
            else:
                self.k_vector_gcd()
                reward = self.reward(self.k_vector)

            done = self.check_episode_end()

        if reward > self.max_reward_in_the_episode:
            self.max_reward_in_the_episode = reward

            if self.function_mode is FunctionMode.SINGLE:
                if reward > self.max_reward:
                    self.max_reward = reward
                    self.max_reward_k_vector = self.k_vector.copy()
            else:
                if self.function in self.max_reward_dict.keys():
                    if reward > self.max_reward_dict[self.function]:
                        self.max_reward_dict[self.function] = reward
                        self.max_reward_k_vector_dict[self.function] = self.k_vector.tolist().copy()
                else:
                    self.max_reward_dict[self.function] = reward
                    self.max_reward_k_vector_dict[self.function] = self.k_vector.tolist().copy()

        if self.episodic_reward:
            if done:
                returned_reward = self.max_reward_in_the_episode
            else:
                returned_reward = 0
        else:
            self.cumulative_reward_in_the_episode += reward
            returned_reward = reward

        observation = self.create_observation()
        info = {}

        return observation, returned_reward, done, info

    def step_without_action(self):
        self.current_step = self.current_step + 1
        reward = self.reward(self.k_vector)
        if reward > self.max_reward_in_the_episode:
            self.max_reward_in_the_episode = reward
        done = self.check_episode_end()

        return reward, done

    def act_increase_decrease(self, action):
        if action < self.k_vector_size * 2:
            selected_index = action % self.k_vector_size
            valid_action = True

            if action > self.increase_last_index:
                if self.k_vector[selected_index] > 1:
                    self.k_vector[selected_index] -= 1
                else:
                    valid_action = False
            else:
                self.k_vector[selected_index] += 1
        else:
            valid_action = False

        if valid_action:
            self.apply_constraints_on_k_vector(selected_index)

        return valid_action

    def act_increase(self, action):
        if action < self.k_vector_size:
            valid_action = True
            self.k_vector[action] += 1
        else:
            valid_action = False

        if valid_action:
            self.apply_constraints_on_k_vector(action)

        return valid_action

    def apply_constraints_on_k_vector(self, selected_index):
        if self.k_vector[selected_index] > self.k_vector_element_max_value:
            self.k_vector[selected_index] -= self.k_vector_element_max_value

    def check_episode_end(self):
        if self.current_step > self.steps_in_each_epoch:
            done = True
        else:
            done = False

        return done

    def reset(self):
        self.current_step = 0

        self.max_rewards_in_the_episodes.append(self.max_reward_in_the_episode)
        self.max_reward_in_the_episode = 0

        if not self.episodic_reward:
            self.cumulative_rewards_in_the_episodes.append(self.cumulative_reward_in_the_episode)
            self.cumulative_reward_in_the_episode = 0

        self.k_vector = numpy.ones(self.two_to_power_dimension)

        if self.function_mode is not FunctionMode.SINGLE:
            self.function_each_episode.append(self.function)

            if self.function_mode is FunctionMode.RANDOM:
                self.function = numpy.random.randint(self.total_number_of_functions)
            elif self.function_mode is FunctionMode.LIST:
                self.function = self.function_list[numpy.random.randint(self.function_list_len)]

            self.set_function(self.function)

        observation = self.create_observation()
        return observation

    def close(self):
        pass

    def set_function(self, function):
        self.function = function % self.total_number_of_functions
        self.function_vector = boolean_function_generator(self.function, self.dimension)
        self.q_matrix = q_matrix_generator(self.function, self.dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)

        if self.q_matrix_representation:
            self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        else:
            self.function_representation = self.walsh_spectrum

    def reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(self.calculate_weights(next_k_vector) == 0)
        reward = next_number_of_zeros**2
        return reward

    def calculate_weights(self, k_vector):
        return numpy.matmul(self.q_matrix, k_vector)

    def create_observation(self):
        observation = numpy.ones([self.state_size])
        observation[0:self.function_representation_size] = self.function_representation
        observation[self.function_representation_size:self.state_size] = self.k_vector
        return observation

    def k_vector_gcd(self):
        k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(self.k_vector, dtype=numpy.int))
        self.k_vector /= k_vector_gcd

    def switch_to_single_mode(self):
        self.function_mode = FunctionMode.SINGLE
