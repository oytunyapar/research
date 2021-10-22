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


class MinTermSrpobfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension, q_matrix_representation, action_type, no_action_episode_end, episodic_reward):
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

        # Example for using image as input:
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, [self.state_size])

        self.current_step = 0

        if action_type == ActionType.INCREASE_DECREASE:
            self.steps_in_each_epoch = 3 * (self.two_to_power_dimension ** 2)
            self.action_size = 2 * self.two_to_power_dimension + 1
            self.increase_last_index = self.two_to_power_dimension - 1
            self.action_function = self.act_increase_decrease
        elif action_type == ActionType.INCREASE:
            self.steps_in_each_epoch = self.two_to_power_dimension ** 2
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

        if self.episodic_reward:
            if done:
                returned_reward = self.max_reward_in_the_episode
            else:
                returned_reward = 0
        else:
            returned_reward = reward

        observation = self.create_observation()
        info = {}

        return observation, returned_reward, done, info

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

        return valid_action

    def act_increase(self, action):
        if action < self.k_vector_size:
            valid_action = True
            self.k_vector[action] += 1
        else:
            valid_action = False

        return valid_action

    def check_episode_end(self):
        if self.current_step > self.steps_in_each_epoch:
            done = True
        else:
            done = False

        return done

    def reset(self):
        self.current_step = 0
        self.max_reward_in_the_episode = 0
        self.k_vector = numpy.ones(self.two_to_power_dimension)
        observation = self.create_observation()
        return observation

    def close(self):
        pass

    def reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(numpy.matmul(self.q_matrix, next_k_vector) == 0)
        reward = next_number_of_zeros**2
        return reward

    def create_observation(self):
        observation = numpy.ones([self.state_size])
        observation[0:self.function_representation_size] = self.function_representation
        observation[self.function_representation_size:self.state_size] = self.k_vector
        return observation

    def k_vector_gcd(self):
        k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(self.k_vector, dtype=numpy.int))
        self.k_vector /= k_vector_gcd
