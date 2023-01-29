import functools
import numpy

from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnvBase import MinTermSrpobfEnvBase

from enum import Enum


class ActionType(Enum):
    INCREASE = 1
    INCREASE_DECREASE = 2


class MinTermSrpobfEnv(MinTermSrpobfEnvBase):
    metadata = {'render.modes': ['human']}

    def __init__(self, function, dimension,
                 function_representation_type,
                 action_type,
                 no_action_episode_end,
                 episodic_reward):

        super(MinTermSrpobfEnv, self).__init__(function, dimension, function_representation_type, episodic_reward)
        self.steps_in_each_epoch = ((self.two_to_power_dimension / 2) - 1) * self.two_to_power_dimension

        self.key_name = "k_vector"
        self.key_size = self.two_to_power_dimension
        self.reset_internal()
        self.k_vector_element_max_value = 2 ** (dimension - 1)

        self.max_reward_key = self.key.copy()

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

        self.no_action_episode_end = no_action_episode_end
        self.no_action_index = self.action_size - 1

        self.state_size = self.function_representation_size + self.key_size

        super(MinTermSrpobfEnv, self)._create_observation_space()
        super(MinTermSrpobfEnv, self)._create_action_space()

    def reset_internal(self):
        self.key = numpy.ones(self.key_size)

    def step(self, action):
        self.current_step = self.current_step + 1

        if self.no_action_index == action:
            if self.no_action_episode_end:
                reward = self.reward(self.key)
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
                reward = self.reward(self.key)

            done = self.check_episode_end()

        if self.episodic_reward:
            if done:
                returned_reward = self.max_reward_in_the_episode
            else:
                returned_reward = 0
        else:
            if done:
                returned_reward = self.max_reward_in_the_episode * self.steps_in_each_epoch
            else:
                returned_reward = reward

        self.update_episode_reward_statistics(returned_reward)

        observation = self.create_observation()
        info = {}

        return observation, returned_reward, done, info

    def step_without_action(self):
        self.current_step = self.current_step + 1
        reward = self.reward(self.key)
        if reward > self.max_reward_in_the_episode:
            self.max_reward_in_the_episode = reward
        done = self.check_episode_end()

        return reward, done

    def act_increase_decrease(self, action):
        if action < self.key_size * 2:
            selected_index = action % self.key_size
            valid_action = True

            if action > self.increase_last_index:
                if self.key[selected_index] > 1:
                    self.key[selected_index] -= 1
                else:
                    valid_action = False
            else:
                self.key[selected_index] += 1
        else:
            valid_action = False

        if valid_action:
            self.apply_constraints_on_k_vector(selected_index)

        return valid_action

    def act_increase(self, action):
        if action < self.key_size:
            valid_action = True
            self.key[action] += 1
        else:
            valid_action = False

        if valid_action:
            self.apply_constraints_on_k_vector(action)

        return valid_action

    def apply_constraints_on_k_vector(self, selected_index):
        if self.key[selected_index] > self.k_vector_element_max_value:
            self.key[selected_index] -= self.k_vector_element_max_value

    def close(self):
        pass

    def reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(self.calculate_weights(next_k_vector) == 0)
        reward = next_number_of_zeros**2
        return reward

    def reward_to_number_of_zeros(self, reward):
        return int(numpy.sqrt(reward))

    def calculate_weights(self, k_vector):
        return numpy.matmul(self.q_matrix, k_vector)

    def k_vector_gcd(self):
        k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(self.key, dtype=numpy.int))
        self.key /= k_vector_gcd

    def env_specific_configuration(self):
        return super(MinTermSrpobfEnv, self).env_specific_configuration() | {"action_type": str(self.action_type)}
