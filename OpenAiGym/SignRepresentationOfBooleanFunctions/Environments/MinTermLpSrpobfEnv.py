from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnvBase import *
from SigmaPiFrameworkPython.SigmaPiLinearProgramming import monomial_exclusion, monomial_exclusion_all_subsets
from SigmaPiFrameworkPython.Utils.CombinationUtils import binary_vector_to_combination, get_eliminated_subsets_size_dict


import numpy


class MinTermLpSrpobfEnv(MinTermSrpobfEnvBase):
    def __init__(self, function, dimension,
                 q_matrix_representation, episodic_reward):
        super(MinTermLpSrpobfEnv, self).\
            __init__(function, dimension, q_matrix_representation, episodic_reward)

        self.steps_in_each_epoch = self.two_to_power_dimension * 2
        self.action_size = self.two_to_power_dimension

        self.key_name = "monomial_set"
        self.key_size = self.two_to_power_dimension
        self.reset_key()

        self.state_size = self.function_representation_size + self.key_size

        self.max_reward_key = self.key.copy()

        if q_matrix_representation:
            super(MinTermLpSrpobfEnv, self)._create_action_and_observation_space(-1, 1, dtype=numpy.int)
        else:
            super(MinTermLpSrpobfEnv, self)._create_action_and_observation_space(-self.two_to_power_dimension,
                                                                                 self.two_to_power_dimension)

    def reset_key(self):
        self.key = numpy.zeros(self.key_size)

    def step(self, action):
        self.current_step = self.current_step + 1
        self.key[action % self.key_size] = 1

        is_feasible = \
            monomial_exclusion(self.q_matrix, self.two_to_power_dimension, binary_vector_to_combination(self.key))

        if is_feasible:
            done = self.check_episode_end()
            returned_reward = self.reward(self.key)
        else:
            done = True
            returned_reward = 0

        self.update_episode_reward_statistics(returned_reward)

        observation = self.create_observation()
        info = {}

        return observation, returned_reward, done, info

    def reward(self, next_key):
        next_number_of_zeroed_monomials = numpy.count_nonzero(next_key == 1)

        if self.function_representation_type == FunctionRepresentationType.Q_MATRIX:
            reward = next_number_of_zeroed_monomials / self.two_to_power_dimension
        else:
            reward = next_number_of_zeroed_monomials ** 2

        if reward > self.max_reward_in_the_episode:
            return reward
        else:
            return 0

    def reward_to_number_of_zeros(self, reward):
        if self.function_representation_type == FunctionRepresentationType.Q_MATRIX:
            return int(numpy.ceil(reward * self.two_to_power_dimension))
        else:
            return int(numpy.sqrt(reward))

    def get_possible_all_action_space(self):
        result = numpy.empty([0, self.state_size], dtype=numpy.int32)

        subsets, subset_elimination = monomial_exclusion_all_subsets(self.function, self.dimension)
        eliminated_subsets_size_dict = get_eliminated_subsets_size_dict(subsets, subset_elimination)

        eliminated_subsets_size_dict_keys = list(eliminated_subsets_size_dict.keys())
        eliminated_subsets_size_dict_keys.pop()

        for key in eliminated_subsets_size_dict_keys:
            subsets = eliminated_subsets_size_dict[key]
            for subset in subsets:
                self.key = subset
                result = numpy.append(result, self.create_observation().reshape([1, self.state_size]), axis=0)

        return result
