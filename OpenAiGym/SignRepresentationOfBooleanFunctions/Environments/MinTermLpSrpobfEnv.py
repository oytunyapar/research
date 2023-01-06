from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnvBase import *
from SigmaPiFrameworkPython.SigmaPiLinearProgramming import monomial_exclusion, monomial_exclusion_all_subsets
from SigmaPiFrameworkPython.Utils.CombinationUtils import binary_vector_to_combination, get_eliminated_subsets_size_dict
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *
from SigmaPiFrameworkPython.Utils.HelperFunctions import *

import numpy


class MinTermLpSrpobfEnv(MinTermSrpobfEnvBase):
    def __init__(self, function, dimension,
                 q_matrix_representation, episodic_reward):
        super(MinTermLpSrpobfEnv, self).\
            __init__(function, dimension, q_matrix_representation, episodic_reward)

        self.steps_in_each_epoch = self.two_to_power_dimension * 2
        self.action_size = self.two_to_power_dimension

        self.remaining_monomials = [*range(0, self.action_size)]
        self.selected_monomials = set()
        self.non_elimination_statistics = dict.fromkeys(self.remaining_monomials, 0)

        self.key_name = "monomial_set"
        self.key_size = self.two_to_power_dimension
        self.reset_internal()

        self.state_size = self.function_representation_size + self.key_size

        self.max_reward_key = self.key.copy()

        if q_matrix_representation:
            super(MinTermLpSrpobfEnv, self)._create_observation_space(-1, 1, dtype=numpy.int)
            super(MinTermLpSrpobfEnv, self)._create_action_space(self.generate_action)
        else:
            super(MinTermLpSrpobfEnv, self)._create_observation_space(-self.two_to_power_dimension,
                                                                      self.two_to_power_dimension)
            super(MinTermLpSrpobfEnv, self)._create_action_space(self.generate_action)

    def reset_internal(self):
        self.key = numpy.zeros(self.key_size)
        self.remaining_monomials = [*range(0, self.action_size)]
        self.selected_monomials.clear()

    def step(self, action):
        self.current_step = self.current_step + 1

        action_normalized = action % self.key_size
        self.key[action_normalized] = 1

        is_feasible = \
            monomial_exclusion(self.q_matrix, self.two_to_power_dimension, binary_vector_to_combination(self.key))

        if is_feasible:
            done = self.check_episode_end()
            returned_reward = self.reward(self.key)

            if action_normalized in self.remaining_monomials:
                self.remaining_monomials.remove(action_normalized)

            self.selected_monomials.add(action_normalized)
        else:
            done = True
            returned_reward = -1
            self.update_non_elimination_statistics()

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

    def get_possible_all_state_space(self, directory=None):
        data_structure_file = str(self.dimension) + "dim_" + str(hex(self.function)) + "_possible_all_state_space"

        try:
            return open_data_structure(directory, data_structure_file)
        except:
            subsets, subset_elimination = monomial_exclusion_all_subsets(self.function, self.dimension)
            eliminated_subsets_size_dict = get_eliminated_subsets_size_dict(subsets, subset_elimination)

            eliminated_subsets_size_dict_keys = list(eliminated_subsets_size_dict.keys())
            eliminated_subsets_size_dict_keys.pop()

            result = {}

            for key in eliminated_subsets_size_dict_keys:
                subsets = eliminated_subsets_size_dict[key]
                current_matrix = numpy.empty([0, self.state_size], dtype=numpy.int32)
                for subset in subsets:
                    self.key = subset
                    current_matrix = numpy.append(current_matrix, self.create_observation().reshape([1, self.state_size]),
                                                  axis=0)
                result[key] = current_matrix

            if directory is not None:
                save_data_structure(directory, data_structure_file, result)

            return result

    def update_non_elimination_statistics(self):
        elimination_penalty = 1/len(self.selected_monomials)
        for selected_monomial in self.selected_monomials:
            self.non_elimination_statistics[selected_monomial] -= elimination_penalty

    def generate_action(self):
        remaining_monomials_aws = [self.absolute_walsh_spectrum[x] for x in self.remaining_monomials]
        monomial_selection_possibility_ws = softmax(remaining_monomials_aws)

        remaining_monomials_penalties = [self.non_elimination_statistics[x] for x in self.remaining_monomials]
        monomial_selection_possibility_non_elimination = softmax(remaining_monomials_penalties)

        combined_possibility = (monomial_selection_possibility_ws + monomial_selection_possibility_non_elimination)/2

        return numpy.random.choice(self.remaining_monomials, p=[float(i)/sum(combined_possibility) for i in
                                                                combined_possibility])
