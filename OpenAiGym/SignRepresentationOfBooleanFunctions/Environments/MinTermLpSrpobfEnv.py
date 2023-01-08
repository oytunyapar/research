from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnvBase import *
from SigmaPiFrameworkPython.SigmaPiLinearProgramming import monomial_exclusion, monomial_exclusion_all_subsets
from SigmaPiFrameworkPython.Utils.CombinationUtils import binary_vector_to_combination, get_eliminated_subsets_size_dict
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *
from SigmaPiFrameworkPython.Utils.HelperFunctions import *

import numpy


class MinTermLpSrpobfEnv(MinTermSrpobfEnvBase):
    def __init__(self, function, dimension,
                 function_representation_type, episodic_reward):
        super(MinTermLpSrpobfEnv, self).\
            __init__(function, dimension, function_representation_type, episodic_reward)

        self.minus_absolute_walsh_spectrum = [-abs(x) for x in self.walsh_spectrum]

        self.steps_in_each_epoch = self.two_to_power_dimension * 2
        self.action_size = self.two_to_power_dimension

        self.remaining_monomials = [*range(0, self.action_size)]
        self.selected_monomials = set()

        self.non_elimination_statistics = dict()
        self.initialize_non_elimination_statistics()

        self.action_selection_statistics = dict.fromkeys([*range(self.action_size)], 0)

        self.key_name = "monomial_set"
        self.key_size = self.two_to_power_dimension
        self.reset_internal()

        self.state_size = self.function_representation_size + self.key_size

        self.max_reward_key = self.key.copy()

        if function_representation_type is FunctionRepresentationType.Q_MATRIX:
            super(MinTermLpSrpobfEnv, self)._create_observation_space(-1, 1, dtype=numpy.int)
        else:
            super(MinTermLpSrpobfEnv, self)._create_observation_space(-self.two_to_power_dimension,
                                                                      self.two_to_power_dimension)
        super(MinTermLpSrpobfEnv, self)._create_action_space()

    def initialize_non_elimination_statistics(self):
        if self.function not in self.non_elimination_statistics:
            monomial_dict = {}
            for monomial in range(0, self.action_size):
                monomial_dict[monomial] = [0, 1]
            self.non_elimination_statistics[self.function] = monomial_dict

    def set_function(self, function):
        super(MinTermLpSrpobfEnv, self).set_function(function)
        self.minus_absolute_walsh_spectrum = [-abs(x) for x in self.walsh_spectrum]

        self.initialize_non_elimination_statistics()

    def reset_internal(self):
        self.key = numpy.zeros(self.key_size)
        self.remaining_monomials = [*range(0, self.action_size)]
        self.selected_monomials.clear()

    def step(self, action):
        self.current_step = self.current_step + 1

        action_normalized = action % self.key_size

        if action_normalized not in self.selected_monomials:
            self.key[action_normalized] = 1

            self.action_selection_statistics[action_normalized] += 1

            is_feasible = \
                monomial_exclusion(self.q_matrix, self.two_to_power_dimension, binary_vector_to_combination(self.key))

            if is_feasible:
                done = self.check_episode_end()
                returned_reward = self.reward(self.key)

                if action_normalized in self.remaining_monomials:
                    self.remaining_monomials.remove(action_normalized)

                self.selected_monomials.add(action_normalized)
                self.update_elimination_statistics()
            else:
                done = True
                returned_reward = -2
                self.update_non_elimination_statistics()

            self.update_episode_reward_statistics(returned_reward)
        else:
            done = self.check_episode_end()
            returned_reward = -1

        observation = self.create_observation()
        info = {}

        return observation, returned_reward, done, info

    def reward(self, next_key):
        return numpy.count_nonzero(next_key == 1)/self.two_to_power_dimension

    def reward_to_number_of_zeros(self, reward):
        return int(numpy.ceil(reward * self.two_to_power_dimension))

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

    def dump_env_statistics(self, output_directory):
        super(MinTermLpSrpobfEnv, self).dump_env_statistics(output_directory)
        dump_json(self.action_selection_statistics, output_directory, "action_selection_statistics")

    def update_non_elimination_statistics(self):
        elimination_penalty = 1/len(self.selected_monomials)
        for selected_monomial in self.selected_monomials:
            self.non_elimination_statistics[self.function][selected_monomial][0] -= elimination_penalty
            self.non_elimination_statistics[self.function][selected_monomial][1] += 1

    def update_elimination_statistics(self):
        for selected_monomial in self.selected_monomials:
            self.non_elimination_statistics[self.function][selected_monomial][1] += 1

    def generate_action(self):
        random_generate_action_policy_selection = numpy.random.rand()

        if random_generate_action_policy_selection < 0.25:
            remaining_monomials_aws = [self.minus_absolute_walsh_spectrum[x] for x in self.remaining_monomials]
            monomial_selection_possibility_ws = softmax(remaining_monomials_aws)
            action = numpy.random.choice(self.remaining_monomials, p=monomial_selection_possibility_ws)

        elif random_generate_action_policy_selection < 0.5:
            remaining_monomials_penalties = [self.non_elimination_statistics[self.function][x][0] /
                                             self.non_elimination_statistics[self.function][x][1] for x in
                                             self.remaining_monomials]
            if min(remaining_monomials_penalties) == 0:
                monomial_selection_possibility_non_elimination = softmax(remaining_monomials_penalties)
            else:
                monomial_selection_possibility_non_elimination =\
                    softmax([-1/min(remaining_monomials_penalties) * x for x in remaining_monomials_penalties])
            action = numpy.random.choice(self.remaining_monomials, p=monomial_selection_possibility_non_elimination)

        elif random_generate_action_policy_selection < 0.75:
            action = numpy.random.choice(self.remaining_monomials)
        else:
            action = numpy.random.choice(self.action_size)

        return action
