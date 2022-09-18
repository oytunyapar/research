import numpy

from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.RLRunner import rl_load_model
from SigmaPiFrameworkPython.Utils.CombinationUtils import *
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.Utils.EnvironmentHelperFunctions import *


def dqn_model_all_state_performance(model_output_directory, function, dimension,
                                    elimination_relation_data_structure=None,
                                    elimination_relation_data_structure_dir=None):
    model = rl_load_model(model_output_directory)
    env = env_creator(function, dimension, KeyType.MONOMIAL_SET)

    if elimination_relation_data_structure is not None:
        elimination_relation_data_structure_internal = elimination_relation_data_structure
    elif elimination_relation_data_structure_dir is not None:
        elimination_relation_data_structure_internal =\
            load_elimination_relation_dictionary(function, dimension, elimination_relation_data_structure_dir)
    else:
        elimination_relation_data_structure_internal = get_elimination_relation_dictionary(function, dimension)

    state_space = env.get_possible_all_state_space()
    action_space_keys = list(state_space.keys())

    possible_points = 0
    obtained_points = 0

    for key in action_space_keys:
        current_state_space = state_space[key]

        possible_actions = numpy.empty(0, dtype=numpy.int)
        for state in current_state_space:
            possible_functions_and_actions = \
                elimination_relation_data_structure_internal[key][binary_vector_to_int(env.get_key_from_state(state))]

            for possible_function_and_action in possible_functions_and_actions:
                possible_actions = numpy.append(possible_actions, possible_function_and_action[1])

            possible_actions_size = possible_actions.size

            obs, vec = model.policy.obs_to_tensor(state)
            predicted_actions_sorted = numpy.argsort(model.q_net(obs).flatten().tolist())
            predicted_actions_sorted_size = predicted_actions_sorted.size

            possible_points += (2 * (predicted_actions_sorted_size - 1) - possible_actions_size - 1) * \
                               (possible_actions_size / 2)

            for possible_action in possible_actions:
                obtained_points += numpy.where(predicted_actions_sorted == possible_action)[0][0]

    return obtained_points/possible_points
