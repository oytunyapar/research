from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.RLRunner import rl_load_model
from SigmaPiFrameworkPython.Utils.CombinationUtils import get_elimination_relation_dictionary, \
    load_elimination_relation_dictionary
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.Utils.EnvironmentHelperFunctions import *


def dqn_model_all_state_performance(model_output_directory, function, dimension,
                                    elimination_relation_data_structure=None,
                                    elimination_relation_data_structure_dir=None):
    model = rl_load_model(model_output_directory)
    env = env_creator(function, dimension, KeyType.K_VECTOR)

    if elimination_relation_data_structure is not None:
        elimination_relation_data_structure_internal = elimination_relation_data_structure
    elif elimination_relation_data_structure_dir is not None:
        elimination_relation_data_structure_internal =\
            load_elimination_relation_dictionary(function, dimension, elimination_relation_data_structure_dir)
    else:
        elimination_relation_data_structure_internal = get_elimination_relation_dictionary(function, dimension)
