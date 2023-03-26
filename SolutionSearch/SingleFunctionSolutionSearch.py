from SigmaPiFrameworkPython.Applications.RandomLogicLinearProgramming import *
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.RLRunner import *
from SignRepresentationNN.prune_runner import *

from enum import Enum


class SearchPolicy(Enum):
    RANDOM = 0,
    RL = 1,
    REGULARIZATION = 2


def single_function_solution(function, dimension, search_policy, arguments=None):
    if search_policy is SearchPolicy.RANDOM:
        data, _ = apply_random_search_linear_programming_on_functions(0, 1, dimension, [function])
        number_of_zeros = data[function][0]
    elif search_policy is SearchPolicy.RL:
        env, _, training_data_performance_results, _ = \
            rl_runner_functions(function, dimension, time_steps=arguments, key_type=KeyType.MONOMIAL_SET)
        number_of_zeros = training_data_performance_results[function][0]
    elif search_policy is SearchPolicy.REGULARIZATION:
        number_of_zeros = prune_runner_single(function, dimension, arguments)
    else:
        raise Exception("Unknown policy.")

    return number_of_zeros
