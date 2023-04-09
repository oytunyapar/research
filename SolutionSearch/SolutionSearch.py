from SigmaPiFrameworkPython.Applications.RandomLogicLinearProgramming import *
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.RLRunner import *
from SignRepresentationNN.prune_runner import *
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *
from Utils.DumpOutputs import dump_json, dump_csv
import datetime
import time
import numpy

from enum import Enum


class SearchPolicy(Enum):
    RANDOM = 0,
    RL = 1,
    REGULARIZATION = 2


def random_solution_search(function, dimension):
    data, _ = apply_random_search_linear_programming_on_functions(0, 1, dimension, [function])
    return data[function][0]


def rl_solution_search(function, dimension, time_steps):
    env, _, training_data_performance_results, _ = \
        rl_runner_functions(function, dimension, time_steps=time_steps, key_type=KeyType.MONOMIAL_SET)
    return training_data_performance_results[function][0]


def solution_search_object(search_policy, arguments=None):
    if search_policy is SearchPolicy.RANDOM:
        def func(function, dimension): return random_solution_search(function, dimension)
    elif search_policy is SearchPolicy.RL:
        def func(function, dimension): return rl_solution_search(function, dimension, time_steps=arguments)
    elif search_policy is SearchPolicy.REGULARIZATION:
        def func(function, dimension): return prune_runner_single(function, dimension,
                                                                  prune_runner_configuration=arguments)
    else:
        raise Exception("Unknown policy.")

    return func


def solution_search_policy_string(search_policy):
    if search_policy is SearchPolicy.RANDOM:
        string = str("RANDOM")
    elif search_policy is SearchPolicy.RL:
        string = str("RL")
    elif search_policy is SearchPolicy.REGULARIZATION:
        string = str("REGULARIZATION")
    else:
        raise Exception("Unknown policy.")

    return string


def solution_search_dump_extra(search_policy, dir_name, arguments):
    if search_policy is SearchPolicy.RL:
        dump_json(str(arguments), dir_name, "time_steps")
    elif search_policy is SearchPolicy.REGULARIZATION:
        dump_json(prune_runner_parameters(arguments), dir_name, "parameters")


def solution_search(functions_in_dimensions, search_policy, number_of_runs=1, arguments=None,
                    output_dir="/home/oytun/PycharmProjects/research/Data/solution_search/"):
    data = {}
    solution_search_impl = solution_search_object(search_policy, arguments)

    for dimension in functions_in_dimensions.keys():
        data[dimension] = {}
        num_functions = len(functions_in_dimensions[dimension])
        function_counter = 0
        for function in functions_in_dimensions[dimension]:
            data[dimension][function] = []
            function_counter += 1
            time_in_seconds = time.time()
            for run in range(number_of_runs):
                data[dimension][function].append(solution_search_impl(function, dimension))
            print("Dimension:", dimension, " Function:", function_counter, "/", num_functions,
                  " Elapsed time:", time.time() - time_in_seconds)

    dir_name = output_dir + "/" + solution_search_policy_string(search_policy) + "/" + str(datetime.datetime.now())
    save_data_structure(dir_name, "dictionary", data)
    solution_search_dump_extra(search_policy, dir_name, arguments)

    max_data = max_number_of_zeros_in_data(data=data)
    for dimension in max_data.keys():
        rows = max_data_dict_to_rows(max_data[dimension])
        dump_csv(["function", "value"], rows, dir_name, "dimension_" + str(dimension))

    return data


def equivalence_classes_solution_search(search_policy, number_of_runs=1, arguments=None,
                                        output_dir="/home/oytun/PycharmProjects/research/Data/solution_search/"):
    solution_search(BooleanFunctionsEquivalentClasses, search_policy, number_of_runs, arguments, output_dir)


def max_number_of_zeros_in_data(directory=None, file=None, data=None):
    if data is None:
        data = open_data_structure(directory, file)

    max_data = {}
    for dimension in data.keys():
        current_functions = data[dimension]
        max_data[dimension] = {}
        for function in current_functions.keys():
            values = current_functions[function]
            if len(values) > 0:
                max_data[dimension][hex(function)] = numpy.max(values)
            else:
                max_data[dimension][hex(function)] = None

    return max_data


def max_data_dict_to_rows(max_data_for_a_dimension):
    rows = []
    for key, value in max_data_for_a_dimension.items():
        rows.append([key, value])

    return rows

