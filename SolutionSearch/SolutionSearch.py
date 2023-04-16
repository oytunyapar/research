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
    GREEDY = 0,
    RL = 1,
    REGULARIZATION = 2


def random_solution_search(function, dimension):
    data, _ = apply_random_search_linear_programming_on_functions(0, 1, dimension, [function])
    return data[function][0]


def rl_solution_search(function, dimension, time_steps):
    env, _, training_data_performance_results, _ = \
        rl_runner_functions(function, dimension, time_steps=time_steps, key_type=KeyType.MONOMIAL_SET)
    return training_data_performance_results[function][0]


def get_rl_time_steps(dimension):
    return 750 * 2 ** dimension


def solution_search_object(search_policy, arguments=None):
    if search_policy is SearchPolicy.GREEDY:
        def func(function, dimension): return random_solution_search(function, dimension)
    elif search_policy is SearchPolicy.RL:
        if arguments is None:
            def func(function, dimension): return rl_solution_search(function, dimension,
                                                                     time_steps=get_rl_time_steps(dimension))
        else:
            def func(function, dimension): return rl_solution_search(function, dimension, time_steps=arguments)
    elif search_policy is SearchPolicy.REGULARIZATION:
        def func(function, dimension): return prune_runner_single(function, dimension,
                                                                  prune_runner_configuration=arguments)
    else:
        raise Exception("Unknown policy.")

    return func


def solution_search_policy_string(search_policy):
    if search_policy is SearchPolicy.GREEDY:
        string = str("GREEDY")
    elif search_policy is SearchPolicy.RL:
        string = str("RL")
    elif search_policy is SearchPolicy.REGULARIZATION:
        string = str("REGULARIZATION")
    else:
        raise Exception("Unknown policy.")

    return string


def solution_search_dump_extra(search_policy, dir_name, arguments):
    if search_policy is SearchPolicy.REGULARIZATION:
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

    max_data, _ = process_search_data(process_policy=numpy.max, data=data)
    for dimension in max_data.keys():
        rows = processed_data_dict_to_rows(max_data[dimension])
        dump_csv(["function", "max"], rows, dir_name, "max_dimension_" + str(dimension))

    average_data, compared_data = process_search_data(process_policy=data_average_std, data=data)
    for dimension in average_data.keys():
        rows = processed_data_dict_to_rows(average_data[dimension])
        dump_csv(["function", "average", "std"], rows, dir_name, "average_dimension_" + str(dimension))
        rows = processed_data_dict_to_rows(compared_data[dimension])
        dump_csv(["function", "ratio"], rows, dir_name, "theoretical_compare_dimension_" + str(dimension))

    return data


def equivalence_classes_solution_search(search_policy, number_of_runs=1, arguments=None,
                                        output_dir="/home/oytun/PycharmProjects/research/Data/solution_search/"):
    return solution_search(BooleanFunctionsEquivalentClasses, search_policy, number_of_runs, arguments, output_dir)


def dimension_solution_search(search_policy, dimension, number_of_runs=1, arguments=None,
                              output_dir="/home/oytun/PycharmProjects/research/Data/solution_search/"):
    functions = {dimension: BooleanFunctionsEquivalentClasses[dimension]}
    return solution_search(functions, search_policy, number_of_runs, arguments, output_dir)


def data_average_std(data):
    precision = 2
    return [round(numpy.average(data), precision), round(numpy.std(data), precision)]


def process_search_data(process_policy, directory=None, file=None, data=None):
    if data is None:
        data = open_data_structure(directory, file)

    processed_data = {}
    compared_data = {}

    for dimension in data.keys():
        current_functions = data[dimension]
        processed_data[dimension] = {}
        compared_data[dimension] = {}

        for function in current_functions.keys():
            values = current_functions[function]
            if len(values) > 0:
                process_result = process_policy(values)
                processed_data[dimension][hex(function)] = process_result

                if isinstance(process_result, list):
                    value = process_result[0]
                else:
                    value = process_result

                compared_data[dimension][hex(function)] = \
                    round(value/(2**dimension - BooleanFunctionsEquivalentClassesDensity[dimension][function]), 2)
            else:
                processed_data[dimension][hex(function)] = None

    return processed_data, compared_data


def processed_data_dict_to_rows(max_data_for_a_dimension):
    rows = []
    for key, value in max_data_for_a_dimension.items():
        if isinstance(value, list):
            row = [key]
            row.extend(value)
            rows.append(row)
        else:
            rows.append([key, value])

    return rows

