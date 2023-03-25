from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.Utils.StringHelperFunctions import function_to_hex_string
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import *
import json
import numpy


def max_reward_data_to_number_of_zeros(functions_root_directory, dimension, env, functions=None):
    max_zeros = {}

    if functions is None:
        functions = BooleanFunctionsEquivalentClasses[dimension]

    for function in functions:
        max_reward_file_name = "max_rewards_in_the_episodes.json"
        function_string = function_to_hex_string(dimension, function)
        json_file_name =\
            functions_root_directory + "/" + function_string + "/" + max_reward_file_name
        with open(json_file_name) as json_file:
            data = json.load(json_file)
            max_zeros[function_string] = env.reward_to_number_of_zeros(numpy.max(data))
            json_file.close()

    return max_zeros


def reward_performance(env, reward, function=None):
    precision = 2

    if function is None:
        function = env.function

    no_zeroes = env.reward_to_number_of_zeros(reward)
    dimension = env.dimension

    if dimension < 5:
        theoretical_no_zeroes = \
            BooleanFunctionsWalshSpectrumNoZeroes[dimension][str(walsh_spectrum_compact(function, dimension,
                                                                                        env.d_matrix))]

        return [theoretical_no_zeroes - no_zeroes, round(no_zeroes / theoretical_no_zeroes, precision)]
    else:
        if dimension in BooleanFunctionsEquivalentClassesDensity:
            density_dic = BooleanFunctionsEquivalentClassesDensity[dimension]
            if function in density_dic:
                theoretical_no_zeroes = 2**dimension - density_dic[function]
                return [theoretical_no_zeroes - no_zeroes, round(no_zeroes / theoretical_no_zeroes, precision)]
        return no_zeroes


def runner_overall_performance(performance):
    no_of_functions = len(performance.keys())

    if no_of_functions == 0:
        return float('inf'), float('inf')

    precision = 4

    percentages = [None] * no_of_functions
    index_counter = 0

    for _, value in performance.items():
        percentages[index_counter] = value[1]
        index_counter += 1

    return round(numpy.mean(percentages), precision), round(numpy.std(percentages), precision)


def runner_equivalence_class_performance(performance, dimension):
    equivalence_classes = all_equivalence_classes_hex_string(dimension)
    equivalence_class_seperated_performances = {}
    equivalence_class_seperated_overall_performances = {}

    for equivalence_class in equivalence_classes:
        equivalence_class_seperated_performances[equivalence_class] = {}
        equivalence_class_seperated_overall_performances[equivalence_class] = [0, 0]

    for key, value in performance.items():
        equivalence_class_seperated_performances[
            function_to_equivalence_class_hex_string(int(key), dimension)][int(key)] = value

    for key, value in equivalence_class_seperated_performances.items():
        perf_mean, perf_std = runner_overall_performance(value)
        equivalence_class_seperated_overall_performances[key][0] = perf_mean
        equivalence_class_seperated_overall_performances[key][1] = perf_std

    return equivalence_class_seperated_overall_performances
