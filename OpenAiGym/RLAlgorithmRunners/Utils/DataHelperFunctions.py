from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.RLAlgorithmRunners.Utils.StringHelperFunctions import function_to_hex_string
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsWalshSpectrumNoZeroes
from SigmaPiFrameworkPython.Utils.boolean_function_utils import *
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

    theoretical_no_zeroes = \
        BooleanFunctionsWalshSpectrumNoZeroes[dimension][str(walsh_spectrum_compact(function, dimension))]

    return [theoretical_no_zeroes - no_zeroes, round(no_zeroes / theoretical_no_zeroes, precision)]


def runner_overall_performance(performance):
    no_of_functions = len(performance.keys())
    precision = 4

    percentages = [None] * no_of_functions
    index_counter = 0

    for _, value in performance.items():
        percentages[index_counter] = value[1]
        index_counter += 1

    return round(numpy.mean(percentages), precision), round(numpy.std(percentages), precision)