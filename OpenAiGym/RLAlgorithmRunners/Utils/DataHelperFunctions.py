from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import reward_to_number_of_zeros
from OpenAiGym.RLAlgorithmRunners.Utils.StringHelperFunctions import function_to_hex_string
import json
import numpy


def max_reward_data_to_number_of_zeros(functions_root_directory, dimension, functions=None):
    max_zeros = {}

    if functions is None:
        functions = BooleanFunctionsEquivalentClasses[dimension]

    for function in functions:
        max_reward_file_name = "max_rewards_in_the_episodes.json"
        function_string = function_to_hex_string(dimension, function)
        json_file_name =\
            functions_root_directory + "/" + function_string + "/" +max_reward_file_name
        with open(json_file_name) as json_file:
            data = json.load(json_file)
            max_zeros[function_string] = reward_to_number_of_zeros(numpy.max(data))
            json_file.close()

    return max_zeros
