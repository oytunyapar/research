from SigmaPiFrameworkPython.SigmaPiLinearProgramming import monomial_exclusion_iterative
import random
import numpy
import datetime
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *


def apply_random_search_linear_programming_on_functions(number_of_functions,
                                                        number_of_iterations,
                                                        dimension,
                                                        specific_functions=None,
                                                        save_dir=None):
    functions = []

    for i in range(0, number_of_functions):
        functions.append(random.getrandbits(2**dimension))

    if specific_functions is not None:
        for specific_function in specific_functions:
            specific_function = specific_function % 2**(2**dimension)
            functions.append(specific_function)

    functions = list(set(functions))
    number_of_functions = len(functions)

    function_zeroes = {}
    counter = 0

    for function in functions:
        counter += 1

        function = int(function)
        function_zeroes[function] = []
        for i in range(0, number_of_iterations):
            function_zeroes[function].append(monomial_exclusion_iterative(function, dimension).size)
            if i % 500 == 0:
                print("apply_random_search_linear_programming_on_functions iteration:" + str(i) + "/" + str(number_of_iterations))

        print("apply_random_search_linear_programming_on_functions function:" + str(counter) + "/" + str(number_of_functions))

    function_zeroes_average_std = {}
    for key in function_zeroes.keys():
        function_zeroes_average_std[key] = [numpy.average(function_zeroes[key]), numpy.std(function_zeroes[key])]

    if save_dir is not None:
        dir_name = save_dir + "/apply_random_search_linear_programming_on_functions_" + str(datetime.datetime.now())
        save_data_structure(dir_name, str(dimension) + "dim_" + "function_zeroes", function_zeroes)
        save_data_structure(dir_name, str(dimension) + "dim_" + "function_zeroes_average_std",
                            function_zeroes_average_std)

    return function_zeroes, function_zeroes_average_std


def analyze_function_zeroes_average_std(function_zeroes_average_std, dimension):
    max_average = 0
    min_average = 2 ** dimension
    average_of_averages = 0

    max_std = 0
    min_std = 2 ** dimension
    average_of_stds = 0

    keys = function_zeroes_average_std.keys()
    for key in keys:
        item = function_zeroes_average_std[key]
        current_average = item[0]
        current_std = item[1]

        if current_average > max_average:
            max_average = current_average

        if current_average < min_average:
            min_average = current_average

        average_of_averages += current_average

        if current_std > max_std:
            max_std = current_std

        if current_std < min_std:
            min_std = current_std

        average_of_stds += current_std

    average_of_averages /= len(keys)
    average_of_stds /= len(keys)

    return {"max_average": max_average, "min_average": min_average, "average_of_averages": average_of_averages,
            "max_std": max_std, "min_std": min_std, "average_of_stds": average_of_stds}
