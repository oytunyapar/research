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

        print("apply_random_search_linear_programming_on_functions:" + str(counter) + "/" + str(number_of_functions))

    function_zeroes_average_std = {}
    for key in function_zeroes.keys():
        function_zeroes_average_std[key] = [numpy.average(function_zeroes[key]), numpy.std(function_zeroes[key])]

    if save_dir is not None:
        dir_name = save_dir + "/apply_random_search_linear_programming_on_functions_" + str(datetime.datetime.now())
        save_data_structure(dir_name, str(dimension) + "dim_" + "function_zeroes", function_zeroes)
        save_data_structure(dir_name, str(dimension) + "dim_" + "function_zeroes_average_std",
                            function_zeroes_average_std)

    return function_zeroes, function_zeroes_average_std
