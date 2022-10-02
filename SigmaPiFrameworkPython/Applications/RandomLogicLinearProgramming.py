from SigmaPiFrameworkPython.SigmaPiLinearProgramming import monomial_exclusion_iterative
import random
import numpy


def apply_random_search_linear_programming_on_functions(number_of_functions,
                                                        number_of_iterations,
                                                        dimension,
                                                        specific_functions=None):
    functions = numpy.array([])

    for i in range(0, number_of_functions):
        functions = numpy.append(functions, random.getrandbits(2**dimension))

    if specific_functions is not None:
        specific_functions = numpy.array(specific_functions) % 2**(2**dimension)
        functions = numpy.append(functions, specific_functions)

    functions = numpy.unique(functions)

    function_zeroes = {}
    for function in functions:
        function = int(function)
        function_zeroes[function] = []
        for i in range(0, number_of_iterations):
            function_zeroes[function].append(monomial_exclusion_iterative(function, dimension).size)

    function_zeroes_average_std = {}
    for key in function_zeroes.keys():
        function_zeroes_average_std[key] = [numpy.average(function_zeroes[key]), numpy.std(function_zeroes[key])]

    return function_zeroes, function_zeroes_average_std
