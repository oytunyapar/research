import numpy
import math
from SigmaPiFrameworkPython.boolean_function_generator import boolean_function_generator


def monsetup(dimension):
    two_to_the_power_dimension = 2**dimension
    d_matrix = numpy.zeros((two_to_the_power_dimension, two_to_the_power_dimension), dtype=numpy.float32)
    for power_of_dimension_iterator in range(0, two_to_the_power_dimension):
        local_power_of_dimension_iterator = power_of_dimension_iterator

        sign_vector = numpy.zeros(dimension, dtype=numpy.int8)
        for dimension_iterator in range(0, dimension):
            if local_power_of_dimension_iterator % 2 == 0:
                sign_vector[dimension_iterator] = 1
            else:
                sign_vector[dimension_iterator] = -1
            local_power_of_dimension_iterator = math.floor(local_power_of_dimension_iterator/2)

        for power_of_dimension_iterator_second in range(0, two_to_the_power_dimension):
            local_power_of_dimension_iterator = power_of_dimension_iterator_second
            multiply_factor = 1
            for dimension_iterator in range(0, dimension):
                if local_power_of_dimension_iterator % 2 == 1:
                    multiply_factor = multiply_factor * sign_vector[dimension_iterator]
                local_power_of_dimension_iterator = math.floor(local_power_of_dimension_iterator/2)
            d_matrix[power_of_dimension_iterator, power_of_dimension_iterator_second] = multiply_factor

    return d_matrix


def q_matrix_generator(function, dimension):
    q_matrix = \
        numpy.matmul(monsetup(dimension),
                     numpy.diag(boolean_function_generator(function, dimension)))

    return q_matrix


def get_functions_from_spectrum(spectrum):
    spectrum_size = numpy.size(spectrum)

    dimension = math.log2(spectrum_size)
    if not dimension.is_integer():
        print("Sizes of absolute spectrum is not power of two")
        return

    dimension = int(dimension)
    number_of_functions = 2**(2**dimension)

    spectrum = spectrum.flatten()
    spectrum = numpy.abs(spectrum)
    spectrum = numpy.sort(spectrum)

    function_list = []
    spectrum_list = []

    for function_iterator in range(number_of_functions):
        q_matrix = q_matrix_generator(function_iterator, dimension)
        function_spectrum_raw = numpy.sum(q_matrix, axis=1)
        function_spectrum = numpy.abs(function_spectrum_raw)
        function_spectrum = numpy.sort(function_spectrum)

        if (function_spectrum == spectrum).all():
            function_list.append(function_iterator)
            spectrum_list.append(function_spectrum_raw)

    return numpy.array(function_list), numpy.array(spectrum_list)

