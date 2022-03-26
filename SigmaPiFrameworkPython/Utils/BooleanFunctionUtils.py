import numpy
import math
from SigmaPiFrameworkPython.MonomialSetup import q_matrix_generator
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *


def bf_to_dimension(function_vector):
    function_vector_size = numpy.size(function_vector, axis=0)

    dimension = math.log2(function_vector_size)
    if not dimension.is_integer():
        raise Exception("Sizes of function vector is not power of two")

    return dimension, function_vector_size


def walsh_spectrum(function, dimension):
    return q_matrix_generator(function, dimension).sum(1).astype(int)


def walsh_spectrum_compact(function, dimension):
    ws = numpy.abs(walsh_spectrum(function, dimension))
    ws_unique, counts = numpy.unique(ws, return_counts=True)
    ws_unique = numpy.flip(ws_unique).tolist()
    counts = numpy.flip(counts).tolist()

    return dict(zip(ws_unique, counts))


def function_to_equivalence_class(function, dimension):
    return BooleanFunctionsWalshSpectrumEquivalentClass[dimension][str(walsh_spectrum_compact(function, dimension))]


def function_to_equivalence_class_hex_string(function, dimension):
    return hex(function_to_equivalence_class(function, dimension))


def all_equivalence_classes_hex_string(dimension):
    return [hex(equivalence_class) for equivalence_class in BooleanFunctionsEquivalentClasses[dimension]]


def get_functions_from_walsh_spectrum(equivalence_class, dimension):
    number_of_functions = 2**(2**dimension)

    spectrum = walsh_spectrum_compact(equivalence_class, dimension)

    function_list = []
    spectrum_list = []

    for function_iterator in range(number_of_functions):
        q_matrix = q_matrix_generator(function_iterator, dimension)
        function_spectrum_raw = numpy.sum(q_matrix, axis=1)
        function_spectrum = walsh_spectrum_compact(function_iterator, dimension)

        if function_spectrum == spectrum:
            function_list.append(function_iterator)
            spectrum_list.append(function_spectrum_raw)

    return numpy.array(function_list), numpy.array(spectrum_list)


def get_complement_function_list(dimension, functions):
    number_of_functions = 2**(2**dimension)
    all_functions = list(range(number_of_functions))
    unique_functions = (numpy.unique(functions) % number_of_functions).tolist()
    return list(set(all_functions) - set(unique_functions))
