import numpy
import math
import random
import itertools
from SigmaPiFrameworkPython.MonomialSetup import q_matrix_generator, monomial_setup
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *


def bf_to_dimension(function_vector):
    function_vector_size = numpy.size(function_vector, axis=0)

    dimension = math.log2(function_vector_size)
    if not dimension.is_integer():
        raise Exception("Sizes of function vector is not power of two")

    return dimension, function_vector_size


def walsh_spectrum(function, dimension, hadamard_matrix=None):
    return q_matrix_generator(function, dimension, hadamard_matrix).sum(1).astype(int)


def walsh_spectrum_compact(function, dimension, hadamard_matrix=None):
    ws = numpy.abs(walsh_spectrum(function, dimension, hadamard_matrix))
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


def get_random_sample_of_functions(dimension, sample_size):
    number_of_functions = 2 ** (2 ** dimension)
    min_sample_size = 1
    max_sample_size = number_of_functions

    sample_size = int(sample_size)

    if min_sample_size > sample_size or sample_size > max_sample_size:
        print("Invalid sample size:" + str(sample_size) +
              " Min:" + str(min_sample_size) + " Max:" + str(max_sample_size))
        raise Exception("Invalid sample size")

    return random.sample(range(0, number_of_functions), sample_size)


def get_equivalence_class_samples(dimension, sample_size):
    functions = get_random_sample_of_functions(dimension, sample_size)

    sampled_functions = {}
    equivalence_class_strings = all_equivalence_classes_hex_string(dimension)

    for equivalence_class_string in equivalence_class_strings:
        sampled_functions[equivalence_class_string] = []

    for function in functions:
        equivalence_class_string = function_to_equivalence_class_hex_string(function, dimension)
        sampled_functions[equivalence_class_string].append(function)

    return sampled_functions


def get_compact_ws_key_equivalence_classes(dimension, iterations):
    functions = get_random_sample_of_functions(dimension, iterations)
    ws_keys = set()
    hadamard_matrix = monomial_setup(dimension)
    for function in functions:
        ws_keys.add(str(walsh_spectrum_compact(function, dimension, hadamard_matrix)))

    return ws_keys


def get_equivalence_class_samples_compact_ws_key(dimension, sample_size):
    functions = get_random_sample_of_functions(dimension, sample_size)
    sampled_functions = {}
    hadamard_matrix = monomial_setup(dimension)

    for function in functions:
        function_walsh_spectrum = str(walsh_spectrum_compact(function, dimension, hadamard_matrix))
        if function_walsh_spectrum in sampled_functions.keys():
            sampled_functions[function_walsh_spectrum].append(function)
        else:
            sampled_functions[function_walsh_spectrum] = [function]

    return sampled_functions


def get_functions_from_walsh_spectrum(equivalence_class, dimension, sample_size=None):
    number_of_functions = 2**(2**dimension)

    hadamard_matrix = monomial_setup(dimension)

    spectrum = walsh_spectrum_compact(equivalence_class, dimension, hadamard_matrix)

    function_list = []
    spectrum_list = []

    for function_iterator in range(number_of_functions):
        q_matrix = q_matrix_generator(function_iterator, dimension, hadamard_matrix)
        function_spectrum_raw = numpy.sum(q_matrix, axis=1)
        function_spectrum = walsh_spectrum_compact(function_iterator, dimension, hadamard_matrix)

        if function_spectrum == spectrum:
            function_list.append(function_iterator)
            spectrum_list.append(function_spectrum_raw)

            if sample_size is not None and len(function_list) == sample_size:
                break

    return numpy.array(function_list), numpy.array(spectrum_list)


def get_complement_function_list(dimension, functions):
    number_of_functions = 2**(2**dimension)
    all_functions = list(range(number_of_functions))
    unique_functions = (numpy.unique(functions) % number_of_functions).tolist()
    return list(set(all_functions) - set(unique_functions))


def create_truth_table_input(dimension):
    return list(itertools.product([False, True], repeat=dimension))


def create_truth_table_output(dimension, boolean_function_callback):
    result = [None] * 2**dimension
    truth_table_input = create_truth_table_input(dimension)
    counter = 0
    for current_row in truth_table_input:
        result[counter] = boolean_function_callback(current_row)
        counter += 1

    return result
