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


def get_equivalence_class_samples(dimension, sample_size, equivalence_class=None):
    number_of_functions = 2 ** (2 ** dimension)
    min_sample_size = 1
    max_sample_size = int(number_of_functions / len(BooleanFunctionsEquivalentClasses[dimension]))

    if min_sample_size > sample_size or sample_size > max_sample_size:
        print("Invalid sample size:" + str(sample_size) +
              " Min:" + str(min_sample_size) + " Max:" + str(max_sample_size))
        return None

    if equivalence_class is not None:
        equivalence_class_spectrum = walsh_spectrum_compact(equivalence_class, dimension)
        sampled_functions = set()

        for function in range(number_of_functions):
            if equivalence_class_spectrum == walsh_spectrum_compact(function, dimension):
                sampled_functions.add(function)

                if len(sampled_functions) == sample_size:
                    break
    else:
        sampled_functions = {}
        finished_map = {}
        equivalence_class_strings = all_equivalence_classes_hex_string(dimension)

        for equivalence_class_string in equivalence_class_strings:
            sampled_functions[equivalence_class_string] = set()
            finished_map[equivalence_class_string] = False

        for function in range(number_of_functions):
            equivalence_class_string = function_to_equivalence_class_hex_string(function, dimension)
            if len(sampled_functions[equivalence_class_string]) < sample_size:
                sampled_functions[equivalence_class_string].add(function)
            else:
                finished_map[equivalence_class_string] = True

            if all(value for value in finished_map.values()):
                break

    return sampled_functions


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
