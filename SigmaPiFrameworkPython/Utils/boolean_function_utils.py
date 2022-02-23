import numpy
import math
from SigmaPiFrameworkPython.monomial_setup import q_matrix_generator


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


def get_functions_from_walsh_spectrum(spectrum):
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
