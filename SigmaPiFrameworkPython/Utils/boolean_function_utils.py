import numpy
import math


def bf_to_dimension(function_vector):
    function_vector_size = numpy.size(function_vector, axis=0)

    dimension = math.log2(function_vector_size)
    if not dimension.is_integer():
        raise Exception("Sizes of function vector is not power of two")

    return dimension, function_vector_size
