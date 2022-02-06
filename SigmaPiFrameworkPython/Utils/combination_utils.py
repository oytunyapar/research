import numpy


def binary_vector_to_combination(binary_vector):
    binary_vector_size = len(binary_vector)
    combination = numpy.array([], dtype=numpy.int)
    for index in range(0, binary_vector_size):
        if binary_vector[index] == 1:
            combination = numpy.append(combination, index)

    return combination
