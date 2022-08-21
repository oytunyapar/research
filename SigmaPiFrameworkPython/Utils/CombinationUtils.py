import numpy


def binary_vector_to_combination(binary_vector):
    binary_vector_size = len(binary_vector)
    combination = numpy.array([], dtype=numpy.int)
    for index in range(0, binary_vector_size):
        if binary_vector[index] == 1:
            combination = numpy.append(combination, index)

    return combination


def binary_vector_to_int(binary_vector):
    power = 0
    result = 0

    for item in binary_vector:
        result += item * (2 ** power)
        power += 1

    return result


def int_to_binary_vector(value, precision):
    binary_vector = numpy.zeros([precision])

    if value < 2 ** precision:
        for power in reversed(range(precision)):
            two_to_the_power = 2 ** power
            if value >= two_to_the_power:
                binary_vector[power] = 1
                value -= two_to_the_power

    return binary_vector


def get_eliminated_subsets(subsets, subset_elimination):
    return subsets[numpy.where(subset_elimination == 1), :][0]


def get_eliminated_subsets_size_dict(subsets, subset_elimination):
    eliminated_subsets = get_eliminated_subsets(subsets, subset_elimination)
    eliminated_subsets_size_dict = {}

    row_size = eliminated_subsets[0].size

    for subset_size in range(1, row_size):
        eliminated_subsets_size_dict[subset_size] = numpy.empty([0, row_size])

    for subset in eliminated_subsets:
        eliminated_subsets_size_dict[numpy.where(subset == 1)[0].size] = \
            numpy.append(eliminated_subsets_size_dict[numpy.where(subset == 1)[0].size], [subset], axis=0)

    for dic_key in list(eliminated_subsets_size_dict.keys()):
        if eliminated_subsets_size_dict[dic_key].size == 0:
            del eliminated_subsets_size_dict[dic_key]

    return eliminated_subsets_size_dict
