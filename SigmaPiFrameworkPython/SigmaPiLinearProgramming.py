from SigmaPiFrameworkPython import MonomialSetup as monomial_setup

import itertools

import numpy
from scipy.optimize import linprog


def monomial_exclusion_all_subsets(function, dimension):
    q_matrix = monomial_setup.q_matrix_generator(function, dimension)
    size = 2 ** dimension
    main_list = range(0, size)

    input_data = numpy.zeros([2 ** size - 1, size], dtype=numpy.uint8)
    output_data = numpy.zeros(2 ** size - 1, dtype=numpy.uint8)

    iterations = 0

    for iterator in range(1, size + 1):
        current_combinations = list(itertools.combinations(main_list, iterator))
        for combination in current_combinations:

            for index in combination:
                input_data[iterations, index] = 1

            if monomial_exclusion(q_matrix, size, numpy.array(combination)):
                output_data[iterations] = 1

            iterations = iterations + 1

    return input_data, output_data


def monomial_exclusion_iterative(function, dimension):
    q_matrix = monomial_setup.q_matrix_generator(function, dimension)
    size = 2 ** dimension
    indexes = numpy.array(range(size))
    combination = numpy.array([], dtype=numpy.int32)

    for iterator in range(size):
        index = numpy.random.choice(indexes)
        indexes = numpy.delete(indexes, indexes == index)

        combination = numpy.append(combination, index)
        if monomial_exclusion(q_matrix, size, numpy.array(combination)) is False:
            combination = numpy.delete(combination, combination == index)

    return combination


def monomial_exclusion(q_matrix, q_matrix_column_size, combination):
    result = linprog(c=numpy.ones(q_matrix_column_size),
                     A_ub=None,
                     b_ub=None,
                     A_eq=q_matrix[combination, :],
                     b_eq=numpy.zeros(combination.size),
                     bounds=(1, None))

    return result.success
