from SigmaPiFrameworkPython import MonomialSetup as monomial_setup

import itertools

import numpy
from scipy.optimize import linprog


def monomial_exclusion_all_dimension(function, dimension):
    q_matrix = monomial_setup.q_matrix_generator(function, dimension)
    size = 2 ** dimension
    main_list = range(0, size)

    input_data = numpy.zeros([2 ** size, size], dtype=numpy.uint8)
    output_data = numpy.zeros(2 ** size, dtype=numpy.uint8)

    iterations = 0

    for iterator in range(0, size + 1):
        current_combinations = list(itertools.combinations(main_list, iterator))
        for combination in current_combinations:

            for index in combination:
                input_data[iterations, index] = 1

            if monomial_exclusion(q_matrix, size, combination):
                output_data[iterations] = 1

            iterations = iterations + 1

    return input_data, output_data


def monomial_exclusion(q_matrix, q_matrix_column_size, combination):
    result = linprog(c=numpy.ones(q_matrix_column_size),
                     A_ub=None,
                     b_ub=None,
                     A_eq=q_matrix[combination, :],
                     b_eq=numpy.zeros(combination.size),
                     bounds=(1, None))

    return result.success
