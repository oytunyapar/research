import itertools

import numpy
from scipy.optimize import linprog

from SigmaPiFrameworkPython import monomial_setup as monomial_setup


# linprog(c = numpy.ones(2**dimension),
# A_ub = None,
# b_ub = None,
# A_eq = Q_matrix[combination,:],
# b_eq = numpy.zeros(iterator),
# bounds = (0,None))


def linear_programming_dimension_result(function, dimension):
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

            result = linprog(c=numpy.ones(size),
                             A_ub=None,
                             b_ub=None,
                             A_eq=q_matrix[combination, :],
                             b_eq=numpy.zeros(iterator),
                             bounds=(0.05, None))

            if result.success:
                output_data[iterations] = 1

            iterations = iterations + 1

    return input_data, output_data
