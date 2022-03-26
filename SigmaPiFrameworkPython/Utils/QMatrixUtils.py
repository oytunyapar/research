import numpy
import math


def q_matrix_to_dimension(q_matrix):
    q_matrix_row_size = numpy.size(q_matrix, axis=0)
    q_matrix_column_size = numpy.size(q_matrix, axis=1)

    if q_matrix_row_size != q_matrix_column_size:
        raise Exception("Q matrix is not a square matrix")

    dimension = math.log2(q_matrix_row_size)

    if not dimension.is_integer():
        raise Exception("Q matrix dimensions are not power of two")

    return dimension, q_matrix_row_size
