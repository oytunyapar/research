import numpy
import math
import boolean_function_generator

def monsetup(dimension):
    two_to_the_power_dimension = 2**dimension
    D_Matrix = numpy.zeros( ( two_to_the_power_dimension, two_to_the_power_dimension ), dtype=numpy.float32 )
    for power_of_dimension_iterator in range( 0, two_to_the_power_dimension ):
        local_power_of_dimension_iterator = power_of_dimension_iterator

        sign_vector = numpy.zeros( dimension, dtype=numpy.int8 )
        for dimension_iterator in range( 0, dimension ):
            if ( local_power_of_dimension_iterator % 2 == 0 ):
                sign_vector[dimension_iterator] = 1
            else:
                sign_vector[dimension_iterator] = -1
            local_power_of_dimension_iterator = math.floor(local_power_of_dimension_iterator/2)

        for power_of_dimension_iterator_second in range( 0, two_to_the_power_dimension ):
            local_power_of_dimension_iterator = power_of_dimension_iterator_second
            multiply_factor = 1
            for dimension_iterator in range( 0, dimension ):
                if ( local_power_of_dimension_iterator % 2 == 1 ):
                    multiply_factor = multiply_factor * sign_vector[dimension_iterator]
                local_power_of_dimension_iterator = math.floor(local_power_of_dimension_iterator/2)
            D_Matrix[ power_of_dimension_iterator, power_of_dimension_iterator_second ] = multiply_factor

    return D_Matrix
    
def qMatrixGenerator(function,dimension):
    Q_Matrix = \
        numpy.matmul( monsetup( dimension ),
                      numpy.diag( boolean_function_generator.\
                      booleanFunctionGenerator( function, dimension ) ) )

    return Q_Matrix
