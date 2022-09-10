import numpy


def decimal_to_binary(number):
    return bin(number).replace("0b", "")


def boolean_function_generator(function, dimension):
    binary_number = decimal_to_binary(function)
    vector_size = 2**dimension
    binary_size = len(binary_number)

    if vector_size < binary_size:
        print("Invalid function for this dimension.\n")
        return numpy.zeros(vector_size, dtype=numpy.float32)
    
    sign_vector = numpy.ones(vector_size, dtype=numpy.float32)
    for iterator in range(binary_size):
        if binary_number[binary_size - iterator - 1] == '1':
            sign_vector[iterator] = -1

    return sign_vector


def algebraic_logic_to_decimal(function):
    length_of_vector = len(function)
    dimension = numpy.log2(length_of_vector)

    if dimension.is_integer() is False:
        raise Exception("Function length is not power of 2.")

    power = 0
    result = 0

    for element in function:
        if element == -1:
            result += 2 ** power
        elif element == 1:
            pass
        else:
            raise Exception("Unknown element type")

        power += 1

    return result


def logic_boolean_to_algebraic_logic_conversion(function):
    size_of_input = len(function)
    result = [None] * size_of_input
    counter = 0

    for function_element in function:
        if function_element is True:
            result[counter] = -1
        elif function_element is False:
            result[counter] = 1
        else:
            raise Exception("Unknown element type.")
        counter += 1

    return result
