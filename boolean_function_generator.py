import numpy

def decimalToBinary(number):  
    return bin(number).replace("0b", "")

def boolean_function_generator(number,dimension):
    binary_number = decimalToBinary(number)
    vector_size = 2**dimension
    binary_size = len(binary_number)

    if ( vector_size < binary_size ):
        print("Invalid function for this dimension.\n")
        return numpy.zeros( vector_size, dtype=numpy.int8 )
    
    sign_vector = numpy.ones( vector_size, dtype=numpy.int8 )
    for iterator in range (binary_size):
        if ( binary_number[binary_size - iterator - 1] == '1' ):
            sign_vector[iterator] = -1
    
    print(sign_vector)
    return sign_vector

boolean_function_generator(65535,5)