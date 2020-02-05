import numpy

def decimalToBinary(number):  
    return bin(number).replace("0b", "")

def booleanFunctionGenerator(function,dimension):
    binary_number = decimalToBinary(function)
    vector_size = 2**dimension
    binary_size = len(binary_number)

    if ( vector_size < binary_size ):
        print("Invalid function for this dimension.\n")
        return numpy.zeros( vector_size, dtype=numpy.float32 )
    
    sign_vector = numpy.ones( vector_size, dtype=numpy.float32 )
    for iterator in range (binary_size):
        if ( binary_number[binary_size - iterator - 1] == '1' ):
            sign_vector[iterator] = -1

    return sign_vector
