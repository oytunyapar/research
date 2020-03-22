import monsetup

import math
import numpy
import itertools
from scipy.optimize import linprog

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation

def linprog_result(function,dimension):
    Q_matrix = monsetup.qMatrixGenerator(function,dimension)
    size = 2**dimension
    main_list=range(0,size)

    input_data = numpy.zeros([2**(size),size],dtype=numpy.uint8)
    output_data = numpy.zeros(2**(size),dtype=numpy.uint8)

    iterations = 0

    for iterator in range(0,size + 1):
        current_combinations = list(itertools.combinations(main_list, iterator))
        for combination in current_combinations:

            for index in combination:
                input_data[iterations,index] = 1
            
            result = linprog( c = numpy.ones(size), \
                              A_ub = None, \
                              b_ub = None, \
                              A_eq = Q_matrix[combination,:], \
                              b_eq = numpy.zeros(iterator), \
                              bounds = (0.05,None) )
            
            if ( result.success ):
                output_data[iterations] = 1
            
            iterations = iterations + 1

    return input_data,output_data

def create_dataset(Q_matrix, input_data, output_data):
    Q_matrix_row_size = numpy.size(Q_matrix,axis=0)
    Q_matrix_column_size = numpy.size(Q_matrix,axis=1)

    if ( Q_matrix_row_size != Q_matrix_column_size ):
        print("Q matrix is not a square matrix")
        return

    dimension = math.log2(Q_matrix_row_size)
    if ( not dimension.is_integer() ):
        print("Sizes of Q matrix is not power of two")
        return

    input_data_row_size = numpy.size(input_data,axis=0)
    input_data_column_size = numpy.size(input_data,axis=1)

    if ( input_data_column_size != Q_matrix_row_size ):
        print("Input column and Q matrix row sizes does not match.")
        return

    if ( input_data_row_size != numpy.size(output_data) ):
        print("Input row and output length does not match.")
        return
    
    q_flattened = numpy.reshape(Q_matrix,[1,Q_matrix_row_size*Q_matrix_column_size])

    train_ds = numpy.zeros( [ input_data_row_size, Q_matrix_row_size*Q_matrix_column_size +  input_data_column_size ], dtype=numpy.float32 )

    for iterator in range ( input_data_row_size ):
        train_ds[iterator, :] = numpy.concatenate( (q_flattened, numpy.reshape(input_data[iterator, :],[1,input_data_column_size])), axis=1)
    
    return train_ds, output_data

def monomial_exclusion_linear_programming_nn( function,dimension ):
    Q_matrix = monsetup.qMatrixGenerator(function,dimension)
    input_data, output_data = linprog_result(function,dimension)
    train_ds, output_data = create_dataset(Q_matrix, input_data, output_data)

    Input_Layer_Size = (2**dimension)**2 + 2**dimension
    First_Hidden_Layer_Size = 2*dimension*(2**dimension)
    Second_Hidden_Layer_Size = 2**dimension
    Output_Layer_Size = 1

    model = Sequential()
    model.add( Dense(First_Hidden_Layer_Size, activation='relu', input_dim=Input_Layer_Size, name="First") )
    model.add( Dense(Second_Hidden_Layer_Size, activation='relu', input_dim=First_Hidden_Layer_Size, name="Second") )
    model.add( Dense(Output_Layer_Size, activation='tanh', name="Output") )

    model.compile(optimizer='adam',\
                loss='mean_squared_error',\
                metrics=['accuracy'])

    model.fit(train_ds, output_data, epochs=10, batch_size=32)
    return 0

#linprog( c = numpy.ones(2**dimension), A_ub = None, b_ub = None, A_eq = Q_matrix[combination,:], b_eq = numpy.zeros(iterator), bounds = (0,None)  )