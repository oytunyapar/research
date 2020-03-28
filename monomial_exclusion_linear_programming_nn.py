import monsetup
import boolean_function_generator as bf

import math
import numpy
import itertools
from scipy.optimize import linprog

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation


def linprog_result(function, dimension):
    Q_matrix = monsetup.qMatrixGenerator(function, dimension)
    size = 2**dimension
    main_list = range(0, size)

    input_data = numpy.zeros([2**size, size], dtype=numpy.uint8)
    output_data = numpy.zeros(2**size, dtype=numpy.uint8)

    iterations = 0

    for iterator in range(0, size + 1):
        current_combinations = list(itertools.combinations(main_list, iterator))
        for combination in current_combinations:

            for index in combination:
                input_data[iterations, index] = 1
            
            result = linprog(c=numpy.ones(size),
                             A_ub=None,
                             b_ub=None,
                             A_eq=Q_matrix[combination, :],
                             b_eq=numpy.zeros(iterator),
                             bounds=(0.05, None))
            
            if result.success:
                output_data[iterations] = 1
            
            iterations = iterations + 1

    return input_data, output_data


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


def create_dataset_f(function_vector, input_data, output_data):
    function_vector_size = numpy.size(function_vector, axis=0)

    dimension = math.log2(function_vector_size)
    if not dimension.is_integer():
        print("Sizes of function vector is not power of two")
        return

    input_data_row_size = numpy.size(input_data, axis=0)
    input_data_column_size = numpy.size(input_data, axis=1)

    if input_data_column_size != function_vector_size:
        print("Input column and function vector size does not match.")
        return

    if input_data_row_size != numpy.size(output_data):
        print("Input row and output length does not match.")
        return

    f_flattened = numpy.reshape(function_vector, [1, function_vector_size])

    train_ds = numpy.zeros([input_data_row_size, function_vector_size + input_data_column_size], dtype=numpy.float32)

    for iterator in range(input_data_row_size):
        train_ds[iterator, :] = numpy.concatenate((f_flattened, numpy.reshape(input_data[iterator, :],
                                                                              [1, input_data_column_size])), axis=1)

    return train_ds, output_data


def create_and_save_dataset_spectrum(spectrum, input_file_name, output_file_name):
    function_list, spectrum_list = monsetup.get_functions_from_spectrum(spectrum)

    dimension = int(math.log2(numpy.size(spectrum)))

    function_list_size = len(function_list)
    combination_size = 2**(2**dimension)
    all_train_ds = numpy.zeros([combination_size*function_list_size, 2*(2**dimension)], dtype=numpy.uint8)
    all_output = numpy.zeros(combination_size*function_list_size, dtype=numpy.uint8)

    for counter in range(function_list_size):
        input_data, output_data = linprog_result(function_list[counter], dimension)
        train_ds, output_data = create_dataset_f(spectrum_list[counter], input_data, output_data)
        all_train_ds[counter*combination_size:(counter + 1)*combination_size, :] = train_ds
        all_output[counter*combination_size:(counter + 1)*combination_size] = output_data
        print("Function:", function_list[counter], " is finished.")

    numpy.save(input_file_name, all_train_ds)
    numpy.save(output_file_name, all_output)


def monomial_exclusion_linear_programming_nn_function(function, dimension):
    input_data, output_data = linprog_result(function,dimension)

    function = bf.booleanFunctionGenerator(function,dimension)
    train_ds, output_data = create_dataset_f(function, input_data, output_data)
    train_ds, output_data = create_dataset_f(function,input_data, output_data)

    Input_Layer_Size = 2*(2**dimension)
    First_Hidden_Layer_Size = 2**dimension
    #Second_Hidden_Layer_Size = 2**dimension
    Output_Layer_Size = 1

    model = Sequential()
    model.add( Dense(First_Hidden_Layer_Size, activation='relu', input_dim=Input_Layer_Size, name="First") )
    #model.add( Dense(Second_Hidden_Layer_Size, activation='relu', input_dim=First_Hidden_Layer_Size, name="Second") )
    model.add( Dense(Output_Layer_Size, activation='sigmoid', name="Output") )

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(train_ds, output_data, epochs=10, batch_size=32)
    return 0


def monomial_exclusion_linear_programming_nn( function,dimension ):
    Q_matrix = monsetup.qMatrixGenerator(function,dimension)
    input_data, output_data = linprog_result(function,dimension)
    train_ds, output_data = create_dataset(Q_matrix, input_data, output_data)

    Input_Layer_Size = (2**dimension)**2 + 2**dimension
    First_Hidden_Layer_Size = 2*dimension*(2**dimension)
    #Second_Hidden_Layer_Size = 2**dimension
    Output_Layer_Size = 1

    model = Sequential()
    model.add( Dense(First_Hidden_Layer_Size, activation='relu', input_dim=Input_Layer_Size, name="First") )
    #model.add( Dense(Second_Hidden_Layer_Size, activation='relu', input_dim=First_Hidden_Layer_Size, name="Second") )
    model.add( Dense(Output_Layer_Size, activation='sigmoid', name="Output") )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, output_data, epochs=10, batch_size=32)
    return 0


def normalize_predicted_output(predicted_output):
    predicted_output_size = numpy.size(predicted_output)
    normalized_predicted_output = numpy.zeros(predicted_output_size)

    for index in range(predicted_output_size):
        if predicted_output[index] >= 0.5:
            normalized_predicted_output[index] = 1

    return normalized_predicted_output




#linprog( c = numpy.ones(2**dimension), A_ub = None, b_ub = None, A_eq = Q_matrix[combination,:], b_eq = numpy.zeros(iterator), bounds = (0,None)  )