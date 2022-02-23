import SigmaPiFrameworkPython.monomial_setup as monomial_setup
from SigmaPiFrameworkPython.boolean_function_generator import boolean_function_generator
from SigmaPiFrameworkPython.sigma_pi_linear_programming import monomial_exclusion_all_dimension
from SigmaPiFrameworkPython.Utils.q_matrix_utils import q_matrix_to_dimension
from SigmaPiFrameworkPython.Utils.boolean_function_utils import bf_to_dimension, get_functions_from_walsh_spectrum

import os
import pandas

import math
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_dataset(q_matrix, input_data, output_data):
    dimension, q_matrix_row_size = q_matrix_to_dimension(q_matrix)

    input_data_row_size = numpy.size(input_data, axis=0)
    input_data_column_size = numpy.size(input_data, axis=1)

    if input_data_column_size != q_matrix_row_size:
        print("Input column and Q matrix row sizes does not match.")
        return

    if input_data_row_size != numpy.size(output_data):
        print("Input row and output length does not match.")
        return
    
    q_flattened = numpy.reshape(q_matrix, [1, q_matrix_row_size * q_matrix_row_size])

    train_ds = numpy.zeros([input_data_row_size, q_matrix_row_size * q_matrix_row_size + input_data_column_size],
                           dtype=numpy.float32)

    for iterator in range(input_data_row_size):
        train_ds[iterator, :] = numpy.concatenate((q_flattened,
                                                   numpy.reshape(input_data[iterator, :],
                                                                 [1, input_data_column_size])), axis=1)
    
    return train_ds, output_data


def create_dataset_f(function_vector, input_data, output_data):
    dimension, function_vector_size = bf_to_dimension(function_vector)

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
    function_list, spectrum_list = get_functions_from_walsh_spectrum(spectrum)

    dimension = int(math.log2(numpy.size(spectrum)))

    function_list_size = len(function_list)
    combination_size = 2**(2**dimension)
    all_train_ds = numpy.zeros([combination_size*function_list_size, 2*(2**dimension)], dtype=numpy.uint8)
    all_output = numpy.zeros(combination_size*function_list_size, dtype=numpy.uint8)

    for counter in range(function_list_size):
        input_data, output_data = monomial_exclusion_all_dimension(function_list[counter], dimension)
        train_ds, output_data = create_dataset_f(spectrum_list[counter], input_data, output_data)
        all_train_ds[counter*combination_size:(counter + 1)*combination_size, :] = train_ds
        all_output[counter*combination_size:(counter + 1)*combination_size] = output_data
        print("Function:", function_list[counter], " is finished.")

    numpy.save(input_file_name, all_train_ds)
    numpy.save(output_file_name, all_output)


def read_data_set(functions, dimension,
                  data_folder="/home/oytun/Projects/research/SigmaPiFrameworkMatlab/dimension_4/spectrum/",
                  input_file_format="%d.input", output_file_format="%d.output"):
    number_of_functions = numpy.size(functions)
    number_of_variables = 2**dimension
    number_of_rows = 2**number_of_variables
    number_of_columns = 32

    train_ds = numpy.zeros([number_of_functions*number_of_rows, number_of_columns], dtype=numpy.int8)
    output_ds = numpy.zeros([number_of_functions*number_of_rows], dtype=numpy.int8)

    input_file_name_template = data_folder + input_file_format
    output_file_name_template = data_folder + output_file_format

    for iterator in range(number_of_functions):
        input_file_name = input_file_name_template % functions[iterator]
        output_file_name = output_file_name_template % functions[iterator]

        if os.path.exists(input_file_name) and os.path.exists(output_file_name):
            begin_index = iterator*number_of_rows
            end_index = begin_index + number_of_rows
            train_ds[begin_index:end_index, :] = numpy.array(pandas.read_csv(input_file_name, sep=" ", header=None),
                                                             dtype=numpy.int8)
            output_ds[begin_index:end_index] = \
                numpy.reshape(numpy.array(pandas.read_csv(output_file_name, sep=" ", header=None), dtype=numpy.int8),
                           [number_of_rows])

        print("FUNCTION[%d]:%d\n" % (iterator, functions[iterator]))

    return train_ds, output_ds


def monomial_exclusion_linear_programming_nn_function(function, dimension):
    input_data, output_data = monomial_exclusion_all_dimension(function, dimension)

    function = boolean_function_generator(function, dimension)
    train_ds, output_data = create_dataset_f(function, input_data, output_data)
    train_ds, output_data = create_dataset_f(function, input_data, output_data)

    input_layer_size = 2*(2**dimension)
    first_hidden_layer_size = 2**dimension
    #second_hidden_layer_size = 2**dimension
    output_layer_size = 1

    model = Sequential()
    model.add(Dense(first_hidden_layer_size, activation='relu', input_dim=input_layer_size, name="First"))
    #model.add(Dense(second_hidden_layer_size, activation='relu', input_dim=first_hidden_layer_size, name="Second"))
    model.add(Dense(output_layer_size, activation='sigmoid', name="Output"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, output_data, epochs=10, batch_size=32)
    return 0


def monomial_exclusion_linear_programming_nn(function, dimension):
    q_matrix = monomial_setup.q_matrix_generator(function, dimension)
    input_data, output_data = monomial_exclusion_all_dimension(function, dimension)
    train_ds, output_data = create_dataset(q_matrix, input_data, output_data)

    input_layer_size = (2**dimension)**2 + 2**dimension
    first_hidden_layer_size = 2*dimension*(2**dimension)
    #second_hidden_layer_size = 2**dimension
    output_layer_size = 1

    model = Sequential()
    model.add(Dense(first_hidden_layer_size, activation='relu', input_dim=input_layer_size, name="First"))
    #model.add(Dense(second_hidden_layer_size, activation='relu', input_dim=first_hidden_layer_size, name="Second"))
    model.add(Dense(output_layer_size, activation='sigmoid', name="Output"))

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
