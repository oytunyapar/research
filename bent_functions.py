import monsetup as ms
import numpy as np
import os
import pandas as pd


def all_same(items):
    return all(x == items[0] for x in items)


def find_bent_functions(dimension):
    if (dimension % 2) != 0:
        print("There is not any bent function in odd dimensions.\n")
        return

    number_of_variables = 2**dimension
    number_of_functions = 2**number_of_variables

    bent_function_list = []

    for function_iterator in range(number_of_functions+1):
        spectrum = abs(np.sum(ms.qMatrixGenerator(function_iterator, dimension), 1))
        if all_same(spectrum) and spectrum[0] > 0:
            bent_function_list.append(function_iterator)

    return np.array(bent_function_list)


def create_bent_data_set(dimension, data_folder="/home/oytun/Projects/research/SigmaPiFramework/dimension_4/spectrum/",
                         input_file_format="%d.input", output_file_format="%d.output"):

    bent_function_list = find_bent_functions(dimension)
    number_of_functions = np.size(bent_function_list)
    number_of_variables = 2**dimension
    number_of_rows = 2**number_of_variables
    number_of_columns = 32

    train_ds = np.zeros([number_of_functions*number_of_rows, number_of_columns], dtype=np.int8)
    output_ds = np.zeros([number_of_functions*number_of_rows], dtype=np.int8)

    input_file_name_template = data_folder + input_file_format
    output_file_name_template = data_folder + output_file_format

    for iterator in range(number_of_functions):
        input_file_name = input_file_name_template % bent_function_list[iterator]
        output_file_name = output_file_name_template % bent_function_list[iterator]

        if os.path.exists(input_file_name) and os.path.exists(output_file_name):
            begin_index = iterator*number_of_rows
            end_index = begin_index + number_of_rows
            train_ds[begin_index:end_index, :] = np.array(pd.read_csv(input_file_name, sep=" ", header=None),
                                                          dtype=np.int8)
            output_ds[begin_index:end_index] = \
                np.reshape(np.array(pd.read_csv(output_file_name, sep=" ", header=None), dtype=np.int8),
                           [number_of_rows])

        print("ITERATOR:%d\n" % iterator)

    return train_ds, output_ds
