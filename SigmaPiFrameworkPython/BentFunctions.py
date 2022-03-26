from SigmaPiFrameworkPython.MonomialSetup import q_matrix_generator
import numpy as np
import SigmaPiFrameworkPython.MonomialExclusionLinearProgrammingNN as mnn


def all_same(items):
    return all(x == items[0] for x in items)


def find_bent_functions(dimension, requested_number_of_functions=None):
    if (dimension % 2) != 0:
        print("There is not any bent function in odd dimensions.\n")
        return

    number_of_variables = 2**dimension
    number_of_functions = 2**number_of_variables

    bent_function_list = []

    for function_iterator in range(number_of_functions):
        spectrum = abs(np.sum(q_matrix_generator(function_iterator, dimension), 1))
        if all_same(spectrum) and spectrum[0] > 0:
            bent_function_list.append(function_iterator)

        if requested_number_of_functions is not None and requested_number_of_functions == len(bent_function_list):
            break

    return np.array(bent_function_list)


def create_bent_data_set(dimension, data_folder="/home/oytun/Projects/research/SigmaPiFrameworkMatlab/dimension_4/spectrum/",
                         input_file_format="%d.input", output_file_format="%d.output"):

    bent_function_list = find_bent_functions(dimension)
    return mnn.read_data_set(bent_function_list, dimension, data_folder, input_file_format, output_file_format)