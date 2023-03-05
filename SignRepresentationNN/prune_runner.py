from SignRepresentationNN.prune import *
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *
from Utils.DumpOutputs import dump_json, dump_csv
import datetime
import time
import numpy


def prune_runner(number_of_runs=1, output_dir="/home/oytun/PycharmProjects/research/Data/prune_runner/"):
    data = {}
    loss_function = LossFunction.RELU
    regularization_function = RegularizationFunction.HOYER_SQUARE
    regularization_strength = 0.05
    simple_model = True

    for dimension in BooleanFunctionsEquivalentClasses.keys():
        data[dimension] = {}
        num_functions = len(BooleanFunctionsEquivalentClasses[dimension])
        function_counter = 0
        for function in BooleanFunctionsEquivalentClasses[dimension]:
            data[dimension][function] = []
            function_counter += 1
            time_in_seconds = time.time()
            for run in range(number_of_runs):
                learning = PruneSigmaPiModel(function, dimension, regularization_strength, simple_model, loss_function,
                                             regularization_function)
                if learning.operation():
                    data[dimension][function].append(len(learning.zeroed_weights()))
            print("Dimension:", dimension, " Function:", function_counter, "/", num_functions,
                  " Elapsed time:", time.time() - time_in_seconds)

    learning = PruneSigmaPiModel(4, 4, regularization_strength, simple_model, loss_function, regularization_function)
    dir_name = output_dir + str(datetime.datetime.now())
    save_data_structure(dir_name, "dictionary", data)
    dump_json(learning.parameters(), dir_name, "parameters")

    max_data = max_prune_runner(data=data)
    for dimension in max_data.keys():
        rows = max_data_dict_to_rows(max_data[dimension])
        dump_csv(["function", "value"], rows, dir_name, "dimension_" + str(dimension))

    return data


def max_prune_runner(directory=None, file=None, data=None):
    if data is None:
        data = open_data_structure(directory, file)

    max_data = {}
    for dimension in data.keys():
        current_functions = data[dimension]
        max_data[dimension] = {}
        for function in current_functions.keys():
            values = current_functions[function]
            if len(values) > 0:
                max_data[dimension][hex(function)] = numpy.max(values)
            else:
                max_data[dimension][hex(function)] = None

    return max_data


def max_data_dict_to_rows(max_data_for_a_dimension):
    rows = []
    for key, value in max_data_for_a_dimension.items():
        if value is None:
            value = "-"
        rows.append([key, value])

    return rows
