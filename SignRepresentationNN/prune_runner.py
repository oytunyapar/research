from SignRepresentationNN.prune import *
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *
from SigmaPiFrameworkPython.Utils.DataStructureUtils import *
import datetime
import time


def prune_runner(number_of_runs=1):
    data = {}

    for dimension in BooleanFunctionsEquivalentClasses.keys():
        data[dimension] = {}
        num_functions = len(BooleanFunctionsEquivalentClasses[dimension])
        function_counter = 0
        for function in BooleanFunctionsEquivalentClasses[dimension]:
            data[dimension][function] = []
            function_counter += 1
            time_in_seconds = time.time()
            for run in range(number_of_runs):
                learning = PruneSigmaPiModel(function, dimension, False)
                if learning.operation(0.05):
                    data[dimension][function].append(len(learning.zeroed_weights()))
            print("Dimension:", dimension, " Function:", function_counter, "/", num_functions,
                  " Elapsed time:", time.time() - time_in_seconds)

    dir_name = "/home/oytun/PycharmProjects/research/Data/prune_runner"
    save_data_structure(dir_name, "/" + str(datetime.datetime.now()), data)

    return data

