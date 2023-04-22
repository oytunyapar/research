from Utils.PlotData import *
from SolutionSearch.SolutionSearch import *
from Utils.DumpOutputs import *
import numpy

from enum import Enum


class AnalysisPolicy(Enum):
    MAX = 0,
    AVERAGE = 1,
    THEORETICAL_COMPARE = 2


def analysis_policy_file_prefix(analysis_policy, dimension):
    if analysis_policy is AnalysisPolicy.MAX:
        string = max_csv_file_name(dimension)
    elif analysis_policy is AnalysisPolicy.AVERAGE:
        string = average_csv_file_name(dimension)
    elif analysis_policy is AnalysisPolicy.THEORETICAL_COMPARE:
        string = theoretical_compare_csv_file_name(dimension)
    else:
        raise Exception("Unknown analysis policy.")

    return string


def regularization_compare(dimension, directories, analysis_policy=AnalysisPolicy.AVERAGE, compact=False):
    graph_labels = []
    graph_group_names = []
    y_data = []
    y_data_std = None

    file_name = analysis_policy_file_prefix(analysis_policy, dimension)

    for directory in directories:
        parameters = load_json(directory, regularization_parameters_file_name())
        graph_label = "LF:" + loss_function_string(parameters["loss_function"]) + "\n" +\
                      "RF:" + regularization_function_string(parameters["regularization_func"]) + "\n" +\
                      "AF:" + activation_function_string(parameters["simple_model"])
        graph_labels.append(graph_label)

        data = numpy.array(load_csv(directory, file_name))

        if len(graph_group_names) == 0:
            graph_group_names = data[:, 0].tolist()
        elif graph_group_names != data[:, 0].tolist():
            raise Exception("graph_group_names is not same")

        y_data.append([float(number) for number in data[:, 1].tolist()])

        if analysis_policy is AnalysisPolicy.AVERAGE:
            if y_data_std is None:
                y_data_std = []
            y_data_std.append([float(number) for number in data[:, 2].tolist()])

    plot_bar(y_data, graph_labels, graph_group_names, y_data_std=y_data_std, show=True)
