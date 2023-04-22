from Utils.PlotData import *
from SolutionSearch.SolutionSearch import *
from Utils.DumpOutputs import *
import numpy
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *

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


def extract_data(dimension, directories, analysis_policy=AnalysisPolicy.AVERAGE, compact=False):
    graph_group_names = []
    y_data = []

    if analysis_policy is AnalysisPolicy.AVERAGE and compact is False:
        y_data_std = []
    else:
        y_data_std = None

    file_name = analysis_policy_file_prefix(analysis_policy, dimension)

    for directory in directories:
        data = numpy.array(load_csv(directory, file_name))

        if len(graph_group_names) == 0:
            graph_group_names.extend(data[:, 0].tolist())
        elif graph_group_names != data[:, 0].tolist():
            raise Exception("graph_group_names is not same")

        if compact is True:
            casted_array = [float(number) for number in data[:, 1].tolist()]
            y_data.append([sum(casted_array)])
        else:
            y_data.append([float(number) for number in data[:, 1].tolist()])

            if y_data_std is not None:
                y_data_std.append([float(number) for number in data[:, 2].tolist()])

    if compact is True:
        graph_group_names = ["All equivalence classes"]

    return y_data, y_data_std, graph_group_names


def get_graph_title(analysis_policy, dimension, compact, prefix=None):
    string = "Dim " + str(dimension)

    if compact is True:
        string += " Compact"

    if analysis_policy is AnalysisPolicy.MAX:
        string += " Max"
    elif analysis_policy is AnalysisPolicy.AVERAGE:
        string += " Average"
    elif analysis_policy is AnalysisPolicy.THEORETICAL_COMPARE:
        string += " Theoretical Compare"

    if prefix is not None:
        string = prefix + " " + string

    return string


def get_regularization_labels(directories):
    graph_labels = []
    for directory in directories:
        parameters = load_json(directory, regularization_parameters_file_name())
        graph_label = "LF:" + loss_function_string(parameters["loss_function"]) + "\n" +\
                      "RF:" + regularization_function_string(parameters["regularization_func"]) + "\n" +\
                      "AF:" + activation_function_string(parameters["simple_model"])
        graph_labels.append(graph_label)

    return graph_labels


def regularization_compare(dimension, directories, analysis_policy=AnalysisPolicy.AVERAGE, compact=False):
    y_data, y_data_std, graph_group_names = extract_data(dimension, directories, analysis_policy, compact)
    title = get_graph_title(analysis_policy, dimension, compact, "Regularization Methods'")

    graph_labels = get_regularization_labels(directories)

    plot_bar(y_data=y_data, graph_labels=graph_labels, graph_group_names=graph_group_names, y_data_std=y_data_std,
             y_label="Number of zeros", x_label="Equivalence classes", title=title, show=True)


def density_order(dimension, equivalence_classes):
    densities = BooleanFunctionsEquivalentClassesDensity[dimension]

    densities_converted = numpy.ndarray([len(equivalence_classes), 2], dtype=int)
    counter = 0

    for equivalence_class in equivalence_classes:
        equivalence_class_converted = int(equivalence_class, 16)
        densities_converted[counter] = [equivalence_class_converted,
                                        2 ** dimension - densities[equivalence_class_converted]]
        counter += 1

    new_order = numpy.argsort(densities_converted[:, 1])[::-1]

    return new_order, densities_converted[new_order]


def compare_theoretical_2d(dimension, directories, graph_labels, analysis_policy=AnalysisPolicy.AVERAGE):
    if analysis_policy is AnalysisPolicy.THEORETICAL_COMPARE:
        print("AnalysisPolicy.THEORETICAL_COMPARE is not valid for this function")
        return

    y_data, y_data_std, x_data = extract_data(dimension, directories, analysis_policy, False)
    title = get_graph_title(analysis_policy, dimension, False)

    new_order, densities_converted = density_order(dimension, x_data)

    x_data = numpy.array(x_data)[new_order].tolist()

    number_of_data = len(y_data)

    for counter in range(number_of_data):
        y_data[counter] = numpy.array(y_data[counter])[new_order].tolist()
        if y_data_std is not None:
            y_data_std[counter] = numpy.array(y_data_std[counter])[new_order].tolist()

    y_data.insert(0, densities_converted[:, 1].tolist())
    x_data = [x_data] * len(y_data)

    if y_data_std is not None:
        y_data_std.insert(0, [0]*len(y_data_std[0]))

    plot_2d(y_data=y_data, x_data=x_data, y_data_std=y_data_std, title=title,
            y_label="Number of zeros", x_label="Equivalence classes", graph_labels=graph_labels,
            show=True)
