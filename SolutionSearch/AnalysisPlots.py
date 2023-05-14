from Utils.PlotData import *
from SolutionSearch.SolutionSearch import *
from Utils.DumpOutputs import *
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import get_theoretical_number_of_zeroes
import numpy
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import *

from enum import Enum


class AnalysisPolicy(Enum):
    MAX = 0,
    AVERAGE = 1,
    THEORETICAL_COMPARE = 2


def compare_data(data):
    data_converted = numpy.array(data)

    num_compared = data_converted.shape[0]
    max_indexes_dict = {m: 0 for m in range(num_compared)}

    num_data = data_converted.shape[1]

    for counter in range(num_data):
        row = data_converted[:, counter]
        max_indexes = numpy.where(row == numpy.max(row))[0].tolist()
        for index in max_indexes:
            max_indexes_dict[index] += 1

    return max_indexes_dict, num_data


def add_compare_data_to_graph_label_impl(max_indexes_dict, num_data, graph_labels):
    keys = list(max_indexes_dict.keys())
    graph_labels_size = len(graph_labels)

    if len(keys) != graph_labels_size:
        print("compare_data_add_to_graph_label: Data inconsistency")
        return

    for key in keys:
        graph_labels[key] += "\n" + str(max_indexes_dict[key]) + "/" + str(num_data)


def add_compare_data_to_graph_label(data, graph_labels):
    max_indexes, num_data = compare_data(data)
    add_compare_data_to_graph_label_impl(max_indexes, num_data, graph_labels)


def x_label_font_size_rotation(dimension, number_of_data_points):
    font_power_coefficient = 1.4
    full_x_label_size = 36 * (11 ** font_power_coefficient) #digit * square(font_size)
    number_of_digits = 2 ** (dimension - 2)
    hex_overhead = 2

    total_number_of_digits = (number_of_digits + hex_overhead) * number_of_data_points

    f_size = (full_x_label_size / total_number_of_digits) ** (1/font_power_coefficient)

    if f_size <= 4:
        full_x_label_size_vertical = 3 * (6 ** 2)
        total_number_of_digits = number_of_digits + hex_overhead
        rotation = 90

        f_size = (full_x_label_size_vertical / total_number_of_digits) ** (1 / font_power_coefficient)
    else:
        rotation = 0

    if f_size > 10:
        f_size = 10

    return f_size, rotation


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
        graph_label = "AF:" + activation_function_string(parameters["simple_model"]) + "\n" +\
                      "LF:" + loss_function_string(parameters["loss_function"]) + "\n" +\
                      "RF:" + regularization_function_string(parameters["regularization_func"])
        graph_labels.append(graph_label)

    return graph_labels


def regularization_compare_bar(dimension, directories, analysis_policy=AnalysisPolicy.AVERAGE, compact=False,
                               output_directory=None, file_name_prefix=None):
    graph_labels = get_regularization_labels(directories)
    title = get_graph_title(analysis_policy, dimension, compact, "Regularization Methods'")
    compare_bar(dimension, directories, graph_labels, analysis_policy=analysis_policy, compact=compact, title=title,
                output_directory=output_directory, file_name_prefix=file_name_prefix)


def regularization_compare_2d(dimension, directories, analysis_policy=AnalysisPolicy.AVERAGE,
                              output_directory=None, file_name_prefix=None):
    graph_labels = get_regularization_labels(directories)
    title = get_graph_title(analysis_policy, dimension, False, "Regularization Methods'")
    compare_theoretical_2d(dimension, directories, graph_labels, analysis_policy, output_directory=output_directory,
                           title=title, file_name_prefix=file_name_prefix)


def density_order(dimension, equivalence_classes):
    densities = BooleanFunctionsEquivalentClassesDensity[dimension]

    densities_converted = numpy.ndarray([len(equivalence_classes), 2], dtype=numpy.uint64)
    counter = 0

    for equivalence_class in equivalence_classes:
        equivalence_class_converted = int(equivalence_class, 16)
        densities_converted[counter] = [equivalence_class_converted,
                                        2 ** dimension - densities[equivalence_class_converted]]
        counter += 1

    new_order = numpy.argsort(densities_converted[:, 1])[::-1]

    return new_order, densities_converted[new_order]


def compare_bar(dimension, directories, graph_labels, analysis_policy=AnalysisPolicy.AVERAGE, compact=False,
                title=None, output_directory=None, file_name_prefix=None, add_theoretical=False):
    y_data, y_data_std, graph_group_names = extract_data(dimension, directories, analysis_policy, compact)

    add_compare_data_to_graph_label(y_data, graph_labels)

    if add_theoretical:
        graph_labels.insert(0, "Theoretical\nlimit")
        y_data.insert(0, get_theoretical_number_of_zeroes(dimension,
                                                          [int(eq_class, 16) for eq_class in graph_group_names]))
        if y_data_std is not None:
            y_data_std.insert(0, [0] * len(y_data_std[0]))

    if title is None:
        title = get_graph_title(analysis_policy, dimension, compact)

    x_font_size, x_rotation = x_label_font_size_rotation(dimension, len(graph_group_names))

    plot_bar(y_data=y_data, graph_labels=graph_labels, graph_group_names=graph_group_names, y_data_std=y_data_std,
             y_label="Number of zeros", x_label="Equivalence classes", title=title, show=True,
             output_directory=output_directory, file_name_prefix=file_name_prefix, x_font_size=x_font_size,
             x_rotation=x_rotation)


def compare_theoretical_2d(dimension, directories, graph_labels, analysis_policy=AnalysisPolicy.AVERAGE,
                           title=None, output_directory=None, file_name_prefix=None,):
    if analysis_policy is AnalysisPolicy.THEORETICAL_COMPARE:
        print("AnalysisPolicy.THEORETICAL_COMPARE is not valid for this function")
        return

    y_data, y_data_std, x_data = extract_data(dimension, directories, analysis_policy, False)

    add_compare_data_to_graph_label(y_data, graph_labels)

    num_data_points = len(x_data)
    if title is None:
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

    if graph_labels is not None:
        graph_labels.insert(0, "Theoretical\nlimit")

    x_font_size, x_rotation = x_label_font_size_rotation(dimension, num_data_points)

    plot_2d(y_data=y_data, x_data=x_data, y_data_std=y_data_std, title=title,
            y_label="Number of zeros", x_label="Equivalence classes", graph_labels=graph_labels,
            show=True, output_directory=output_directory, file_name_prefix=file_name_prefix,
            x_font_size=x_font_size, x_rotation=x_rotation)
