import numpy


def binary_vector_to_combination(binary_vector):
    binary_vector_size = len(binary_vector)
    combination = numpy.array([], dtype=numpy.int)
    for index in range(0, binary_vector_size):
        if binary_vector[index] == 1:
            combination = numpy.append(combination, index)

    return combination


def binary_vector_to_int(binary_vector):
    power = 0
    result = 0

    for item in binary_vector:
        result += item * (2 ** power)
        power += 1

    return int(result)


def int_to_binary_vector(value, precision):
    binary_vector = numpy.zeros([precision])

    if value < 2 ** precision:
        for power in reversed(range(precision)):
            two_to_the_power = 2 ** power
            if value >= two_to_the_power:
                binary_vector[power] = 1
                value -= two_to_the_power

    return binary_vector


def get_eliminated_subsets(subsets, subset_elimination):
    return subsets[numpy.where(subset_elimination == 1), :][0]


def get_eliminated_subsets_size_dict(subsets, subset_elimination):
    eliminated_subsets = get_eliminated_subsets(subsets, subset_elimination)
    eliminated_subsets_size_dict = {}

    row_size = eliminated_subsets[0].size

    for subset_size in range(1, row_size):
        eliminated_subsets_size_dict[subset_size] = numpy.empty([0, row_size])

    for subset in eliminated_subsets:
        eliminated_subsets_size_dict[numpy.where(subset == 1)[0].size] = \
            numpy.append(eliminated_subsets_size_dict[numpy.where(subset == 1)[0].size], [subset], axis=0)

    for dic_key in list(eliminated_subsets_size_dict.keys()):
        if eliminated_subsets_size_dict[dic_key].size == 0:
            del eliminated_subsets_size_dict[dic_key]

    return eliminated_subsets_size_dict


def check_if_binary_vector_includes(superset, subset):
    subset_1_indices = numpy.where(subset == 1)[0]
    superset_1_indices = numpy.where(superset == 1)[0]
    includes = all(numpy.isin(subset_1_indices, superset_1_indices))

    if includes:
        return True, list(set(superset_1_indices) - set(subset_1_indices))
    else:
        return False, numpy.empty([0])


def check_superset_inclusion(eliminated_subsets_size_dict):
    keys = eliminated_subsets_size_dict.keys()
    result = {}
    for key in keys:
        if key + 1 in keys:
            supersets = eliminated_subsets_size_dict[key + 1]
            subsets = eliminated_subsets_size_dict[key]

            list_of_dicts = []

            for subset in subsets:
                current_key = binary_vector_to_int(subset)
                subset_superset_relation = {current_key: numpy.empty([0, 2], dtype=numpy.int32)}
                for superset in supersets:
                    includes, spare_index = check_if_binary_vector_includes(superset, subset)
                    if includes:
                        subset_superset_relation[current_key] =\
                            numpy.append(subset_superset_relation[current_key],
                                         numpy.array([[binary_vector_to_int(superset), spare_index[0]]]), axis=0)

                list_of_dicts.append(subset_superset_relation)
            result[key] = list_of_dicts
        print("check_superset_inclusion:" + str(key))

    return result


def check_the_elimination_dict_for_inclusion(elimination_dict):
    elimination_dict_keys = elimination_dict.keys()

    for subset_size in elimination_dict_keys:
        current_dimension_list = elimination_dict[subset_size]
        list_counter = 0
        for current_dimension_list_dict_item in current_dimension_list:
            current_dimension_list_dict_item_keys = current_dimension_list_dict_item.keys()
            for current_dimension_list_dict_item_key in current_dimension_list_dict_item_keys:
                if current_dimension_list_dict_item[current_dimension_list_dict_item_key].size == 0:
                    print("[" + str(subset_size) + "][" + str(list_counter) + "]->" +
                          str(current_dimension_list_dict_item))
                    return False
            list_counter += 1

    return True
