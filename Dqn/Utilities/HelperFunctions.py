import math


def softmax_internal(values):
    values_length = len(values)

    for index in range(values_length):
        values[index] = pow(math.e, values[index])

    sum_of_values = sum(values)

    for index in range(values_length):
        values[index] /= sum_of_values

    return values


def make_negatives_zero(values):
    values_length = len(values)

    for index in range(values_length):
        if values[index] < 0:
            values[index] = 0

    return values


def create_number_with_precision(before_dot, after_dot, repeat_count_after_dot):
    decimal_part = 0
    for step in range(1, repeat_count_after_dot + 1):
        decimal_part += after_dot * pow(10, -step)
    return before_dot + decimal_part
