def function_to_hex_string(dimension, function):
    two_to_the_power_dimension = 2 ** dimension
    if function > 2**two_to_the_power_dimension:
        print("Invalid function for given dimension")
        return "0x0"

    number_of_bits_in_a_hexadecimal_digit = 4
    number_of_bits = int(two_to_the_power_dimension/number_of_bits_in_a_hexadecimal_digit)
    return "{0:#0{1}x}".format(function, number_of_bits + 2)
