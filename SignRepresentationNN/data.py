import torch

from SigmaPiFrameworkPython.BooleanFunctionGenerator import boolean_function_generator
from SigmaPiFrameworkPython.MonomialSetup import monomial_setup, q_matrix_generator


class SigmaPiModelDataSet(torch.utils.data.Dataset):
    def __init__(self, function, dimension):
        self.two_to_power_dimension = 2 ** dimension
        self.function_vector =\
            boolean_function_generator(function % 2 ** self.two_to_power_dimension, dimension)
        self.d_matrix = monomial_setup(dimension)
        self.q_matrix = q_matrix_generator(function, dimension, self.d_matrix)
        self.walsh_spectrum = self.q_matrix.sum(1)

    def __len__(self):
        return len(self.function_vector)

    def __getitem__(self, idx):
        return self.d_matrix[idx, :], self.function_vector[idx]
