import numpy
import monsetup
from ..Utilities import HelperFunctions


class MinTermBfTrainingBase:
    def __init__(self, function, dimension, total_rl_steps_factor, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)

        self.k_vector_size = self.two_to_power_dimension
        self.k_vector = numpy.ones(self.two_to_power_dimension)

        self.k_vector_check = numpy.ones(self.two_to_power_dimension)

        self.function_representation_size = self.two_to_power_dimension ** 2
        self.function_representation = self.q_matrix.reshape(1, self.function_representation_size, order='F')

        self.state_size = self.function_representation_size + self.k_vector_size
        self.action_size = self.two_to_power_dimension + 1
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)

        self.current_state = numpy.ones([1, self.state_size])

        self.n_epoch_rl_steps = self.two_to_power_dimension ** 2
        self.n_total_rl_steps = total_rl_steps_factor * self.n_epoch_rl_steps
        self.current_rl_step = 0

        self.short_memory_size = self.n_epoch_rl_steps
        self.short_memory_index = 0

        self.short_memory = numpy.zeros([self.short_memory_size, 2 * self.state_size + self.action_size + 1],
                                        dtype=numpy.float32)

        self.long_memory_size = self.n_total_rl_steps
        self.long_memory_index = 0

        self.long_memory = numpy.zeros([self.long_memory_size, 2 * self.state_size + self.action_size + 1],
                                       dtype=numpy.float32)

        self.batch_size = self.n_epoch_rl_steps

        self.discount_factor = HelperFunctions.create_number_with_precision(0, 9, self.dimension - 1)
        self.learning_rate = 0.1

        self.maximum_zeros_during_training = numpy.count_nonzero(self.walsh_spectrum == 0)
        self.maximum_zeros_k_vector = numpy.ones(self.two_to_power_dimension)

        self.training_function = None

    def random_movement_possibility(self):
        offset = 3
        return numpy.tanh(offset - offset * (self.current_rl_step / self.n_total_rl_steps))

    def get_random_batch_from_memory(self, size):
        if self.long_memory_index < size:
            print(
                "DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        rng = numpy.random.default_rng()
        numbers = rng.choice(self.long_memory_index, size=size, replace=False)
        return self.long_memory[numbers.tolist(), :]

    def predicted_reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(numpy.matmul(self.q_matrix, next_k_vector)== 0)
        return next_number_of_zeros / self.two_to_power_dimension, next_number_of_zeros

    def save_memory(self, previous_state, next_state, action, reward):
        self.short_memory[self.short_memory_index, 0: self.state_size] = previous_state.reshape(self.state_size)
        self.short_memory[self.short_memory_index, self.state_size: 2 * self.state_size] = next_state.reshape(self.state_size)
        self.short_memory[self.short_memory_index, 2 * self.state_size: 2 * self.state_size + self.action_size] = action
        self.short_memory[self.short_memory_index, 2 * self.state_size + self.action_size] = reward
        self.short_memory_index += 1
        if self.short_memory_index >= self.short_memory_size:
            self.short_memory_index = 0
            if self.long_memory_index + self.short_memory_size <= self.long_memory_size:
                self.long_memory[
                self.long_memory_index:self.long_memory_index + self.short_memory_size, :
                ] = self.short_memory
                self.long_memory_index += self.short_memory_size
        return

    def train_agent(self):
        while self.current_rl_step < self.n_total_rl_steps:
            self.reinforcement_learn_step(self.n_epoch_rl_steps)
            memory_batch = self.get_random_batch_from_memory(self.batch_size)
            self.training_function(memory_batch[:, 0: self.state_size],
                                   memory_batch[:, 2 * self.state_size: 2 * self.state_size + self.action_size])
            self.k_vector = numpy.ones(self.two_to_power_dimension)
