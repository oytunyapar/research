import numpy
import monsetup
from ..Utilities import HelperFunctions


class MinTermBfTrainingBase:
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)

        model_layer_sizes.insert(0, self.two_to_power_dimension)
        model_layer_sizes.append(self.two_to_power_dimension)

        self.memory = numpy.zeros([n_total_rl_steps, 3 * self.two_to_power_dimension + 1], dtype=numpy.float32)

        self.n_total_rl_steps = n_total_rl_steps
        self.n_epoch_rl_steps = n_epoch_rl_steps
        self.current_rl_step = 0

        self.memory_index = 0

        self.batch_size = batch_size
        self.k_vector = numpy.ones([self.two_to_power_dimension, 1], dtype=numpy.float32)

        self.discount_factor = HelperFunctions.create_number_with_precision(0, 9, self.dimension - 1)
        self.learning_rate = 0.1

        self.maximum_zeros_during_training = 0

        self.training_function = None

    def random_movement_possibility(self):
        offset = 3
        return numpy.tanh(offset - offset*(self.current_rl_step/self.n_total_rl_steps))

    def get_random_batch_from_memory(self, size):
        if self.memory_index < size:
            print("DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        rng = numpy.random.default_rng()
        numbers = rng.choice(self.memory_index, size=size, replace=False)
        return self.memory[numbers.tolist(), :]

    def predicted_reward(self, next_state):
        next_number_of_zeros = numpy.count_nonzero(next_state == 0)
        number_of_zeros = numpy.count_nonzero(self.current_state == 0)
        return (next_number_of_zeros - number_of_zeros) / self.two_to_power_dimension, next_number_of_zeros

    def save_memory(self, previous_state, next_state, action, reward):
        self.memory[self.current_rl_step, 0: self.two_to_power_dimension] = previous_state
        self.memory[self.current_rl_step, self.two_to_power_dimension: 2 * self.two_to_power_dimension] = next_state
        self.memory[self.current_rl_step, 2 * self.two_to_power_dimension: 3 * self.two_to_power_dimension] = action
        self.memory[self.current_rl_step, 3 * self.two_to_power_dimension] = reward
        self.memory_index += 1
        return

    def train_agent(self):
        while self.current_rl_step < self.n_total_rl_steps:
            self.reinforcement_learn_step(self.n_epoch_rl_steps)
            memory_batch = self.get_random_batch_from_memory(self.batch_size)
            self.training_function(memory_batch[:, 0: self.two_to_power_dimension],
                                   memory_batch[:, 2 * self.two_to_power_dimension: 3 * self.two_to_power_dimension])
