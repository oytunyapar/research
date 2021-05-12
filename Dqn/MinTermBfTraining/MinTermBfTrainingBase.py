import numpy
import monsetup
from ..Utilities import HelperFunctions
import signal
import math


class MinTermBfTrainingBase:
    def __init__(self, function, dimension, number_of_epochs, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)

        self.k_vector_size = self.two_to_power_dimension
        self.k_vector = numpy.ones(self.two_to_power_dimension)

        self.k_vector_check = numpy.ones(self.two_to_power_dimension)

        self.function_representation_size = self.two_to_power_dimension ** 2
        self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)

        self.state_size = self.function_representation_size + self.k_vector_size
        self.action_size = self.two_to_power_dimension + 1
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)

        self.current_state = numpy.ones([1, self.state_size])

        self.n_epoch_rl_steps = self.two_to_power_dimension ** 2
        self.n_total_rl_steps = number_of_epochs * self.n_epoch_rl_steps
        self.current_rl_step = 0
        self.number_of_epochs = number_of_epochs

        self.current_epoch = 0
        self.epoch_total_reward = 0
        self.reward_per_epoch = numpy.zeros([number_of_epochs])

        self.memory_zero_reward_size = self.n_total_rl_steps
        self.memory_zero_reward_index = 0

        self.memory_zero_reward = numpy.zeros([self.memory_zero_reward_size, 2 * self.state_size + self.action_size + 1],
                                              dtype=numpy.float32)

        self.memory_positive_reward_size = self.n_total_rl_steps
        self.memory_positive_reward_index = 0

        self.memory_positive_reward = numpy.zeros(
            [self.memory_positive_reward_size, 2 * self.state_size + self.action_size + 1],
            dtype=numpy.float32)

        self.replay_memory_size = self.two_to_power_dimension * 50

        self.positive_reward_bias_factor = 0.6

        self.batch_size = self.n_epoch_rl_steps

        self.discount_factor = HelperFunctions.create_number_with_precision(0, 9, self.dimension - 1)
        self.learning_rate = 0.1

        self.maximum_zeros_during_training = numpy.count_nonzero(self.walsh_spectrum == 0)
        self.maximum_zeros_k_vector = numpy.ones(self.two_to_power_dimension)

        self.training_function = None
        self.continue_training = True

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, signum, frame):
        print("User interrupt received.")
        self.continue_training = False

    def random_movement_possibility(self):
        offset = 3
        return numpy.tanh(offset - offset * (self.current_rl_step / self.n_total_rl_steps))

    def get_random_batch_from_memory(self, size):
        if self.memory_zero_reward_index + self.memory_positive_reward_index < size:
            print(
                "DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        positive_reward_memory_size = math.floor(size * self.positive_reward_bias_factor)

        if positive_reward_memory_size > self.memory_positive_reward_index:
            positive_reward_memory_size = self.memory_positive_reward_index

        remaining_positive_reward_memory_size = self.memory_positive_reward_index - positive_reward_memory_size

        zero_reward_memory_size = size - positive_reward_memory_size
        if zero_reward_memory_size > self.memory_zero_reward_index:
            zero_reward_memory_size = self.memory_zero_reward_index

        remaining_zero_reward_memory_size = self.memory_zero_reward_index - zero_reward_memory_size

        remaining_memory_size = size - (positive_reward_memory_size + zero_reward_memory_size)

        if remaining_memory_size > 0:
            if remaining_positive_reward_memory_size >= remaining_memory_size:
                remaining_positive_reward_memory_size -= remaining_memory_size
                positive_reward_memory_size += remaining_memory_size
                remaining_memory_size = 0
            else:
                remaining_positive_reward_memory_size = 0
                positive_reward_memory_size += remaining_positive_reward_memory_size
                remaining_memory_size -= remaining_positive_reward_memory_size

        if remaining_memory_size > 0:
            remaining_zero_reward_memory_size -= remaining_memory_size
            zero_reward_memory_size += remaining_memory_size
            remaining_memory_size = 0

        rng = numpy.random.default_rng()
        positive_reward_indices = rng.choice(self.memory_positive_reward_index,
                                             size=positive_reward_memory_size, replace=False)
        zero_reward_indices = rng.choice(self.memory_zero_reward_index, size=zero_reward_memory_size, replace=False)

        positive_reward_memory_partition = self.memory_positive_reward[positive_reward_indices.tolist(), :]
        zero_reward_memory_partition = self.memory_zero_reward[zero_reward_indices.tolist(), :]
        returned_memory = numpy.concatenate((positive_reward_memory_partition, zero_reward_memory_partition), axis=0)
        numpy.random.shuffle(returned_memory)
        return returned_memory

    def predicted_reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(numpy.matmul(self.q_matrix, next_k_vector) == 0)
        return next_number_of_zeros / self.two_to_power_dimension, next_number_of_zeros

    def save_memory(self, previous_state, next_state, action, reward):
        if reward > 0:
            if self.memory_positive_reward_index >= self.memory_positive_reward_size:
                print("ERROR: Positive reward memory is full.")
            else:
                self.memory_positive_reward[self.memory_positive_reward_index, 0: self.state_size] = previous_state.reshape(self.state_size)
                self.memory_positive_reward[self.memory_positive_reward_index, self.state_size: 2 * self.state_size] = next_state.reshape(self.state_size)
                self.memory_positive_reward[self.memory_positive_reward_index, 2 * self.state_size: 2 * self.state_size + self.action_size] = action
                self.memory_positive_reward[self.memory_positive_reward_index, 2 * self.state_size + self.action_size] = reward
                self.memory_positive_reward_index += 1
        else:
            if self.memory_zero_reward_index >= self.memory_zero_reward_size:
                print("ERROR: Zero reward memory is full.")
            else:
                self.memory_zero_reward[self.memory_zero_reward_index, 0: self.state_size] = previous_state.reshape(self.state_size)
                self.memory_zero_reward[self.memory_zero_reward_index, self.state_size: 2 * self.state_size] = next_state.reshape(self.state_size)
                self.memory_zero_reward[self.memory_zero_reward_index, 2 * self.state_size: 2 * self.state_size + self.action_size] = action
                self.memory_zero_reward[self.memory_zero_reward_index, 2 * self.state_size + self.action_size] = reward
                self.memory_zero_reward_index += 1

        return

    def train_agent(self):
        if self.n_total_rl_steps < self.replay_memory_size:
            print("Total RL steps is smaller than replay memory size.")
            return
        while self.current_rl_step < self.n_total_rl_steps and self.continue_training:
            print("EPOCH ", self.current_epoch, "/", self.number_of_epochs)
            self.reinforcement_learn_step(self.n_epoch_rl_steps)
            if self.current_rl_step >= self.replay_memory_size:
                memory_batch = self.get_random_batch_from_memory(self.replay_memory_size)
                self.training_function(memory_batch[:, 0: self.state_size],
                                       memory_batch[:, 2 * self.state_size: 2 * self.state_size + self.action_size])
            self.reward_per_epoch[self.current_epoch] = self.epoch_total_reward
            self.epoch_total_reward = 0
            self.k_vector = numpy.ones(self.two_to_power_dimension)
            self.current_epoch += 1
