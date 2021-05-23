import numpy
import monsetup
import signal
import math


class MinTermBfTrainingBase:
    def __init__(self, function, dimension, number_of_epochs, model_layer_sizes):
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension

        self.number_of_all_functions = 2 ** self.two_to_power_dimension

        self.function = function

        self.all_function_training = False
        if self.function == 0:
            self.all_function_training = True
        elif self.function >= self.number_of_all_functions:
            raise Exception("Function must be smaller than " + str(self.number_of_all_functions))

        if self.all_function_training:
            self.function = numpy.random.randint(low=1, high=self.number_of_all_functions)
            self.training_each_episode = 2 ** self.two_to_power_dimension
        else:
            self.training_each_episode = self.two_to_power_dimension

        self.q_matrix = monsetup.q_matrix_generator(function, self.dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)
        #self.function_representation_size = self.two_to_power_dimension ** 2
        #self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)

        self.function_representation_size = self.two_to_power_dimension
        self.function_representation = self.walsh_spectrum

        self.k_vector_size = self.two_to_power_dimension
        self.k_vector = numpy.ones(self.two_to_power_dimension)

        self.k_vector_check = numpy.ones(self.two_to_power_dimension)

        self.state_size = self.function_representation_size + self.k_vector_size
        self.action_size = self.two_to_power_dimension + 1
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)

        self.current_state = numpy.ones([1, self.state_size])

        self.n_epoch_rl_steps = self.two_to_power_dimension ** 2
        self.n_total_rl_steps = number_of_epochs * self.n_epoch_rl_steps
        self.current_rl_step = 0
        self.number_of_epochs = number_of_epochs

        self.replay_memory_size = (self.n_epoch_rl_steps * self.two_to_power_dimension)*4
        self.training_factor = math.floor(self.training_each_episode/4)

        self.current_epoch = 0
        self.epoch_total_reward = 0
        self.reward_per_epoch = numpy.zeros([number_of_epochs])

        self.memory_row_length = 2 * self.state_size + self.action_size + 1

        self.pile_memory = False

        if self.pile_memory:
            self.pile_memory_zero_reward_size = self.n_total_rl_steps
            self.pile_memory_zero_reward_index = 0

            self.pile_memory_zero_reward = numpy.zeros(
                [self.pile_memory_zero_reward_size, self.memory_row_length],
                dtype=numpy.float32)

            self.pile_memory_positive_reward_size = self.n_total_rl_steps
            self.pile_memory_positive_reward_index = 0

            self.pile_memory_positive_reward = numpy.zeros(
                [self.pile_memory_positive_reward_size, self.memory_row_length],
                dtype=numpy.float32)
        else:
            self.zero_reward_memory = numpy.zeros([0, self.memory_row_length])
            self.zero_reward_memory_all_functions = {}

            self.priority_reward_memory = {}
            self.priority_reward_memory_all_functions = {}

        self.positive_reward_bias_factor = 0.5

        self.batch_size = self.n_epoch_rl_steps

        self.discount_factor = 0.99
        self.learning_rate = 0.6

        self.maximum_zeros_during_training = numpy.zeros(self.number_of_all_functions)
        self.maximum_zeros_during_training[self.function] = numpy.count_nonzero(self.walsh_spectrum == 0)
        self.maximum_zeros_k_vector = numpy.ones([self.number_of_all_functions, self.two_to_power_dimension])

        self.training_function = None
        self.continue_training = True

        signal.signal(signal.SIGINT, self.sigint_handler)

    def set_function(self, function):
        if function == 0 or function >= self.number_of_all_functions:
            print("Function must be between 0 and ", str(self.number_of_all_functions))
            return

        self.function = function
        self.q_matrix = monsetup.q_matrix_generator(function, self.dimension)
        self.walsh_spectrum = self.q_matrix.sum(1)

        number_of_zeros = numpy.count_nonzero(self.walsh_spectrum == 0)
        if number_of_zeros > self.maximum_zeros_during_training[self.function]:
            self.maximum_zeros_during_training[self.function] = number_of_zeros
            self.maximum_zeros_k_vector[self.function, :] = numpy.ones(self.two_to_power_dimension)

        if not self.pile_memory:
            if self.function in self.zero_reward_memory_all_functions:
                self.zero_reward_memory = self.zero_reward_memory_all_functions[self.function]
            else:
                self.zero_reward_memory = numpy.zeros([0, self.memory_row_length])

            if self.function in self.priority_reward_memory_all_functions:
                self.priority_reward_memory = self.priority_reward_memory_all_functions[self.function]
            else:
                self.priority_reward_memory = {}

        #self.function_representation = self.q_matrix.reshape(1, self.function_representation_size)
        self.function_representation = self.walsh_spectrum

    def sigint_handler(self, signum, frame):
        print("User interrupt received.")
        self.continue_training = False

    def random_movement_possibility(self):
        offset = 2
        return numpy.tanh(offset - offset * (self.current_rl_step / self.n_total_rl_steps))

    def get_random_batch_from_memory(self, size):
        if self.pile_memory:
            total_positive_reward_memory_size = self.pile_memory_positive_reward_index
            total_zero_reward_memory_size = self.pile_memory_zero_reward_index
        else:
            total_positive_reward_memory_size = self.prioritized_reward_memory_size()
            total_zero_reward_memory_size = self.zero_reward_memory.shape[0]

        if total_zero_reward_memory_size + total_positive_reward_memory_size < size:
            print(
                "DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros([0, self.memory_row_length])

        positive_reward_memory_size = math.floor(size * self.positive_reward_bias_factor)

        if positive_reward_memory_size > total_positive_reward_memory_size:
            positive_reward_memory_size = total_positive_reward_memory_size

        remaining_positive_reward_memory_size = total_positive_reward_memory_size - positive_reward_memory_size

        zero_reward_memory_size = size - positive_reward_memory_size
        if zero_reward_memory_size > total_zero_reward_memory_size:
            zero_reward_memory_size = total_zero_reward_memory_size

        remaining_zero_reward_memory_size = total_zero_reward_memory_size - zero_reward_memory_size

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

        zero_reward_memory_partition = self.construct_zero_reward_batch(zero_reward_memory_size)

        if positive_reward_memory_size != 0:
            positive_reward_memory_partition = self.construct_positive_reward_batch(positive_reward_memory_size)
            returned_memory = numpy.concatenate((positive_reward_memory_partition, zero_reward_memory_partition), axis=0)
            numpy.random.shuffle(returned_memory)
        else:
            returned_memory = zero_reward_memory_partition

        return returned_memory

    def predicted_reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(numpy.matmul(self.q_matrix, next_k_vector) == 0)
        #/ self.two_to_power_dimension
        return next_number_of_zeros, next_number_of_zeros

    def save_memory(self, previous_state, next_state, action, reward):
        memory_entry = numpy.zeros([1, self.memory_row_length])
        memory_entry[0, 0: self.state_size] = previous_state.reshape(self.state_size)
        memory_entry[0, self.state_size: 2 * self.state_size] = next_state.reshape(self.state_size)
        memory_entry[0, 2 * self.state_size: 2 * self.state_size + self.action_size] = action
        memory_entry[0, 2 * self.state_size + self.action_size] = reward

        if reward > 0:
            if self.pile_memory:
                if self.pile_memory_positive_reward_index >= self.pile_memory_positive_reward_size:
                    print("ERROR: Positive reward memory is full.")
                else:
                    self.pile_memory_positive_reward[self.pile_memory_positive_reward_index] = memory_entry

                    self.pile_memory_positive_reward_index += 1
            else:
                if self.function in self.priority_reward_memory_all_functions:
                    self.priority_reward_memory = self.priority_reward_memory_all_functions[self.function]
                else:
                    self.priority_reward_memory = {}

                if reward in self.priority_reward_memory:
                    self.priority_reward_memory[reward] = \
                        numpy.concatenate((self.priority_reward_memory[reward], memory_entry), axis=0)
                else:
                    self.priority_reward_memory[reward] = memory_entry.copy()

                self.priority_reward_memory_all_functions[self.function] = self.priority_reward_memory
        else:
            if self.pile_memory:
                if self.pile_memory_zero_reward_index >= self.pile_memory_zero_reward_size:
                    print("ERROR: Zero reward memory is full.")
                else:
                    self.pile_memory_zero_reward[self.pile_memory_zero_reward_index] = memory_entry

                    self.pile_memory_zero_reward_index += 1
            else:
                if self.function in self.zero_reward_memory_all_functions:
                    self.zero_reward_memory_all_functions[self.function] = \
                        numpy.concatenate((self.zero_reward_memory_all_functions[self.function], memory_entry), axis=0)
                else:
                    self.zero_reward_memory_all_functions[self.function] = memory_entry.copy()

                self.zero_reward_memory = self.zero_reward_memory_all_functions[self.function]

        return

    def construct_zero_reward_batch(self, size):
        if size == 0:
            print("construct_zero_reward_batch: size is 0.")
            return numpy.zeros([0, self.memory_row_length])
        if self.pile_memory:
            rng = numpy.random.default_rng()
            zero_reward_indices = rng.choice(self.pile_memory_zero_reward_index, size=size, replace=False)
            zero_reward_memory_partition = self.pile_memory_zero_reward[zero_reward_indices.tolist(), :]

            return zero_reward_memory_partition
        else:
            if self.function not in self.zero_reward_memory_all_functions:
                print("construct_zero_reward_batch: function positive reward does not exists.")
                return numpy.zeros([0, self.memory_row_length])

            self.zero_reward_memory = self.zero_reward_memory_all_functions[self.function]
            zero_reward_memory_size = self.zero_reward_memory.shape[0]

            if size > zero_reward_memory_size:
                print("construct_zero_reward_batch: size is bigger than total size")
                return numpy.zeros([0, self.memory_row_length])

            rng = numpy.random.default_rng()
            zero_reward_indices = rng.choice(zero_reward_memory_size, size=size, replace=False)

            zero_reward_memory_partition = self.zero_reward_memory[zero_reward_indices.tolist(), :]

            return zero_reward_memory_partition

    def prioritized_reward_memory_size(self):
        if self.pile_memory:
            return 0

        priority_reward_memory_size = 0
        for key in self.priority_reward_memory:
            priority_reward_memory_size += len(self.priority_reward_memory[key])

        return priority_reward_memory_size

    def construct_positive_reward_batch(self, size):
        if size == 0:
            print("construct_prioritized_reward_batch: size is 0.")
            return numpy.zeros([0, self.memory_row_length])

        if self.pile_memory:
            rng = numpy.random.default_rng()
            positive_reward_indices = rng.choice(self.pile_memory_positive_reward_index, size=size, replace=False)
            positive_reward_memory_partition = self.pile_memory_positive_reward[positive_reward_indices.tolist(), :]

            return positive_reward_memory_partition
        else:
            if self.function not in self.priority_reward_memory_all_functions:
                print("construct_prioritized_reward_batch: function positive reward does not exists.")
                return numpy.zeros([0, self.memory_row_length])

            self.priority_reward_memory = self.priority_reward_memory_all_functions[self.function]
            keys = list(self.priority_reward_memory.keys())
            keys_length = len(keys)
            size_of_each_priority = math.floor(size/keys_length)
            remaining = size - size_of_each_priority * keys_length
            keys.sort()
            keys.reverse()

            constructed_priority_memory = numpy.zeros([size, self.memory_row_length])
            constructed_priority_memory_index = 0

            for key in keys:
                current_memory = self.priority_reward_memory[key]
                current_memory_length = current_memory.shape[0]
                if current_memory_length > (size_of_each_priority + remaining):
                    rng = numpy.random.default_rng()
                    priority_reward_indices = rng.choice(current_memory_length,
                                                         size=(size_of_each_priority + remaining), replace=False)

                    priority_reward_memory_partition = current_memory[priority_reward_indices.tolist(), :]

                    remaining = 0
                else:
                    remaining += (size_of_each_priority - current_memory_length)
                    priority_reward_memory_partition = current_memory

                priority_reward_memory_partition_size = priority_reward_memory_partition.shape[0]
                constructed_priority_memory[constructed_priority_memory_index:
                                            constructed_priority_memory_index + priority_reward_memory_partition_size, :] \
                    = priority_reward_memory_partition

                constructed_priority_memory_index += priority_reward_memory_partition_size

            return constructed_priority_memory[0:constructed_priority_memory_index, :]

    def manual_train(self):
        memory_batch = self.get_random_batch_from_memory(self.replay_memory_size)
        if memory_batch.shape[0] == 0:
            print("manual_train: Invalid batch")
            return
        self.training_function(memory_batch[:, 0: self.state_size],
                               memory_batch[:,
                               2 * self.state_size: 2 * self.state_size + self.action_size])

    def train_agent(self):
        if self.n_total_rl_steps < self.replay_memory_size:
            print("Total RL steps is smaller than replay memory size.")
            return
        while self.current_rl_step < self.n_total_rl_steps and self.continue_training:
            print("EPOCH ", self.current_epoch, "/", self.number_of_epochs)
            self.reinforcement_learn_step(self.n_epoch_rl_steps)

            if self.all_function_training and not self.pile_memory:
                if (self.current_epoch % self.training_factor) == 0:
                    for step in range(self.training_each_episode):
                        self.manual_train()
                        self.set_function(numpy.random.randint(low=1, high=self.number_of_all_functions))
            else:
                if self.pile_memory:
                    total_memory_size = self.pile_memory_positive_reward_index + self.pile_memory_zero_reward_index
                else:
                    total_memory_size = self.prioritized_reward_memory_size() + self.zero_reward_memory.shape[0]

                if self.replay_memory_size < total_memory_size:
                    if (self.current_epoch % self.training_factor) == 0:
                        for step in range(self.training_each_episode):
                            self.manual_train()

                if self.all_function_training:
                    self.set_function(numpy.random.randint(low=1, high=self.number_of_all_functions))

            self.reward_per_epoch[self.current_epoch] = self.epoch_total_reward
            self.epoch_total_reward = 0
            self.k_vector = numpy.ones(self.two_to_power_dimension)
            self.current_epoch += 1
