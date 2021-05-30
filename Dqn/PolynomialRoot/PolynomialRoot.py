import numpy
import signal
import math
from ..DqnAgent import DqnAgentTensorflow


class PolynomialRoot:
    def __init__(self, coefficients, degree, number_of_epochs, model_layer_sizes):
        coefficients_length = len(coefficients)

        self.all_polynomial_training = False
        if coefficients_length == 0:
            self.all_polynomial_training = True
        elif coefficients_length != (degree + 1):
            raise Exception("Number of coefficients: ", coefficients_length,
                            "is not compatible with degree: ", degree)

        self.degree = degree
        self.number_of_coefficients = degree + 1

        if not self.all_polynomial_training:
            self.coefficients = coefficients
            self.max_coefficient_value = 100

        self.continue_training = True

        self.variable_value = 1
        self.state_size = coefficients_length + 1
        self.action_size_except_no_action = 2
        self.action_size = self.action_size_except_no_action + 1
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)

        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)

        self.current_state = numpy.ones([1, self.state_size])

        self.n_epoch_rl_steps = 50
        self.n_total_rl_steps = number_of_epochs * self.n_epoch_rl_steps
        self.current_rl_step = 0
        self.number_of_epochs = number_of_epochs
        self.current_epoch = 0

        self.discount_factor = 0.99
        self.learning_rate = 0.6

        self.epoch_total_reward = 0
        self.reward_per_epoch = numpy.zeros([number_of_epochs])

        self.memory_row_length = 2 * self.state_size + self.action_size + 1
        self.memory_size = self.n_epoch_rl_steps
        self.memory_index = 0
        self.replay_memory_size = math.ceil(self.memory_size / self.degree)
        self.batch_size = math.ceil(self.replay_memory_size / self.degree)

        self.memory = numpy.zeros(
            [self.memory_size, self.memory_row_length],
            dtype=numpy.float32)

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, signum, frame):
        print("User interrupt received.")
        self.continue_training = False

    def random_movement_possibility(self):
        offset = 2
        return numpy.tanh(offset - offset * (self.current_rl_step / self.n_total_rl_steps))

    def polynomial(self, variable):
        result = 0
        for power in range(self.degree + 1):
            result += self.coefficients[power] * (variable ** power)
        return result

    def predicted_reward(self, variable):
        return -abs(self.polynomial(variable))

    def save_memory(self, previous_state, next_state, action, reward):
        memory_entry = numpy.zeros([1, self.memory_row_length])
        memory_entry[0, 0: self.state_size] = previous_state.reshape(self.state_size)
        memory_entry[0, self.state_size: 2 * self.state_size] = next_state.reshape(self.state_size)
        memory_entry[0, 2 * self.state_size: 2 * self.state_size + self.action_size] = action
        memory_entry[0, 2 * self.state_size + self.action_size] = reward

        if self.memory_index >= self.memory_size:
            print("ERROR: Memory is full.")
        else:
            self.memory[self.memory_index] = memory_entry
            self.memory_index += 1

    def clear_memory(self):
        self.memory_index = 0

    def get_random_batch_from_memory(self, size):
        if size > self.memory_index:
            print("ERROR: Requested size: ", size, " random memory batch is invalid.")
            return
        rng = numpy.random.default_rng()
        memory_indices = rng.choice(self.memory_index, size=size, replace=False)
        memory_partition = self.memory[memory_indices.tolist(), :]
        return memory_partition

    def change_function(self):
        self.coefficients = numpy.random.randint(self.max_coefficient_value,
                                                 size=self.number_of_coefficients).tolist()
        return

    def train(self):
        memory_batch = self.get_random_batch_from_memory(self.replay_memory_size)
        if memory_batch.shape[0] == 0:
            print("manual_train: Invalid batch")
            return
        self.dqn_agent.fit(memory_batch[:, 0: self.state_size],
                           memory_batch[:, 2 * self.state_size: 2 * self.state_size + self.action_size],
                           epochs=5,
                           batch_size=self.batch_size)

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                self.current_state[0, 0:self.coefficients_length] = self.coefficients
                self.current_state[0, self.coefficients_length] = self.variable_value
                output = self.dqn_agent(self.current_state).numpy().reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()

                no_action = False

                #if selected_action < self.action_size_except_no_action:
                #else:
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")