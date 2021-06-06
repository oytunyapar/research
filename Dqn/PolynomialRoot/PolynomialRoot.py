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
            self.number_of_episodes = 1
            self.coefficients = coefficients
        else:
            self.number_of_episodes = 100
            self.max_coefficient_value = 100
            self.change_function()

        self.continue_training = True

        self.variable_value = 1.0
        self.test_variable_value = 1.0
        self.state_size = self.number_of_coefficients + 1
        self.action_size_except_no_action = 2
        self.action_size = self.action_size_except_no_action + 1
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)

        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)

        self.current_state = numpy.ones([1, self.state_size])
        self.current_state[0, 0:self.number_of_coefficients] = self.coefficients
        self.current_state[0, self.number_of_coefficients] = self.variable_value

        self.check_current_state = self.current_state.copy()

        self.n_epoch_rl_steps = 50
        self.n_episode_rl_steps = number_of_epochs * self.n_epoch_rl_steps
        self.n_total_rl_steps = self.number_of_episodes * self.n_episode_rl_steps
        self.episode_rl_step_counter = 0
        self.total_rl_step_counter = 0
        self.number_of_epochs_per_episode = number_of_epochs
        self.total_number_of_epochs = self.number_of_epochs_per_episode * self.number_of_episodes
        self.current_epoch = 0

        self.discount_factor = 0.99
        self.learning_rate = 0.6

        self.old_absolute_value = abs(self.polynomial(self.variable_value))

        self.epoch_total_reward = 0
        self.reward_per_epoch = numpy.zeros([self.total_number_of_epochs])

        self.memory_row_length = 2 * self.state_size + self.action_size + 1
        self.memory_size = self.n_episode_rl_steps
        self.memory_index = 0
        self.replay_memory_size = self.n_epoch_rl_steps * 10
        self.batch_size = math.ceil(self.replay_memory_size / self.degree)

        self.memory = numpy.zeros(
            [self.memory_size, self.memory_row_length],
            dtype=numpy.float32)

        self.increase_variable_action = 0
        self.decrease_variable_action = 1
        self.change_margin = 0.1

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, signum, frame):
        print("User interrupt received.")
        self.continue_training = False

    def random_movement_possibility(self):
        offset = 2
        return numpy.tanh(offset - offset * (self.total_rl_step_counter / self.n_total_rl_steps))

    def polynomial(self, variable):
        result = 0
        for power in range(self.degree + 1):
            result += self.coefficients[power] * (variable ** power)
        return result

    def predicted_reward(self, variable):
        new_absolute_value = abs(self.polynomial(variable))
        reward = self.old_absolute_value - new_absolute_value
        self.old_absolute_value = new_absolute_value
        return ((self.degree**2) * reward) - new_absolute_value/self.degree

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

    def clear_episode_counters(self):
        self.episode_rl_step_counter = 0

    def get_random_batch_from_memory(self, size):
        if size > self.memory_index:
            print("ERROR: Requested size: ", size, " random memory batch is invalid.")
            return
        rng = numpy.random.default_rng()
        memory_indices = rng.choice(self.memory_index, size=size, replace=False)
        memory_partition = self.memory[memory_indices.tolist(), :]
        return memory_partition

    def change_function(self):
        self.variable_value = 1.0
        self.coefficients = numpy.random.randint(self.max_coefficient_value,
                                                 size=self.number_of_coefficients).tolist()
        self.current_state[0, 0:self.number_of_coefficients] = self.coefficients
        self.current_state[0, self.number_of_coefficients] = self.variable_value
        self.check_current_state = self.current_state.copy()
        self.test_variable_value = 1.0
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
            if self.episode_rl_step_counter + step_size > self.n_episode_rl_steps:
                step_size = self.n_episode_rl_steps - self.episode_rl_step_counter
            for step in range(step_size):
                output = self.dqn_agent(self.current_state).numpy().reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()

                no_action = False

                if selected_action < self.action_size_except_no_action:
                    if selected_action == self.increase_variable_action:
                        self.variable_value = round(self.variable_value + self.change_margin, 2)
                    elif selected_action == self.decrease_variable_action:
                        self.variable_value = round(self.variable_value - self.change_margin, 2)
                    else:
                        raise Exception("reinforcement_learn_step: Undefined behaviour")

                    next_state = self.current_state.copy()
                    next_state[0, self.number_of_coefficients] = self.variable_value

                    reward = self.predicted_reward(self.variable_value)
                else:
                    next_state = self.current_state.copy()
                    no_action = True
                    reward = self.predicted_reward(self.variable_value)

                self.epoch_total_reward += reward

                next_output = self.dqn_agent(next_state).numpy().reshape([self.action_size])
                next_output_max = next_output[next_output.argmax()]

                if not no_action:
                    output_list[selected_action] = \
                        output_list[selected_action] * (1 - self.learning_rate) + \
                        reward * self.learning_rate + \
                        self.learning_rate * self.discount_factor * next_output_max
                else:
                    output_list[selected_action] = \
                        output_list[selected_action] * (1 - self.learning_rate) + \
                        reward * self.learning_rate

                self.save_memory(self.current_state, next_state, output_list, reward)
                self.current_state = next_state
                self.episode_rl_step_counter += 1
                self.total_rl_step_counter += 1
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train_agent(self):
        if self.n_episode_rl_steps < self.replay_memory_size:
            print("Total RL steps is smaller than replay memory size.")
            return

        for episode in range(self.number_of_episodes):
            while self.episode_rl_step_counter < self.n_episode_rl_steps and self.continue_training:
                print("EPOCH ", self.current_epoch, "/", self.total_number_of_epochs)
                self.reinforcement_learn_step(self.n_epoch_rl_steps)

                if self.replay_memory_size < self.memory_index:
                    self.train()

                self.reward_per_epoch[self.current_epoch] = self.epoch_total_reward
                self.epoch_total_reward = 0
                self.current_epoch += 1

            if self.number_of_episodes > 1:
                self.clear_episode_counters()
                self.clear_memory()
                self.change_function()

    def check_agent(self, loop_constant):
        for step in range(loop_constant * self.n_epoch_rl_steps):
            self.check_current_state[0, 0:self.number_of_coefficients] = self.coefficients
            self.check_current_state[0, self.number_of_coefficients] = self.test_variable_value
            output = self.dqn_agent(self.check_current_state).numpy().reshape([self.action_size])
            selected_action = output.argmax().tolist()

            if selected_action < self.action_size_except_no_action:
                if selected_action == self.increase_variable_action:
                    self.test_variable_value = round(self.test_variable_value + self.change_margin, 2)
                elif selected_action == self.decrease_variable_action:
                    self.test_variable_value = round(self.test_variable_value - self.change_margin, 2)
                else:
                    raise Exception("reinforcement_learn_step: Undefined behaviour")
            else:
                print("No Action!")
                break

        print("Agent found:", self.test_variable_value)
        print("When we put in polynomial:", self.polynomial(self.test_variable_value))
