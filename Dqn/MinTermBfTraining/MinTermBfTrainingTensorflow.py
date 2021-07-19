import numpy
import functools
import math
import tensorflow

from .MinTermBfTrainingBase import MinTermBfTrainingBase
from ..DqnAgent import DqnAgentTensorflow


class MinTermBfTrainingTensorflow(MinTermBfTrainingBase):
    def __init__(self, function, dimension, number_of_epochs, model_layer_sizes,
                 q_matrix_representation, multiprocess, pile_memory):
        super().__init__(function, dimension, number_of_epochs, model_layer_sizes,
                         q_matrix_representation, multiprocess, pile_memory)
        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)

        if multiprocess:
            with tensorflow.device("/cpu:0"):
                self.dqn_agent_cpu = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)
                self.dqn_agent_cpu.set_weights(self.dqn_agent.get_weights())

            self.training_function = self.train_multi_process
        else:
            self.training_function = self.train

        self.check_current_state = self.current_state.copy()

    def reinforcement_learn_step_multi_process(self, step_size):
        if step_size > 0:
            zero_reward_memory = numpy.zeros([step_size, self.memory_row_length])
            zero_reward_memory_index = 0
            positive_reward_memory = numpy.zeros([step_size, self.memory_row_length])
            positive_reward_memory_index = 0

            k_vector = numpy.ones(self.two_to_power_dimension)
            current_state = numpy.ones([1, self.state_size])
            current_state[0, 0:self.function_representation_size] = self.function_representation
            current_state[0, self.function_representation_size:self.state_size] = k_vector

            for step in range(step_size):
                with tensorflow.device("/cpu:0"):
                    output = self.dqn_agent_cpu(current_state).numpy().reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility_epoch():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()

                no_action = False

                if selected_action < k_vector.size:
                    k_vector[selected_action] += 1

                    k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(k_vector, dtype=numpy.int))

                    k_vector /= k_vector_gcd

                    next_state = current_state.copy()
                    next_state[0, self.function_representation_size:self.state_size] = k_vector

                    reward, number_of_zeros = super().predicted_reward(k_vector)

                else:
                    next_state = current_state.copy()
                    no_action = True
                    reward, number_of_zeros = super().predicted_reward(k_vector)
                with tensorflow.device("/cpu:0"):
                    next_output = self.dqn_agent_cpu(next_state).numpy().reshape([self.action_size])

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

                zero_reward_memory_index, positive_reward_memory_index = \
                    self.save_memory_local(zero_reward_memory, zero_reward_memory_index,
                                           positive_reward_memory, positive_reward_memory_index,
                                           current_state, next_state, output_list, reward)
                current_state = next_state
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

        return zero_reward_memory[0:zero_reward_memory_index, :],\
            positive_reward_memory[0:positive_reward_memory_index, :]

    def reinforcement_learn_step_multi_process_wrapper(self, return_value_queue, step_size):
        return_value_queue.put(self.reinforcement_learn_step_multi_process(step_size))

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step

            for step in range(step_size):
                output = self.dqn_agent(self.current_state).numpy().reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()

                no_action = False

                if selected_action < self.k_vector.size:
                    self.k_vector[selected_action] += 1

                    k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(self.k_vector, dtype=numpy.int))

                    self.k_vector /= k_vector_gcd

                    next_state = self.current_state.copy()
                    next_state[0, self.function_representation_size:self.state_size] = self.k_vector

                    reward, number_of_zeros = super().predicted_reward(self.k_vector)

                    if number_of_zeros > self.maximum_zeros_during_training[self.function]:
                        self.maximum_zeros_during_training[self.function] = number_of_zeros
                        self.maximum_zeros_k_vector[self.function, :] = self.k_vector

                else:
                    next_state = self.current_state.copy()
                    no_action = True
                    reward, number_of_zeros = super().predicted_reward(self.k_vector)

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

                super().save_memory(self.current_state, next_state, output_list, reward)

                self.current_state = next_state
                self.current_rl_step += 1
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

        self.k_vector = numpy.ones(self.two_to_power_dimension)
        self.current_state[0, self.function_representation_size:self.state_size] = self.k_vector

    def train(self, given_input, desired_output):
        self.dqn_agent.fit(given_input,
                           desired_output,
                           epochs=5, batch_size=self.batch_size)

        return

    def train_multi_process(self, given_input, desired_output):
        self.dqn_agent.fit(given_input,
                           desired_output,
                           epochs=5, batch_size=self.batch_size)

        self.dqn_agent_cpu.set_weights(self.dqn_agent.get_weights())

        return

    def check_agent(self, loop_constant):
        number_of_zeros = 0
        for step in range(loop_constant * self.n_epoch_rl_steps):

            reward, number_of_zeros = super().predicted_reward(self.k_vector_check)

            if number_of_zeros >= self.maximum_zeros_during_training[self.function]:
                print("Already max zero found during training.")

            self.check_current_state[0, 0:self.function_representation_size] = self.function_representation
            self.check_current_state[0, self.function_representation_size:self.state_size] = self.k_vector_check
            output = self.dqn_agent(self.check_current_state).numpy().reshape([self.action_size])

            selected_action = output.argmax().tolist()

            if selected_action < self.k_vector.size:
                self.k_vector_check[selected_action] += 1

                k_vector_check_gcd = functools.reduce(numpy.gcd, numpy.array(self.k_vector_check, dtype=numpy.int))

                self.k_vector_check /= k_vector_check_gcd
            else:
                print("No Action!")
                break

        self.coefficients = (2 ** -self.dimension) * numpy.matmul(self.q_matrix, self.k_vector_check)

        print("Agent find maximum zeros:", number_of_zeros)
        print("Agent found:", self.k_vector_check)
        self.k_vector_check = numpy.ones(self.two_to_power_dimension)

        return
