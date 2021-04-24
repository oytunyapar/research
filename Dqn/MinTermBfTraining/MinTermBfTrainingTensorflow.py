import numpy
import functools
import math

from .MinTermBfTrainingBase import MinTermBfTrainingBase
from ..DqnAgent import DqnAgentTensorflow


class MinTermBfTrainingTensorflow(MinTermBfTrainingBase):
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        super().__init__(function, dimension, n_total_rl_steps, n_epoch_rl_steps, batch_size, model_layer_sizes)
        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)
        self.current_state = self.q_matrix
        self.walsh_spectrum = self.q_matrix.sum(1)

        self.check_current_state = self.current_state

        self.maximum_zeros_during_training = numpy.count_nonzero(self.current_state.sum(1) == 0)

        self.training_function = self.train

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                output = self.dqn_agent(self.current_state.reshape(self.state_size, order='F')).numpy().\
                    reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()
                temp_k_vector = self.k_vector

                for index in range(0, self.two_to_power_dimension):
                    temp_k_vector[index] += 1

                    next_state = numpy.transpose(numpy.matmul(self.q_matrix, temp_k_vector)). \
                        reshape([self.two_to_power_dimension])

                    next_state = (next_state / functools.reduce(numpy.gcd, numpy.array(next_state, dtype=numpy.int)))

                    reward, number_of_zeros = super().predicted_reward(next_state)

                    output_list[index] = \
                        output_list[index] * (1 - self.learning_rate) + \
                        reward * self.learning_rate * pow(self.discount_factor, self.current_rl_step)

                    temp_k_vector[index] -= 1

                #output_list = softmax_internal(output_list)

                self.k_vector[selected_action] += 1

                next_state = numpy.transpose(numpy.matmul(self.q_matrix, self.k_vector)). \
                    reshape([self.two_to_power_dimension])

                next_state = (next_state / functools.reduce(numpy.gcd, numpy.array(next_state, dtype=numpy.int)))

                reward, number_of_zeros = super().predicted_reward(next_state)

                #print("Agent found: reward", reward)

                if number_of_zeros > self.maximum_zeros_during_training:
                    self.maximum_zeros_during_training = number_of_zeros

                super().save_memory(self.current_state.reshape([self.two_to_power_dimension]), next_state, output_list,
                                    reward)

                self.current_state = next_state.reshape([1, self.two_to_power_dimension])

                self.current_rl_step += 1

                #print("Agent found next_state:", next_state)
                #print("Agent found: self.k_vector", self.k_vector.reshape(1, self.dimension_square))
                #print("Agent found: output", output)

        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train(self, given_input, desired_output):
        self.dqn_agent.fit(given_input,
                           desired_output,
                           epochs=50, batch_size=math.floor(self.batch_size / 4))
        return

    def check_agent(self):
        self.check_current_state = numpy.sum(self.q_matrix, 1).reshape([1, self.two_to_power_dimension])
        self.check_k_vector = numpy.ones([self.two_to_power_dimension, 1], dtype=numpy.float32)
        for step in range(self.n_epoch_rl_steps):

            reward, number_of_zeros = super().predicted_reward(self.check_current_state)

            if number_of_zeros >= self.maximum_zeros_during_training:
                print("Agent find maximum zeros:", number_of_zeros)
                print("Agent found:", self.check_current_state)
                break

            output = self.dqn_agent(self.check_current_state).numpy().reshape([self.two_to_power_dimension])

            selected_action = output.argmax().tolist()

            self.check_k_vector[selected_action] += 1

            next_state = numpy.transpose(numpy.matmul(self.q_matrix, self.check_k_vector)). \
                reshape([self.two_to_power_dimension])

            next_state = (next_state / functools.reduce(numpy.gcd, numpy.array(next_state, dtype=numpy.int)))

            self.check_current_state = next_state.reshape([1, self.two_to_power_dimension])
        return
