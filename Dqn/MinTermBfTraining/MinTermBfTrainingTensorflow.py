import numpy
import functools
import math

from .MinTermBfTrainingBase import MinTermBfTrainingBase
from ..DqnAgent import DqnAgentTensorflow


class MinTermBfTrainingTensorflow(MinTermBfTrainingBase):
    def __init__(self, function, dimension, number_of_epochs, model_layer_sizes):
        super().__init__(function, dimension, number_of_epochs, model_layer_sizes)
        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)

        self.check_current_state = self.current_state.copy()

        self.training_function = self.train

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                self.current_state[0, 0:self.function_representation_size] = self.function_representation
                self.current_state[0, self.function_representation_size:self.state_size] = self.k_vector
                output = self.dqn_agent(self.current_state).numpy().reshape([self.action_size])

                if numpy.random.uniform(0, 1) > self.random_movement_possibility():
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.action_size)

                output_list = output.tolist()

                reward = 0
                no_action = False

                if selected_action < self.k_vector.size:
                    self.k_vector[selected_action] += 1

                    k_vector_gcd = functools.reduce(numpy.gcd, numpy.array(self.k_vector, dtype=numpy.int))

                    self.k_vector /= k_vector_gcd

                    next_state = numpy.ones([1, self.state_size])
                    next_state[0, 0:self.function_representation_size] = self.function_representation
                    next_state[0, self.function_representation_size:self.state_size] = self.k_vector

                    reward, number_of_zeros = super().predicted_reward(self.k_vector)

                    if number_of_zeros > self.maximum_zeros_during_training:
                        self.maximum_zeros_during_training = number_of_zeros
                        self.maximum_zeros_k_vector = self.k_vector.copy()

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
                        self.learning_rate * pow(self.discount_factor, self.current_rl_step) * next_output_max
                else:
                    output_list[selected_action] = \
                        output_list[selected_action] * (1 - self.learning_rate) + \
                        reward * self.learning_rate

                super().save_memory(self.current_state, next_state, output_list, reward)
                self.current_state = next_state
                self.current_rl_step += 1
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train(self, given_input, desired_output):
        self.dqn_agent.fit(given_input,
                           desired_output,
                           epochs=self.two_to_power_dimension, batch_size=math.floor(self.batch_size / 4))
        return

    def check_agent(self, loop_constant):
        number_of_zeros = 0
        for step in range(loop_constant * self.n_epoch_rl_steps):

            reward, number_of_zeros = super().predicted_reward(self.k_vector_check)

            if number_of_zeros >= self.maximum_zeros_during_training:
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

        print("Agent find maximum zeros:", number_of_zeros)
        print("Agent found:", self.k_vector_check)
        self.k_vector_check = numpy.ones(self.two_to_power_dimension)

        return
