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

        if not self.all_polynomial_training:
            self.coefficients = coefficients

        self.continue_training = True

        self.variable_value = 1
        self.state_size = self.degree + 1
        self.action_size = 2
        model_layer_sizes.insert(0, self.state_size)
        model_layer_sizes.append(self.action_size)
        self.memory_row_length = 2 * self.state_size + self.action_size + 1

        self.dqn_agent = DqnAgentTensorflow.create_dqn_agent_tensorflow(model_layer_sizes)

        self.current_state = numpy.ones([1, self.state_size])

        self.n_epoch_rl_steps = 100
        self.n_total_rl_steps = number_of_epochs * self.n_epoch_rl_steps
        self.current_rl_step = 0
        self.number_of_epochs = number_of_epochs

        self.discount_factor = 0.99
        self.learning_rate = 0.6

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
