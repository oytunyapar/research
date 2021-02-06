import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from DqnAgent.DqnAgent import create_dqn_agent
import numpy
import monsetup


class DqnAgentTraining:

    def __init__(self, function, dimension, number_of_trials, batch_size, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.q_matrix = monsetup.qMatrixGenerator(function, dimension)
        self.dqn_agent = create_dqn_agent(model_layer_sizes)
        self.memory = numpy.zeros([number_of_trials, 4])

    def get_random_batch_from_memory(self, size):
        if self.memory.shape[0] < size:
            print("DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        rng = numpy.random.default_rng()
        numbers = rng.choice(self.memory.shape[0], size=size, replace=False)
        return self.memory[numbers.tolist(), ...]

    def train_agent(self):
        return
