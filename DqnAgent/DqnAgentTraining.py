import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from DqnAgent.DqnAgent import create_dqn_agent
import numpy
import monsetup
import torch


class DqnAgentTraining:

    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)
        self.dqn_agent = create_dqn_agent(model_layer_sizes)
        self.memory = numpy.zeros([n_total_rl_steps, 3*(dimension**2)])
        self.input_variable = numpy.sum(self.q_matrix, 1)
        self.remaining_rl_steps = n_total_rl_steps
        self.n_epoch_rl_steps = n_epoch_rl_steps
        self.batch_size = batch_size
        self.k_vector = numpy.ones([dimension**2, 1])

    def get_random_batch_from_memory(self, size):
        if self.memory.shape[0] < size:
            print("DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        rng = numpy.random.default_rng()
        numbers = rng.choice(self.memory.shape[0], size=size, replace=False)
        return self.memory[numbers.tolist(), :]

    def predicted_q_value(self, next_state):
        return

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            for step in range(step_size):
                input_tensor = torch.tensor(self.input_variable)
                output = self.model(input_tensor)
                selected_action = output.argmax().tolist()
                self.k_vector[selected_action] += 1
                next_state = numpy.matmul(self.q_matrix, self.k_vector)
                q_value = self.predicted_q_value(next_state)
                self.input_variable = next_state
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train_agent(self):
        self.dqn_agent.train()

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        loss_function = torch.nn.MSELoss(reduction='sum')

        optimizer.zero_grad()
        #loss = loss_function(output, target)
        #loss.backward()
        optimizer.step()

        return
