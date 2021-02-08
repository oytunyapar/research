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
        self.dimension_square = dimension**2
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)

        model_layer_sizes.insert(0, dimension**2)
        model_layer_sizes.append(dimension**2)
        self.dqn_agent = create_dqn_agent(model_layer_sizes)

        self.memory = numpy.zeros([n_total_rl_steps, 3*self.dimension_square + 1])
        self.current_state = numpy.sum(self.q_matrix, 1)

        self.n_total_rl_steps = n_total_rl_steps
        self.n_epoch_rl_steps = n_epoch_rl_steps
        self.current_rl_step = 0

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
        next_number_of_zeros = numpy.count_nonzero(next_state == 0)
        number_of_zeros = numpy.count_nonzero(self.current_state == 0)
        return (next_number_of_zeros - number_of_zeros)/self.dimension_square

    def save_memory(self, previous_state, next_state, action, reward):
        self.memory[self.current_rl_step, 0: self.dimension_square] = previous_state
        self.memory[self.current_rl_step, self.dimension_square: 2 * self.dimension_square] = next_state
        self.memory[self.current_rl_step, 2 * self.dimension_square: 3 * self.dimension_square] = action
        self.memory[self.current_rl_step, 3 * self.dimension_square] = reward
        return

    def reinforcement_learn_step(self, step_size):
        if step_size > 0 and self.current_rl_step + step_size < self.n_total_rl_steps:
            for step in range(step_size):
                input_tensor = torch.tensor(self.current_state)
                output = self.model(input_tensor)

                selected_action = output.argmax().tolist()
                self.k_vector[selected_action] += 1

                next_state = numpy.matmul(self.q_matrix, self.k_vector)
                q_value = self.predicted_q_value(next_state)

                self.current_state = next_state

                self.current_rl_step += 1
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
