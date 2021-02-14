import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from DqnAgent.DqnAgent import create_dqn_agent_torch, create_dqn_agent_tensorflow
import numpy
import monsetup
import torch
import math
import tensorflow


class DqnAgentTraining:
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        self.function = function
        self.dimension = dimension
        self.dimension_square = dimension**2
        self.q_matrix = monsetup.q_matrix_generator(function, dimension)

        model_layer_sizes.insert(0, dimension**2)
        model_layer_sizes.append(dimension**2)

        self.memory = numpy.zeros([n_total_rl_steps, 3*self.dimension_square + 1], dtype=numpy.float32)

        self.n_total_rl_steps = n_total_rl_steps
        self.n_epoch_rl_steps = n_epoch_rl_steps
        self.current_rl_step = 0

        self.memory_index = 0

        self.batch_size = batch_size
        self.k_vector = numpy.ones([dimension**2, 1], dtype=numpy.float32)

        self.training_function = None

    def get_random_batch_from_memory(self, size):
        if self.memory_index < size:
            print("DqnAgentTraining:get_random_batch_from_memory ERROR: invalid sized: " + str(size) + " batch request.")
            return numpy.zeros(1)

        rng = numpy.random.default_rng()
        numbers = rng.choice(self.memory_index, size=size, replace=False)
        return self.memory[numbers.tolist(), :]

    def predicted_reward(self, next_state):
        next_number_of_zeros = numpy.count_nonzero(next_state == 0)
        number_of_zeros = numpy.count_nonzero(self.current_state == 0)
        return (next_number_of_zeros - number_of_zeros)/self.dimension_square

    def save_memory(self, previous_state, next_state, action, reward):
        self.memory[self.current_rl_step, 0: self.dimension_square] = previous_state
        self.memory[self.current_rl_step, self.dimension_square: 2 * self.dimension_square] = next_state
        self.memory[self.current_rl_step, 2 * self.dimension_square: 3 * self.dimension_square] = action
        self.memory[self.current_rl_step, 3 * self.dimension_square] = reward
        self.memory_index += 1
        return

    def train_agent(self):
        while self.current_rl_step < self.n_total_rl_steps:
            self.reinforcement_learn_step(self.n_epoch_rl_steps)
            memory_batch = self.get_random_batch_from_memory(self.batch_size)
            self.training_function(memory_batch[:, 0: self.dimension_square],
                                   memory_batch[:, 2 * self.dimension_square: 3 * self.dimension_square])


class DqnAgentTrainingTensorflow(DqnAgentTraining):
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        super().__init__(function, dimension, n_total_rl_steps, n_epoch_rl_steps, batch_size, model_layer_sizes)
        self.dqn_agent = create_dqn_agent_tensorflow(model_layer_sizes)
        self.current_state = numpy.sum(self.q_matrix, 1).reshape([1, self.dimension_square])

        self.training_function = self.train

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                output = self.dqn_agent(self.current_state).numpy().reshape([self.dimension_square])

                selected_action = output.argmax().tolist()
                self.k_vector[selected_action] += 1

                next_state = numpy.transpose(numpy.matmul(self.q_matrix, self.k_vector)). \
                    reshape([self.dimension_square])

                reward = super().predicted_reward(next_state)

                super().save_memory(self.current_state.reshape([self.dimension_square]), next_state, output.tolist(),
                                    reward)

                self.current_state = next_state.reshape([1, self.dimension_square])

                self.current_rl_step += 1
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train(self, given_input, desired_output):
        self.dqn_agent.fit(given_input,
                           desired_output,
                           epochs=10, batch_size=math.floor(self.batch_size / 4))
        return


class DqnAgentTrainingPytorch(DqnAgentTraining):
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        super().__init__(function, dimension, n_total_rl_steps, n_epoch_rl_steps, batch_size, model_layer_sizes)
        self.dqn_agent = create_dqn_agent_torch(model_layer_sizes)
        self.current_state = numpy.sum(self.q_matrix, 1)

        self.training_function = self.train

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                input_tensor = torch.tensor(self.current_state, dtype=torch.float32)
                output = self.dqn_agent(input_tensor.float())

                selected_action = output.argmax().tolist()
                self.k_vector[selected_action] += 1

                next_state = numpy.transpose(numpy.matmul(self.q_matrix, self.k_vector)). \
                    reshape([self.dimension_square])

                reward = super().predicted_reward(next_state)

                super().save_memory(self.current_state, next_state, output.tolist(), reward)

                self.current_state = next_state

                self.current_rl_step += 1
        else:
            print("RL step size problem step size: " + str(step_size) + "\n")

    def train(self, given_input, desired_output):
        loss_function = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.dqn_agent.train()

        optimizer.zero_grad()
        # loss = loss_function(memory_batch[:, 2 * self.dimension_square: 3 * self.dimension_square],
        #                     target)
        # loss.backward()
        optimizer.step()
        return
