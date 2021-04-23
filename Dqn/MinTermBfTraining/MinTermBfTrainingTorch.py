import numpy
import functools
import torch

from .MinTermBfTrainingBase import MinTermBfTrainingBase
from ..DqnAgent import DqnAgentTorch


class MinTermBfTrainingTorch(MinTermBfTrainingBase):
    def __init__(self, function, dimension, n_total_rl_steps,
                 n_epoch_rl_steps, batch_size, model_layer_sizes):
        super().__init__(function, dimension, n_total_rl_steps, n_epoch_rl_steps, batch_size, model_layer_sizes)
        self.dqn_agent = DqnAgentTorch.create_dqn_agent_torch(model_layer_sizes)
        self.current_state = numpy.sum(self.q_matrix, 1)

        self.training_function = self.train

    def reinforcement_learn_step(self, step_size):
        if step_size > 0:
            if self.current_rl_step + step_size > self.n_total_rl_steps:
                step_size = self.n_total_rl_steps - self.current_rl_step
            for step in range(step_size):
                input_tensor = torch.tensor(self.current_state, dtype=torch.float32)
                output = self.dqn_agent(input_tensor.float())

                if numpy.random.uniform(0, 1) > self.random_movement_possibility:
                    selected_action = output.argmax().tolist()
                else:
                    selected_action = numpy.random.default_rng().choice(self.two_to_power_dimension)

                self.k_vector[selected_action] += 1
                next_state = numpy.transpose(numpy.matmul(self.q_matrix, self.k_vector)). \
                    reshape([self.two_to_power_dimension])

                next_state = (next_state / functools.reduce(numpy.gcd, numpy.array(next_state, dtype=numpy.int)))

                reward, number_of_zeros = super().predicted_reward(next_state)

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