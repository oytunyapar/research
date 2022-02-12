from OpenAiGym.MinTermSrpobfEnvBase.MinTermSrpobfEnvBase import MinTermSrpobfEnvBase
from SigmaPiFrameworkPython.sigma_pi_linear_programming import monomial_exclusion

import numpy


class MinTermLpSrpobfEnv(MinTermSrpobfEnvBase):
    def __init__(self, function, dimension,
                 q_matrix_representation, episodic_reward):
        super(MinTermLpSrpobfEnv, self).\
            __init__(function, dimension, q_matrix_representation, episodic_reward)

        self.steps_in_each_epoch = self.two_to_power_dimension * 2
        self.action_size = self.two_to_power_dimension

        self.key_name = "monomial_set"
        self.key_size = self.two_to_power_dimension
        self.reset_key()

        self.state_size = self.function_representation_size + self.key_size

        self.max_reward_key = self.key.copy()

        super(MinTermLpSrpobfEnv, self)._create_action_and_observation_space()

    def reset_key(self):
        self.key = numpy.zeros(self.key_size)
