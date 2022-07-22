import gym
from gym import spaces
import numpy
import math
import decimal


class ValueAdjustingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, root_limits):
        super(ValueAdjustingEnv, self).__init__()

        self.number_of_roots = 2
        self.size_of_limit_array = 2

        self.decimal_precision = self.find_decimal_precision(root_limits)

        root_limits.sort()
        self.roots = [round(root, self.decimal_precision)
                      for root in numpy.random.uniform(root_limits[0], root_limits[1], self.number_of_roots)]

    def check_root_limits(self, root_limits):
        if not isinstance(root_limits, list):
            raise Exception("Root limits is not an array.")

        if len(root_limits) != self.size_of_limit_array:
            raise Exception("Root limit array length is not 2.")

        if not all(isinstance(limit, int) for limit in root_limits) and \
                not all(isinstance(limit, float) for limit in root_limits):
            raise Exception("Root limit array element type is not float or int.")

    def find_decimal_precision(self, root_limits):
        self.check_root_limits(root_limits)

        first_precision = abs(decimal.Decimal(str(root_limits[0])).as_tuple().exponent)
        second_precision = abs(decimal.Decimal(str(root_limits[1])).as_tuple().exponent)

        if first_precision > second_precision:
            return first_precision
        else:
            return second_precision

    def close(self):
        pass

    def reset(self):
        observation = self.create_observation()
        return observation

    def step(self, action):
        observation = self.create_observation()
        info = {}
        reward = 1
        done = False

        return observation, reward, done, info

    def reward(self, next_k_vector):
        next_number_of_zeros = numpy.count_nonzero(self.calculate_weights(next_k_vector) == 0)
        reward = next_number_of_zeros**2
        return reward
