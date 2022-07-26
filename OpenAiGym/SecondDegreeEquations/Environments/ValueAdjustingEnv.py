import gym
from gym import spaces
import numpy
import decimal


class ValueAdjustingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, root_limits):
        super(ValueAdjustingEnv, self).__init__()

        self.number_of_roots = 2
        self.size_of_limit_array = 2

        self.decimal_precision = self.find_decimal_precision(root_limits)

        self.root_limits = sorted(root_limits)
        self.roots = []
        self.equation_coefficients = []

        self.create_equation()

        self.length_of_equation_coefficients = len(self.equation_coefficients)

        self.steps_in_each_epoch = 300
        self.current_step = 0

        self.action_factor = 10 ** -self.decimal_precision
        self.action_size = 2
        self.action_increase = 0
        self.action_decrease = 1

        self.state_size = self.length_of_equation_coefficients + 1
        self._create_action_and_observation_space()

        self.current_value = 0
        self.reset_current_value()

        _, self.previous_distance = self.minimum_distance_from_roots()

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

    def create_roots(self):
        return [round(root, self.decimal_precision)
                for root in numpy.random.uniform(self.root_limits[0], self.root_limits[1], self.number_of_roots)]

    def create_equation(self):
        self.roots = self.create_roots()
        self.equation_coefficients = [1, -(self.roots[0] + self.roots[1]), self.roots[0] * self.roots[1]]

    def equation_calculate(self, value):
        return self.equation_coefficients[0] * (value ** 2) + self.equation_coefficients[1] * value +\
               self.equation_coefficients[2]

    def _create_action_and_observation_space(self):
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, [self.state_size])

    def create_observation(self):
        observation = numpy.ones([self.state_size])
        observation[0:self.length_of_equation_coefficients] = self.equation_coefficients
        observation[self.length_of_equation_coefficients] = self.current_value
        return observation

    def close(self):
        pass

    def reset_current_value(self):
        selected_root = numpy.random.choice(self.roots)
        current_value_limits = [selected_root,
                                selected_root + numpy.random.choice([1, -1]) *
                                self.steps_in_each_epoch/3 * self.action_factor]

        self.current_value = round(numpy.random.uniform(current_value_limits[0], current_value_limits[1], 1)[0],
                                   self.decimal_precision)

    def reset(self):
        self.current_step = 0
        self.create_equation()
        self.reset_current_value()

        return self.create_observation()

    def step(self, action):
        self.current_step += 1

        if action == self.action_increase:
            self.current_value += self.action_factor
        elif action == self.action_decrease:
            self.current_value -= self.action_factor
        else:
            raise Exception("Unknown action!")

        observation = self.create_observation()
        reward = self.reward()
        done = self.check_episode_end()

        info = {}

        return observation, reward, done, info

    def minimum_distance_from_roots(self):
        distance_from_first_root = abs(self.current_value - self.roots[0])
        distance_from_second_root = abs(self.current_value - self.roots[1])

        if distance_from_first_root > distance_from_second_root:
            target_root = self.roots[1]
            target_distance = distance_from_second_root
        else:
            target_root = self.roots[0]
            target_distance = distance_from_first_root

        return target_root, target_distance

    def reward(self):
        target_root, target_distance = self.minimum_distance_from_roots()

        if self.previous_distance > target_distance:
            if self.current_value == target_root:
                reward = 2 * self.action_factor
            else:
                reward = self.action_factor
        else:
            reward = -self.action_factor

        self.previous_distance = target_distance

        return reward

    def check_episode_end(self):
        return self.current_step > self.steps_in_each_epoch or self.equation_calculate(self.current_value) == 0
