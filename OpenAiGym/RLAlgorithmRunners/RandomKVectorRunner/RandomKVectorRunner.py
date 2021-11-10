from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
import numpy as np


def random_k_vector_runner(dimension):
    result_metrics = {}
    print_constant = 100000
    total_steps = number_of_steps_dictionary[dimension]
    for function in BooleanFunctionsEquivalentClasses[dimension]:
        env = MinTermSrpobfEnv(function, dimension, q_matrix_representation, act,
                               no_action_episode_end, episodic_reward=True)

        for step in range(number_of_steps_dictionary[dimension]):
            env.k_vector = (np.random.randint(env.k_vector_element_max_value, size=env.k_vector_size) + 1)
            returned_reward, done = env.step_without_action()
            if done:
                env.reset()

            if step % print_constant == 0:
                print("Function:" + str(function) +
                      " continues " + str(step) + "/" + str(total_steps))

        result_metrics[str(dimension) + "_" + str(function) + "_max_reward"] = env.max_rewards_in_the_episodes

    return result_metrics
