from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv


def random_action_runner(dimension):
    result_metrics = {}
    print_constant = 100000
    total_steps = number_of_steps_dictionary[dimension]
    for function in BooleanFunctionsEquivalentClasses[dimension]:
        env = MinTermSrpobfEnv(function, dimension, q_matrix_representation, act,
                               no_action_episode_end, episodic_reward=True)

        for step in range(number_of_steps_dictionary[dimension]):
            observation, returned_reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()

            if step % print_constant == 0:
                print("Function:" + str(function) +
                      " continues " + str(step) + "/" + str(total_steps))

        result_metrics[str(dimension) + "_" + str(function) + "_max_reward"] = env.max_rewards_in_the_episodes

    return result_metrics
