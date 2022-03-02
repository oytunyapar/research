from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import *
from OpenAiGym.RLAlgorithmRunners.Utils.EnvironmentHelperFunctions import env_creator, KeyType
from OpenAiGym.RLAlgorithmRunners.Utils.DataHelperFunctions import *
import datetime


def random_action_runner(functions,
                         dimension,
                         time_steps,
                         output_directory=None,
                         output_folder_prefix=None,
                         key_type=KeyType.K_VECTOR):
    env = env_creator(functions, dimension, key_type)
    print_constant = 10000

    for step in range(time_steps):
        observation, returned_reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()

        if step % print_constant == 0:
            print("Step " + str(step) + "/" + str(time_steps))

    random_action_runner_output_helper(output_directory, output_folder_prefix, env)

    return env


def random_action_runner_output_helper(root_directory, dump_directory_prefix, env):
    if root_directory is not None:
        output_directory = root_directory + "/" + str(env.dimension) + \
                                    dump_directory_prefix + "_" + str(datetime.datetime.now())

        dump_json(env.function_each_episode, output_directory, "function_each_episode")
        dump_json(env.max_reward_dict, output_directory, "max_reward_dict")
        dump_json(env.max_reward_key_dict, output_directory, "max_reward_" + env.key_name + "_dict")

        performance_results = {}
        for function, reward in env.max_reward_dict.items():
            performance_results[function] = reward_performance(env, reward, function)

        dump_json(performance_results, output_directory, "performance_results")
