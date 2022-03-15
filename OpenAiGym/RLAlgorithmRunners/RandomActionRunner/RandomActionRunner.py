from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import *
from OpenAiGym.RLAlgorithmRunners.Utils.EnvironmentHelperFunctions import env_creator, KeyType, get_env_name
from OpenAiGym.RLAlgorithmRunners.Utils.DataHelperFunctions import *
import datetime
from pathlib import Path


def random_action_runner(functions,
                         dimension,
                         epochs,
                         env=None,
                         output_directory=None,
                         output_folder_label=None,
                         key_type=KeyType.K_VECTOR,
                         print_epochs=False):
    if env is None:
        env = env_creator(functions, dimension, key_type)

    print_constant = 100

    for epoch in range(epochs):
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
        
        env.reset()

        if epoch % print_constant == 0 and print_epochs:
            print("Epoch:" + str(epoch + 1) + "/" + str(epochs))

    random_action_runner_output_helper(output_directory, output_folder_label, env)

    return env


def random_action_runner_n_times(functions,
                                 dimension,
                                 n_times,
                                 output_directory=None,
                                 output_folder_label=None,
                                 key_type=KeyType.K_VECTOR
                                 ):
    env = env_creator([0], dimension, key_type)

    for times in range(n_times):
        for function in functions:
            env.set_function(function)
            env = random_action_runner(function, dimension, 1, env=env, key_type=key_type)
        print("Times:" + str(times + 1) + "/" + str(n_times))

    random_action_runner_output_helper(output_directory, output_folder_label, env)

    return env


def random_action_monte_carlo_runner(monte_carlo_times, n_times, functions, dimension,
                                     output_folder_label=None, key_type=KeyType.K_VECTOR):
    root_directory = str(Path.home()) + "/PycharmProjects/research/OpenAiGym/" + \
                     get_env_name(key_type) + "/Data/" + str(dimension) + "dim/RandomAction/" + \
                     str(datetime.datetime.now()) + "_" + "Monte_Carlo"

    if output_folder_label is not None:
        root_directory = root_directory + "_" + output_folder_label

    for times in range(monte_carlo_times):
        env = random_action_runner_n_times(functions, dimension, n_times, key_type=key_type)
        output_directory = root_directory + "/" + str(times)
        random_action_runner_output(output_directory, env)
        print("Monte Carlo times:" + str(times + 1) + "/" + str(monte_carlo_times))

    return root_directory


def random_action_runner_output_helper(root_directory, output_folder_label, env):
    if root_directory is not None and output_folder_label is not None:
        output_directory = root_directory + "/" +\
                           type(env).__name__ + "/Data/" + str(env.dimension) + "dim/RandomAction/" + \
                           str(datetime.datetime.now()) + "_" + output_folder_label
        random_action_runner_output(output_directory, env)


def random_action_runner_output(output_directory, env):
    dump_json(env.function_each_episode, output_directory, "function_each_episode")
    dump_json(env.max_reward_dict, output_directory, "max_reward_dict")
    dump_json(env.max_reward_key_dict, output_directory, "max_reward_" + env.key_name + "_dict")

    performance_results = {}
    for function, reward in env.max_reward_dict.items():
        performance_results[function] = reward_performance(env, reward, function)

    dump_json(performance_results, output_directory, "performance_results")
