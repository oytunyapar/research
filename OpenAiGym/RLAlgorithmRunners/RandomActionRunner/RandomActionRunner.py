from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import *
from OpenAiGym.RLAlgorithmRunners.Utils.EnvironmentHelperFunctions import *
from OpenAiGym.RLAlgorithmRunners.Utils.DataHelperFunctions import *
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import *
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
    complement_functions = get_complement_function_list(dimension, functions)

    test_mode = False
    if len(complement_functions) > 0:
        test_mode = True

    parameters_dict = {"monte_carlo_times": monte_carlo_times, "n_times": n_times, "dimension": dimension,
                       "functions": functions}

    performance_mean_variance = {"perf_mean_train": 0, "perf_deviance_train": 0, "perf_mean_test": 0, "perf_deviance_test": 0,
                          "perf_mean": 0, "perf_deviance": 0}

    root_directory = str(Path.home()) + "/PycharmProjects/research/OpenAiGym/" + \
                     get_env_name_from_key_type(key_type) + "/Data/" + str(dimension) + "dim/RandomAction/" + \
                     str(datetime.datetime.now()) + "_" + "Monte_Carlo"

    if output_folder_label is not None:
        root_directory = root_directory + "_" + output_folder_label

    for times in range(monte_carlo_times):
        output_directory = root_directory + "/" + str(times)
        if test_mode:
            _, perf_train = random_action_monte_carlo_runner_impl(functions, dimension, n_times,
                                                                  key_type, output_directory + "/training")
            _, perf_test = random_action_monte_carlo_runner_impl(complement_functions, dimension, n_times,
                                                                 key_type, output_directory + "/test")

            perf_mean_train, perf_deviance_train = runner_overall_performance(perf_train)
            perf_mean_test, perf_deviance_test = runner_overall_performance(perf_test)

            performance_mean_variance["perf_mean_train"] += perf_mean_train
            performance_mean_variance["perf_deviance_train"] += perf_deviance_train
            performance_mean_variance["perf_mean_test"] += perf_mean_test
            performance_mean_variance["perf_deviance_test"] += perf_deviance_test
        else:
            _, perf = random_action_monte_carlo_runner_impl(functions, dimension, n_times, key_type, output_directory)
            perf_mean, perf_deviance = runner_overall_performance(perf)
            performance_mean_variance["perf_mean"] += perf_mean
            performance_mean_variance["perf_deviance"] += perf_deviance

        print("Monte Carlo times:" + str(times + 1) + "/" + str(monte_carlo_times))

    performance_mean_variance.update((key, value / monte_carlo_times)
                                     for key, value in performance_mean_variance.items())
    dump_json(performance_mean_variance, root_directory, "performance_mean_variance")
    dump_json(parameters_dict, root_directory, "parameters")
    return root_directory


def random_action_monte_carlo_runner_impl(functions, dimension, n_times, key_type, output_directory):
    env = random_action_runner_n_times(functions, dimension, n_times, key_type=key_type)
    performance_results = random_action_runner_output(output_directory, env)
    return env, performance_results


def random_action_runner_output_helper(root_directory, output_folder_label, env):
    output_directory = get_test_output_directory(root_directory, output_folder_label, "RandomAction", env)
    if output_directory is not None:
        random_action_runner_output(output_directory, env)


def random_action_runner_output(output_directory, env):
    dump_json(env.function_each_episode, output_directory, "function_each_episode")
    dump_json(env.max_reward_dict, output_directory, "max_reward_dict")
    dump_json(env.max_reward_key_dict, output_directory, "max_reward_" + env.key_name + "_dict")

    performance_results = {}
    for function, reward in env.max_reward_dict.items():
        performance_results[function] = reward_performance(env, reward, function)

    dump_json(performance_results, output_directory, "performance_results")

    return performance_results
