from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import *
from OpenAiGym.RLAlgorithmRunners.Utils.EnvironmentHelperFunctions import *
from OpenAiGym.RLAlgorithmRunners.Utils.DataHelperFunctions import *
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import *
import datetime
from pathlib import Path
import warnings


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
                                 env,
                                 output_directory=None,
                                 output_folder_label=None,
                                 key_type=KeyType.K_VECTOR
                                 ):
    if env is None:
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
    warnings.filterwarnings("ignore")

    complement_functions = get_complement_function_list(dimension, functions)

    environments = dict.fromkeys(list(range(monte_carlo_times)), None)
    test_environments = None

    test_mode = False
    if len(complement_functions) > 0:
        test_environments = dict.fromkeys(list(range(monte_carlo_times)), None)
        test_mode = True

    performance_mean_variance = {"perf_mean_train": 0, "perf_deviance_train": 0, "perf_mean_test": 0,
                                 "perf_deviance_test": 0, "perf_mean": 0, "perf_deviance": 0}

    n_time_points = process_n_times_array(n_times)

    parameters_dict = {"monte_carlo_times": monte_carlo_times, "n_times": 0, "dimension": dimension,
                       "functions": functions}

    for time_point in n_time_points:
        parameters_dict["n_times"] += time_point

        root_directory = str(Path.home()) + "/PycharmProjects/research/OpenAiGym/" +\
                         get_env_name_from_key_type(key_type) + "/Data/" + str(dimension) +\
                         "dim/RandomAction/" + str(datetime.datetime.now()) + "_" + "Monte_Carlo"

        if output_folder_label is not None:
            root_directory = root_directory + "_" + output_folder_label

        for times in range(monte_carlo_times):
            output_directory = root_directory + "/" + str(times)
            if test_mode:
                env = random_action_monte_carlo_statistics(functions, dimension, time_point,
                                                           output_directory + "/training", key_type,
                                                           performance_mean_variance, "perf_mean_train",
                                                           "perf_deviance_train", environments[times])
                environments[times] = env

                env = random_action_monte_carlo_statistics(functions, dimension, time_point,
                                                           output_directory + "/test", key_type,
                                                           performance_mean_variance, "perf_mean_test",
                                                           "perf_deviance_test", test_environments[times])
                test_environments[times] = env
            else:
                env = random_action_monte_carlo_statistics(functions, dimension, time_point, output_directory,
                                                           key_type, performance_mean_variance, "perf_mean",
                                                           "perf_deviance", environments[times])
                environments[times] = env

            print("Monte Carlo times:" + str(times + 1) + "/" + str(monte_carlo_times))

        performance_mean_variance.update((key, value / monte_carlo_times)
                                         for key, value in performance_mean_variance.items())
        dump_json(performance_mean_variance, root_directory, "performance_mean_variance")
        dump_json(parameters_dict, root_directory, "parameters")

        performance_mean_variance.update((key, 0)
                                         for key, value in performance_mean_variance.items())

    warnings.filterwarnings("default")

    return root_directory


def process_n_times_array(n_times):
    len_n_times = len(n_times)
    if len_n_times == 0:
        raise Exception("process_n_times_array list is empty")

    n_times_processed = numpy.unique(list(map(abs, n_times))).tolist()
    len_n_times = len(n_times_processed)

    result = len_n_times * [None]
    subtraction_value = n_times_processed[0]
    result[0] = subtraction_value

    for index in range(1, len_n_times):
        result[index] = n_times_processed[index] - subtraction_value
        subtraction_value = n_times_processed[index]

    return result


def random_action_monte_carlo_statistics(functions, dimension, n_times, output_directory, key_type,
                                         performance_mean_variance, mean_key, deviance_key, input_env):
    output_env, perf = random_action_monte_carlo_impl(functions, dimension, n_times,
                                                      key_type, output_directory, input_env)
    perf_mean, perf_deviance = runner_overall_performance(perf)
    performance_mean_variance[mean_key] += perf_mean
    performance_mean_variance[deviance_key] += perf_deviance
    return output_env


def random_action_monte_carlo_impl(functions, dimension, n_times, key_type, output_directory, input_env):
    output_env = random_action_runner_n_times(functions, dimension, n_times, input_env, key_type=key_type)
    performance_results = random_action_runner_output(output_directory, output_env)
    return output_env, performance_results


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
