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
                env = random_action_monte_carlo_impl(functions, dimension, time_point,
                                                     output_directory + "/training", key_type,
                                                     environments[times])
                environments[times] = env

                env = random_action_monte_carlo_impl(functions, dimension, time_point,
                                                     output_directory + "/test", key_type,
                                                     test_environments[times])
                test_environments[times] = env
            else:
                env = random_action_monte_carlo_impl(functions, dimension, time_point, output_directory,
                                                     key_type, environments[times])
                environments[times] = env

            print("Monte Carlo times:" + str(times + 1) + "/" + str(monte_carlo_times))

        dump_json(parameters_dict, root_directory, "parameters")
        monte_carlo_overall_performance_average(root_directory, test_mode, monte_carlo_times)
        monte_carlo_equivalence_class_performance_average(root_directory, test_mode, monte_carlo_times, dimension)

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


def random_action_monte_carlo_impl(functions, dimension, n_times, output_directory, key_type, input_env):
    output_env = random_action_runner_n_times(functions, dimension, n_times, input_env, key_type=key_type)
    random_action_runner_output(output_directory, output_env)
    return output_env


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
    dump_json(runner_equivalence_class_performance(performance_results, env.dimension), output_directory,
              "performance_mean_variance_equivalence_classes")

    perf_mean, perf_deviance = runner_overall_performance(performance_results)
    performance_mean_variance = {"perf_mean": perf_mean, "perf_deviance": perf_deviance}
    dump_json(performance_mean_variance, output_directory, "performance_mean_variance")


def monte_carlo_overall_performance_average(root_directory, test_mode, monte_carlo_times):
    performance_mean_variance = {"perf_mean_train": 0, "perf_deviance_train": 0, "perf_mean_test": 0,
                                 "perf_deviance_test": 0, "perf_mean": 0, "perf_deviance": 0}
    if monte_carlo_times <= 0:
        raise Exception("Monte carlo times variable is abnormal.")

    for monte_carlo_directory in range(monte_carlo_times):
        if not test_mode:
            monte_carlo_performance_accumulator(root_directory + "/" + str(monte_carlo_directory),
                                                "performance_mean_variance", performance_mean_variance,
                                                "perf_mean", "perf_deviance")
        else:
            monte_carlo_performance_accumulator(root_directory + "/" + str(monte_carlo_directory) + "/training",
                                                "performance_mean_variance", performance_mean_variance,
                                                "perf_mean_train", "perf_deviance_train")
            monte_carlo_performance_accumulator(root_directory + "/" + str(monte_carlo_directory) + "/test",
                                                "performance_mean_variance", performance_mean_variance,
                                                "perf_mean_test", "perf_deviance_test")

    performance_mean_variance.update((key, value / monte_carlo_times)
                                     for key, value in performance_mean_variance.items())
    dump_json(performance_mean_variance, root_directory, "performance_mean_variance")


def monte_carlo_performance_accumulator(root_directory, json_file, performance_mean_variance, mean_key, deviance_key):
    local_performance_mean_variance = load_json(root_directory, json_file)
    performance_mean_variance[mean_key] += local_performance_mean_variance["perf_mean"]
    performance_mean_variance[deviance_key] += local_performance_mean_variance["perf_deviance"]


def monte_carlo_equivalence_class_performance_average(root_directory, test_mode, monte_carlo_times, dimension):
    equivalence_class_keys = all_equivalence_classes_hex_string(dimension)
    equivalence_class_performance_mean_variance = {}
    equivalence_class_performance_mean_variance_test = {}
    equivalence_class_performance_average_factor = {}
    equivalence_class_performance_average_factor_test = {}
    for equivalence_class_key in equivalence_class_keys:
        equivalence_class_performance_mean_variance[equivalence_class_key] = [0, 0]
        equivalence_class_performance_mean_variance_test[equivalence_class_key] = [0, 0]
        equivalence_class_performance_average_factor[equivalence_class_key] = 0
        equivalence_class_performance_average_factor_test[equivalence_class_key] = 0

    if monte_carlo_times <= 0:
        raise Exception("Monte carlo times variable is abnormal.")

    for monte_carlo_directory in range(monte_carlo_times):
        if not test_mode:
            monte_carlo_equivalence_class_accumulator(root_directory + "/" + str(monte_carlo_directory),
                                                      "performance_mean_variance_equivalence_classes",
                                                      equivalence_class_performance_mean_variance,
                                                      equivalence_class_performance_average_factor)
        else:
            monte_carlo_equivalence_class_accumulator(root_directory + "/" + str(monte_carlo_directory) + "/training",
                                                      "performance_mean_variance_equivalence_classes",
                                                      equivalence_class_performance_mean_variance,
                                                      equivalence_class_performance_average_factor)
            monte_carlo_equivalence_class_accumulator(root_directory + "/" + str(monte_carlo_directory) + "/test",
                                                      "performance_mean_variance_equivalence_classes",
                                                      equivalence_class_performance_mean_variance_test,
                                                      equivalence_class_performance_average_factor_test)

    for key in equivalence_class_keys:
        factor = equivalence_class_performance_average_factor[key]
        if factor > 0:
            equivalence_class_performance_mean_variance[key][0] /= factor
            equivalence_class_performance_mean_variance[key][1] /= factor

        if test_mode:
            factor = equivalence_class_performance_average_factor_test[key]
            if factor > 0:
                equivalence_class_performance_mean_variance_test[key][0] /= factor
                equivalence_class_performance_mean_variance_test[key][1] /= factor

    dump_json(equivalence_class_performance_mean_variance, root_directory,
              "equivalence_class_performance_mean_variance")
    if test_mode:
        dump_json(equivalence_class_performance_mean_variance_test, root_directory,
                  "equivalence_class_performance_mean_variance_test")


def monte_carlo_equivalence_class_accumulator(root_directory, json_file, equivalence_class_performance_mean_variance,
                                              equivalence_class_performance_average_factor):
    local_equivalence_class_mean_variance = load_json(root_directory, json_file)

    for key in equivalence_class_performance_mean_variance.keys():
        if key in local_equivalence_class_mean_variance and 1 >= local_equivalence_class_mean_variance[key][0] >= 0:
            equivalence_class_performance_mean_variance[key][0] += local_equivalence_class_mean_variance[key][0]
            equivalence_class_performance_mean_variance[key][1] += local_equivalence_class_mean_variance[key][1]
            equivalence_class_performance_average_factor[key] += 1
