from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import dump_outputs, dump_json
from OpenAiGym.RLAlgorithmRunners.Utils.EnvironmentHelperFunctions import *
from OpenAiGym.RLAlgorithmRunners.Utils.DataHelperFunctions import *
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import *
import torch as th
from stable_baselines3 import DQN
import numpy

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[64, 32]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[256, 128]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[256, 128])
}


def dqn_runner_functions(functions,
                         dimension,
                         time_steps,
                         output_directory=None,
                         output_folder_prefix=None,
                         key_type=KeyType.K_VECTOR,
                         model=None,
                         test_functions=None,
                         ):

    parameters_dict = {"time_steps": time_steps,
                       "net_arch": policy_kwargs_dictionary[dimension]["net_arch"],
                       "dimension": dimension, "functions": functions}

    env = env_creator(functions, dimension, key_type)
    '''model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension],
                exploration_fraction=0.9, batch_size=int(env.steps_in_each_epoch*2), verbose=1,
                learning_rate=0.01)'''

    buffer_factor = 128
    batch_factor = 32

    if model is None:
        model = DQN('MlpPolicy', env,
                    policy_kwargs=policy_kwargs_dictionary[dimension],
                    verbose=1,
                    batch_size=env.steps_in_each_epoch * batch_factor,
                    buffer_size=int(time_steps/buffer_factor))
    else:
        model.set_env(env)

    model.learn(total_timesteps=time_steps)

    training_data_performance_results = dqn_runner_model_performance(env, model, functions)

    test_data_performance_results = {}
    if test_functions is not None:
        test_data_performance_results = dqn_runner_model_performance(env, model, test_functions)

    dqn_runner_output_helper(output_directory, output_folder_prefix, env, model,
                             training_data_performance_results, test_data_performance_results, parameters_dict)

    return env, model


def dqn_runner_equivalent_functions(dimension, output_directory=None, key_type=KeyType.K_VECTOR, model=None):
    functions = BooleanFunctionsEquivalentClasses[dimension]
    time_steps = number_of_steps_dictionary_all_equivalent_functions[dimension]
    return dqn_runner_functions(functions,
                                dimension,
                                time_steps,
                                output_directory,
                                "dimension_equivalent_functions",
                                key_type,
                                model)


def dqn_runner_all_functions(dimension, output_directory=None, key_type=KeyType.K_VECTOR, model=None):
    all_functions = 2 ** (2 ** dimension)
    time_steps = number_of_steps_dictionary_all_functions[dimension]
    return dqn_runner_functions(all_functions,
                                dimension,
                                time_steps,
                                output_directory,
                                "dimension_all_functions",
                                key_type,
                                model)


def dqn_runner_output_helper(root_directory, output_folder_label, env, model,
                             training_data_performance_results, test_data_performance_results, parameters_dict):
    output_directory = get_test_output_directory(root_directory, output_folder_label, "DQN", env)
    if output_directory is not None:
        dump_outputs(env.max_rewards_in_the_episodes, output_directory, "max_rewards_in_the_episodes")
        dump_outputs(env.cumulative_rewards_in_the_episodes, output_directory,
                     "cumulative_rewards_in_the_episodes")

        dump_json(env.function_each_episode, output_directory, "function_each_episode")
        dump_json(env.max_reward_dict, output_directory, "max_reward_dict")
        dump_json(env.max_reward_key_dict, output_directory, "max_reward_" + env.key_name + "_dict")
        dump_json(training_data_performance_results, output_directory, "training_data_performance_results")
        dump_json(test_data_performance_results, output_directory, "test_data_performance_results")

        perf_mean_train, perf_deviance_train = runner_overall_performance(training_data_performance_results)
        perf_mean_test, perf_deviance_test = runner_overall_performance(test_data_performance_results)
        performance_mean_variance = {"perf_mean_train": perf_mean_train, "perf_deviance_train": perf_deviance_train,
                                     "perf_mean_test": perf_mean_test, "perf_deviance_test": perf_deviance_test}

        dump_json(performance_mean_variance, output_directory, "performance_mean_variance")

        dump_json(runner_equivalence_class_performance(training_data_performance_results, env.dimension),
                  output_directory, "training_data_performance_results_equivalence_classes")
        dump_json(runner_equivalence_class_performance(test_data_performance_results, env.dimension),
                  output_directory, "test_data_performance_results_equivalence_classes")

        dump_json(parameters_dict, output_directory, "parameters")
        model.save(output_directory + "/" + "model")


def dqn_runner_test_model(env, model, function=None):
    env.switch_to_single_mode()

    if function is not None:
        env.set_function(function)

    obs = env.reset()
    done = False

    actions = []

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = env.step(action)

    return env.max_reward_in_the_episode, actions


def dqn_runner_model_performance(env, model, functions):
    functions_unique = numpy.unique(functions).tolist()
    performance = {}

    for function in functions_unique:
        reward, _ = dqn_runner_test_model(env, model, function)
        performance[function] = reward_performance(env, reward)

    return performance


def dqn_load_model(output_directory, model_package_name="model.zip"):
    return DQN.load(output_directory + "/" + model_package_name)
