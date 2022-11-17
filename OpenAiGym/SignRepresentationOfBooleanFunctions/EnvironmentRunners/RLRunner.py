from OpenAiGym.Utils.DumpOutputs import dump_json
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.Utils.EnvironmentHelperFunctions import *
from OpenAiGym.SignRepresentationOfBooleanFunctions.EnvironmentRunners.Utils.DataHelperFunctions import *
from SigmaPiFrameworkPython.Utils.BooleanFunctionUtils import *
import torch as th
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import numpy
from enum import Enum


class RLModelType(Enum):
    RL_DQN = 1
    RL_PPO = 2
    RL_A2C = 3


policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[16, 8]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[64, 32]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[256, 128]),
    6: dict(activation_fn=th.nn.ReLU, net_arch=[128, 64])
}


def rl_create_model(model_type, env):
    batch_factor = 128
    batch_size = env.steps_in_each_epoch * batch_factor

    buffer_factor = 100
    buffer_size = batch_size * buffer_factor

    if model_type is RLModelType.RL_DQN:
        model = DQN('MlpPolicy', env,
                    policy_kwargs=policy_kwargs_dictionary[env.dimension],
                    verbose=1,
                    exploration_final_eps=0.3,
                    exploration_fraction=0.8,
                    batch_size=batch_size,
                    buffer_size=buffer_size)
    elif model_type is RLModelType.RL_PPO:
        model = PPO('MlpPolicy', make_vec_env(lambda: env, n_envs=4),
                    policy_kwargs=policy_kwargs_dictionary[env.dimension],
                    verbose=1,
                    batch_size=batch_size)
    elif model_type is RLModelType.RL_A2C:
        model = A2C('MlpPolicy', make_vec_env(lambda: env, n_envs=4),
                    policy_kwargs=policy_kwargs_dictionary[env.dimension],
                    verbose=1,
                    batch_size=batch_size)
    else:
        raise Exception("Undefined model type.")

    return model


def rl_runner_functions(functions,
                        dimension,
                        time_steps,
                        model_type=RLModelType.RL_DQN,
                        output_directory=None,
                        output_folder_prefix=None,
                        key_type=KeyType.K_VECTOR,
                        model=None,
                        test_functions=None,
                        ):

    env = env_creator(functions, dimension, key_type)

    parameters_dict = env.env_specific_configuration() | {"time_steps": time_steps,
                                                          "net_arch": policy_kwargs_dictionary[dimension]["net_arch"],
                                                          "functions": functions, "test_functions": test_functions}

    if model is None:
        model = rl_create_model(model_type, env)
    else:
        model.set_env(env)

    model.learn(total_timesteps=time_steps)

    training_data_performance_results = rl_runner_model_performance(env, model, functions)

    test_data_performance_results = {}
    if test_functions is not None:
        test_data_performance_results = rl_runner_model_performance(env, model, test_functions)

    rl_runner_output_helper(output_directory, output_folder_prefix, env, model, parameters_dict,
                            training_data_performance_results, test_data_performance_results)

    return env, model


def rl_runner_equivalent_functions(dimension, output_directory=None, key_type=KeyType.K_VECTOR, model=None):
    functions = BooleanFunctionsEquivalentClasses[dimension]
    time_steps = number_of_steps_dictionary_all_equivalent_functions[dimension]
    return rl_runner_functions(functions,
                               dimension,
                               time_steps,
                               output_directory,
                                "dimension_equivalent_functions",
                               key_type,
                               model)


def rl_runner_all_functions(dimension, output_directory=None, key_type=KeyType.K_VECTOR, model=None):
    all_functions = 2 ** (2 ** dimension)
    time_steps = number_of_steps_dictionary_all_functions[dimension]
    return rl_runner_functions(all_functions,
                               dimension,
                               time_steps,
                               output_directory,
                                "dimension_all_functions",
                               key_type,
                               model)


def rl_runner_output_helper(root_directory, output_folder_label, env, model, parameters_dict,
                            training_data_performance_results=None, test_data_performance_results=None):
    output_directory = get_experiment_output_directory(root_directory, output_folder_label, type(model).__name__, env)
    if output_directory is not None:
        env.dump_env_statistics(output_directory)
        dump_json(parameters_dict, output_directory, "parameters")

        dump_json(training_data_performance_results, output_directory, "training_data_performance_results")
        dump_json(test_data_performance_results, output_directory, "test_data_performance_results")

        if env.dimension < 6:
            dump_json(runner_equivalence_class_performance(training_data_performance_results, env.dimension),
                      output_directory, "training_data_performance_results_equivalence_classes")
            perf_mean_train, perf_deviance_train = runner_overall_performance(training_data_performance_results)
            performance_mean_variance_train = \
                {"perf_mean_train": perf_mean_train, "perf_deviance_train": perf_deviance_train}
            dump_json(performance_mean_variance_train, output_directory, "performance_mean_variance_train")

            dump_json(runner_equivalence_class_performance(test_data_performance_results, env.dimension),
                      output_directory, "test_data_performance_results_equivalence_classes")
            perf_mean_test, perf_deviance_test = runner_overall_performance(test_data_performance_results)
            performance_mean_variance_test = {"perf_mean_test": perf_mean_test,
                                              "perf_deviance_test": perf_deviance_test}
            dump_json(performance_mean_variance_test, output_directory, "performance_mean_variance_test")

        model.save(output_directory + "/" + "model")


def rl_runner_test_model(env, model, function=None):
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


def rl_runner_model_performance(env, model, functions):
    functions_unique = numpy.unique(functions).tolist()
    performance = {}

    for function in functions_unique:
        reward, _ = rl_runner_test_model(env, model, function)
        performance[function] = reward_performance(env, reward)

    return performance


def rl_load_model(output_directory, model_package_name="model"):
    return DQN.load(output_directory + "/" + model_package_name)
