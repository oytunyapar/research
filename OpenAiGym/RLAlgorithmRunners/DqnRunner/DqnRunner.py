from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import dump_outputs, dump_json
from OpenAiGym.RLAlgorithmRunners.Utils.StringHelperFunctions import function_to_hex_string
import torch as th
from stable_baselines3 import DQN
import datetime

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[64, 32]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[128, 64]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[256, 128])
}


def dqn_runner(dimension, output_directory=None, function_begin_end_indexes=None):
    if function_begin_end_indexes is None:
        functions = BooleanFunctionsEquivalentClasses[dimension]
    else:
        if len(function_begin_end_indexes) != 2:
            print("Size of function_begin_end_indexes must be 2")
            return
        else:
            begin_index = function_begin_end_indexes[0]
            end_index = function_begin_end_indexes[1]

            all_functions = BooleanFunctionsEquivalentClasses[dimension]

            if begin_index >= end_index or begin_index < 0 or end_index > len(all_functions):
                print("Check function_begin_end_indexes. There is a problem")
                return

            functions = all_functions[begin_index:end_index]

    result_metrics = {}
    envs = {}
    for function in functions:
        env = MinTermSrpobfEnv(function, dimension, q_matrix_representation, act,
                               no_action_episode_end, episodic_reward=episodic_reward)
        model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension], verbose=1)
        model.learn(total_timesteps=number_of_steps_dictionary[dimension])
        result_metrics[str(dimension) + "_" + function_to_hex_string(dimension, function) + "_max_reward"] = \
            env.max_rewards_in_the_episodes
        result_metrics[str(dimension) + "_" + function_to_hex_string(dimension, function) + "_episode_total_reward"] =\
            env.cumulative_rewards_in_the_episodes
        envs[str(dimension) + "_" + function_to_hex_string(dimension, function)] = env

        if output_directory is not None:
            function_output_directory = output_directory + "/" + function_to_hex_string(dimension, function)

            dump_outputs(env.max_rewards_in_the_episodes, function_output_directory, "max_rewards_in_the_episodes")
            dump_outputs(env.cumulative_rewards_in_the_episodes, function_output_directory,
                         "cumulative_rewards_in_the_episodes")

    return result_metrics, envs


def dqn_runner_equivalent_functions(dimension, output_directory=None):
    functions = BooleanFunctionsEquivalentClasses[dimension]
    env = MinTermSrpobfEnv(functions, dimension, q_matrix_representation, act,
                           no_action_episode_end, episodic_reward=episodic_reward)
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension],
                exploration_fraction=0.9, batch_size=int(env.steps_in_each_epoch*2), verbose=0,
                learning_rate=0.01)
    model.learn(total_timesteps=number_of_steps_dictionary_all_equivalent_functions[dimension])

    test_results = {}

    for function in functions:
        max_reward = dqn_runner_test_model(env, model, function)
        test_results[function] = max_reward

    dqn_runner_output_helper(output_directory, "dimension_equivalent_functions", env, model, test_results)

    return env, model


def dqn_runner_all_functions(dimension, output_directory=None):
    all_functions = 2 ** (2 ** dimension)
    env = MinTermSrpobfEnv(all_functions, dimension, q_matrix_representation, act,
                           no_action_episode_end, episodic_reward=episodic_reward)
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension], verbose=1)
    model.learn(total_timesteps=number_of_steps_dictionary_all_functions[dimension])

    test_results = {}

    for function in range(all_functions):
        max_reward = dqn_runner_test_model(env, model, function)
        test_results[function] = max_reward

    dqn_runner_output_helper(output_directory, "dimension_all_functions", env, model, test_results)

    return env, model


def dqn_runner_output_helper(root_directory, dump_directory_prefix, env, model, test_results):
    if root_directory is not None:
        function_output_directory = root_directory + "/" + str(env.dimension) + \
                                    dump_directory_prefix + "_" + str(datetime.datetime.now())

        dump_outputs(env.max_rewards_in_the_episodes, function_output_directory, "max_rewards_in_the_episodes")
        dump_outputs(env.cumulative_rewards_in_the_episodes, function_output_directory,
                     "cumulative_rewards_in_the_episodes")

        dump_json(env.function_each_episode, function_output_directory, "function_each_episode")
        dump_json(env.max_reward_dict, function_output_directory, "max_reward_dict")
        dump_json(env.max_reward_k_vector_dict, function_output_directory, "max_reward_k_vector_dict")
        dump_json(test_results, function_output_directory, "test_results")
        model.save(function_output_directory + "/" + "model")


def dqn_runner_test_model(env, model, function=None):
    env.switch_to_single_mode()

    if function is not None:
        env.set_function(function)

    obs = env.reset()
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    return env.max_reward_in_the_episode
