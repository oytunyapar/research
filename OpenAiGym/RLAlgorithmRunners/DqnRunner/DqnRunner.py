from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import dump_outputs, dump_json
from OpenAiGym.RLAlgorithmRunners.Utils.StringHelperFunctions import function_to_hex_string
import torch as th
from stable_baselines3 import DQN

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[250, 250]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[500, 500]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[500, 500])
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


def dqn_runner_all_functions(dimension, output_directory=None):
    all_functions = (2 ** dimension) ** dimension
    env = MinTermSrpobfEnv(all_functions, dimension, q_matrix_representation, act,
                           no_action_episode_end, episodic_reward=episodic_reward)
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension], verbose=1)
    model.learn(total_timesteps=number_of_steps_dictionary_all_functions[dimension])

    if output_directory is not None:
        function_output_directory = output_directory + "/" + str(dimension) + "dimension_all_functions"

        dump_outputs(env.max_rewards_in_the_episodes, function_output_directory, "max_rewards_in_the_episodes")
        dump_outputs(env.cumulative_rewards_in_the_episodes, function_output_directory,
                     "cumulative_rewards_in_the_episodes")

        dump_json(env.function_each_episode, function_output_directory, "function_each_episode")
        dump_json(env.max_reward_dict, function_output_directory, "max_reward_dict")
        dump_json(env.max_reward_k_vector_dict, function_output_directory, "max_reward_k_vector_dict")

    return env
