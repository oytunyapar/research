from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
from OpenAiGym.MinTermLpSrpobfEnv.MinTermLpSrpobfEnv import MinTermLpSrpobfEnv
from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
from OpenAiGym.RLAlgorithmRunners.Utils.DumpOutputs import dump_outputs, dump_json
from OpenAiGym.RLAlgorithmRunners.Utils.StringHelperFunctions import function_to_hex_string
import torch as th
from stable_baselines3 import DQN
import datetime
from enum import Enum

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[64, 32]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[128, 64]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[256, 128])
}


class KeyType(Enum):
    K_VECTOR = 1
    MONOMIAL_SET = 2


def env_creator(function, dimension, key_type):
    if key_type == KeyType.K_VECTOR:
        return MinTermSrpobfEnv(function, dimension, q_matrix_representation,
                                act, no_action_episode_end, episodic_reward=episodic_reward)
    elif key_type == KeyType.MONOMIAL_SET:
        return MinTermLpSrpobfEnv(function, dimension, q_matrix_representation, episodic_reward=episodic_reward)
    else:
        raise Exception("Unsupported env type")


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


def dqn_runner_functions(functions,
                         dimension,
                         time_steps,
                         output_directory=None,
                         output_folder_prefix=None,
                         key_type=KeyType.K_VECTOR,
                         model=None):
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

    test_results = {}

    for function in functions:
        max_reward, _ = dqn_runner_test_model(env, model, function)
        test_results[function] = max_reward

    dqn_runner_output_helper(output_directory, output_folder_prefix, env, model, test_results)

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


def dqn_runner_output_helper(root_directory, dump_directory_prefix, env, model, test_results):
    if root_directory is not None:
        function_output_directory = root_directory + "/" + str(env.dimension) + \
                                    dump_directory_prefix + "_" + str(datetime.datetime.now())

        dump_outputs(env.max_rewards_in_the_episodes, function_output_directory, "max_rewards_in_the_episodes")
        dump_outputs(env.cumulative_rewards_in_the_episodes, function_output_directory,
                     "cumulative_rewards_in_the_episodes")

        dump_json(env.function_each_episode, function_output_directory, "function_each_episode")
        dump_json(env.max_reward_dict, function_output_directory, "max_reward_dict")
        dump_json(env.max_reward_key_dict, function_output_directory, "max_reward_" + env.key_name + "_dict")
        dump_json(test_results, function_output_directory, "test_results")
        model.save(function_output_directory + "/" + "model")


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


def dqn_load_model(output_directory, model_package_name="model.zip"):
    return DQN.load(output_directory + "/" + model_package_name)
