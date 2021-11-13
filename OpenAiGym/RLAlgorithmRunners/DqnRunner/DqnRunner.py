from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *
import torch as th
from stable_baselines3 import DQN

from pathlib import Path
import matplotlib.pyplot as plt
import json

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[250, 250]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[500, 500]),
    5: dict(activation_fn=th.nn.ReLU, net_arch=[500, 500])
}


def dump_outputs(data, output_directory, file_name):
    plt.plot(data)
    plt.savefig(output_directory + "/" + file_name + ".png")
    plt.clf()

    json_file_name = output_directory + "/" + file_name + ".json"
    json_handle = json.dumps(data)
    f = open(json_file_name, "w")
    f.write(json_handle)
    f.close()


def dqn_runner(dimension, output_directory, function_begin_end_indexes=None):

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

    if not Path(output_directory).is_dir():
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    result_metrics = {}
    for function in functions:
        env = MinTermSrpobfEnv(function, dimension, q_matrix_representation, act,
                               no_action_episode_end, episodic_reward=episodic_reward)
        model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension], verbose=1)
        model.learn(total_timesteps=number_of_steps_dictionary[dimension])
        result_metrics[str(dimension) + "_" + hex(function) + "_max_reward"] = env.max_rewards_in_the_episodes
        result_metrics[str(dimension) + "_" + hex(function) + "_episode_total_reward"] =\
            env.cumulative_rewards_in_the_episodes

        function_output_directory = output_directory + "/" + hex(function)
        Path(function_output_directory).mkdir(parents=True, exist_ok=True)

        dump_outputs(env.max_rewards_in_the_episodes, function_output_directory, "max_rewards_in_the_episodes")
        dump_outputs(env.cumulative_rewards_in_the_episodes, function_output_directory,
                     "cumulative_rewards_in_the_episodes")

    return result_metrics

