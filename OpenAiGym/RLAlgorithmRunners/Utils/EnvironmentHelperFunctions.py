from enum import Enum
import datetime

from OpenAiGym.MinTermLpSrpobfEnv.MinTermLpSrpobfEnv import MinTermLpSrpobfEnv
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv
from OpenAiGym.RLAlgorithmRunners.MinTermSrpobfEnvConstants import *


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


def get_env_name_from_key_type(key_type):
    if key_type == KeyType.K_VECTOR:
        return "MinTermSrpobfEnv"
    elif key_type == KeyType.MONOMIAL_SET:
        return "MinTermLpSrpobfEnv"
    else:
        raise Exception("Unsupported env type")


def get_test_output_directory(root_directory, output_folder_label, algorithm, env):
    output_directory = None
    if root_directory is not None and output_folder_label is not None:
        output_directory = root_directory + "/" + \
                           type(env).__name__ + "/Data/" + str(env.dimension) + "dim/" + algorithm + "/" + \
                           str(datetime.datetime.now()) + "_" + output_folder_label
    return output_directory