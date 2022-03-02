from enum import Enum

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
