from BooleanFunctionsEquivalentClasses.BooleanFunctionsEquivalentClasses import BooleanFunctionsEquivalentClasses
from OpenAiGym.MinTermSrpobfEnv.MinTermSrpobfEnv import MinTermSrpobfEnv, ActionType
import torch as th
from stable_baselines3 import DQN

policy_kwargs_dictionary = {
    3: dict(activation_fn=th.nn.ReLU, net_arch=[250, 250]),
    4: dict(activation_fn=th.nn.ReLU, net_arch=[500, 500]),
}

act = ActionType.INCREASE

number_of_steps_dictionary = {3: 400000, 4: 2000000}


def dqn_runner(dimension, episodic_reward):
    result_metrics = {}
    for function in BooleanFunctionsEquivalentClasses[dimension]:
        env = MinTermSrpobfEnv(function, dimension, False, act, False, episodic_reward=episodic_reward)
        model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs_dictionary[dimension], verbose=1)
        model.learn(total_timesteps=number_of_steps_dictionary[dimension])
        result_metrics[str(dimension) + "_" + str(function) + "_max_reward"] = env.max_rewards_in_the_episodes
        result_metrics[str(dimension) + "_" + str(function) + "_episode_total_reward"] =\
            env.cumulative_rewards_in_the_episodes

    return result_metrics

