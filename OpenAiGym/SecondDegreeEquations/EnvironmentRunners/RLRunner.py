import torch as th
from stable_baselines3 import DQN

from OpenAiGym.SecondDegreeEquations.EnvironmentRunners.Utils.HelperFunctions import *
from OpenAiGym.SecondDegreeEquations.Environments.ValueAdjustingEnv import ValueAdjustingEnv


def rl_create_model(env, time_steps):
    batch_factor = 32
    buffer_factor = 128

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[16, 8]),
                verbose=1,
                exploration_final_eps=0.3,
                exploration_fraction=0.8,
                batch_size=env.steps_in_each_epoch * batch_factor,
                buffer_size=int(time_steps / buffer_factor),
                learning_rate=0.01)

    return model


def rl_runner(root_limits, time_steps, output_directory=None, output_folder_prefix=None, model=None):

    env = ValueAdjustingEnv(root_limits)

    if model is None:
        model = rl_create_model(env, time_steps)
    else:
        model.set_env(env)

    model.learn(total_timesteps=time_steps)
    rl_runner_output_helper(output_directory, output_folder_prefix, env, model)

    return env, model


def rl_runner_test_model(env, model):
    observation = env.reset()
    done = False

    actions = []

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        actions.append(action)
        obs, reward, done, info = env.step(action)

    return actions


def rl_runner_output_helper(root_directory, output_folder_label, env, model):
    output_directory = get_experiment_output_directory(root_directory, output_folder_label, env)
    if output_directory is not None:
        model.save(output_directory + "/" + "model")


def rl_load_model(output_directory, model_package_name="model.zip"):
    return DQN.load(output_directory + "/" + model_package_name)
