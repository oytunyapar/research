from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnv import ActionType
from OpenAiGym.SignRepresentationOfBooleanFunctions.Environments.MinTermSrpobfEnvBase import FunctionRepresentationType

function_representation_type = FunctionRepresentationType.WALSH_SPECTRUM
no_action_episode_end = False
act = ActionType.INCREASE
number_of_steps_dictionary = {3: 50000, 4: 2000000, 5: 4000000}
number_of_steps_dictionary_all_equivalent_functions = {3: 200000, 4: 24000000, 5: 64000000}
number_of_steps_dictionary_all_functions = {3: 4000000, 4: 256000000, 5: 2048000000}
episodic_reward = False
