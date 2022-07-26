import datetime


def get_experiment_output_directory(root_directory, output_folder_label, env):
    output_directory = None
    if root_directory is not None and output_folder_label is not None:
        output_directory = root_directory + "/Data/OpenAiGym/SecondDegreeEquations/" + \
                           type(env).__name__ + "/" + str(datetime.datetime.now()) + "_" + output_folder_label

    return output_directory
