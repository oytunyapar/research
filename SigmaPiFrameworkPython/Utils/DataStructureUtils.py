import pickle


def open_data_structure(directory, file_name):
    with open(directory + "/" + file_name + ".pkl", 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def save_data_structure(directory, file_name, data_structure):
    with open(directory + "/" + file_name + ".pkl", 'wb') as f:
        pickle.dump(data_structure, f)
