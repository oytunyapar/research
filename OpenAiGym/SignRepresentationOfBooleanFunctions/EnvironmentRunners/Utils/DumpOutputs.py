import matplotlib.pyplot as plt
import json
from pathlib import Path


def dump_outputs(data, output_directory, file_name_prefix):
    dump_png(data, output_directory, file_name_prefix)
    dump_json(data, output_directory, file_name_prefix)


def dump_png(data, output_directory, file_name_prefix):
    if not Path(output_directory).is_dir():
        Path(output_directory).mkdir(parents=True)

    plt.plot(data)
    plt.savefig(output_directory + "/" + file_name_prefix + ".png")
    plt.clf()


def dump_json(data, output_directory, file_name_prefix):
    if not Path(output_directory).is_dir():
        Path(output_directory).mkdir(parents=True)

    json_file_name = output_directory + "/" + file_name_prefix + ".json"
    json_handle = json.dumps(data)
    f = open(json_file_name, "w")
    f.write(json_handle)
    f.close()


def load_json(output_directory, file_name_prefix):
    json_file_name = output_directory + "/" + file_name_prefix + ".json"

    f = open(json_file_name)
    data = json.load(f)
    f.close()

    return data
