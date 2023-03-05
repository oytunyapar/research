import matplotlib.pyplot as plt
import json
import csv
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


def dump_csv(fields, rows, output_directory, file_name_prefix):
    csv_file_name = output_directory + "/" + file_name_prefix + ".csv"

    with open(csv_file_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)
