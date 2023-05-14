import json
import csv
from pathlib import Path

from Utils.PlotData import plot_2d


def dump_outputs(data, output_directory, file_name_prefix):
    #plot_2d([data], output_directory=output_directory, file_name_prefix=file_name_prefix)
    dump_json(data, output_directory, file_name_prefix)


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


def load_csv(output_directory, file_name_prefix):
    csv_file_name = output_directory + "/" + file_name_prefix + ".csv"

    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        lines = []

        for row in csv_reader:
            lines.append(list(row.values()))

        return lines
