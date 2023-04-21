from pathlib import Path
from matplotlib import pyplot as plt


def plot_2d(y_data, x_data=None, y_data_std=None, title="graph", graph_labels=None,
            x_label="x", y_label="y", output_directory=None, file_name_prefix=None, show=False):
    try:
        plt.clf()

        plt.title(title, fontweight='bold')
        plt.xlabel(x_label, labelpad=8, fontweight='bold')
        plt.ylabel(y_label, labelpad=8, fontweight='bold')

        if x_data is None:
            x_data = []
            for y in y_data:
                x_data.append(list(range(len(y))))

        if y_data_std is None:
            y_data_std = []
            for y in y_data:
                y_data_std.append([0]*len(y))

        no_legend = False

        if graph_labels is None:
            no_legend = True
            graph_labels = [None] * len(y_data_std)

        for x, y, y_err, graph_label in zip(x_data, y_data, y_data_std, graph_labels):
            plt.plot(x, y, label=graph_label)

            y_minus = []
            y_plus = []
            for data, error in zip(y, y_err):
                y_minus.append(data - error)
                y_plus.append(data + error)
            plt.fill_between(x, y_minus, y_plus, alpha=.3)

        if no_legend is False:
            plt.legend(loc='best')

        if output_directory and file_name_prefix:
            if not Path(output_directory).is_dir():
                Path(output_directory).mkdir(parents=True)
            plt.savefig(output_directory + "/" + file_name_prefix + ".png")

        if show:
            plt.show()

        return plt
    except:
        print("Plot error")
