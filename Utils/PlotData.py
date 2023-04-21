from pathlib import Path
from matplotlib import pyplot as plt


def plot_2d(y_data, x_data=None, y_data_std=None, title="graph", x_label="x", y_label="y",
            graph_labels=None, output_directory=None, file_name_prefix=None, show=False):
    try:
        init_graph(plt, title, x_label, y_label)

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
            enable_legend(plt)

        if show:
            plt.show()

        dump_png(plt, output_directory, file_name_prefix)

        return plt
    except Exception as e:
        print("plot_2d error:", e)


def plot_bar(y_data, graph_labels, graph_group_names, y_data_std=None, title="graph", x_label="x", y_label="y",
             output_directory=None, file_name_prefix=None, show=False):
    try:
        init_graph(plt, title, x_label, y_label)

        data_number = len(y_data)

        step = 1
        data_points = len(y_data[0])
        space_factor = 2
        bar_width = step/(data_number + space_factor)

        x_data = [list(range(step, step * data_points + 1, step))]

        for counter in range(0, data_number - 1):
            x_data.append([x + bar_width for x in x_data[counter]])

        if y_data_std is None:
            y_data_std = []
            for counter in range(data_number):
                y_data_std.append(None)

        for x, y, y_err, graph_label in zip(x_data, y_data, y_data_std, graph_labels):
            plt.bar(x, y, width=bar_width, label=graph_label)
            if y_err:
                plt.errorbar(x, y, fmt="k_", yerr=y_err, ecolor="black", elinewidth=bar_width*5, barsabove=True)

        enable_legend(plt)

        plt.xticks(x_data[round(data_number/2 + 0.01) - 1], graph_group_names)

        if show:
            plt.show()

        dump_png(plt, output_directory, file_name_prefix)
    except Exception as e:
        print("plot_bar error:", e)


def enable_legend(plot):
    plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
    plot.tight_layout()


def init_graph(plot, title="graph", x_label="x", y_label="y"):
    plot.clf()
    plot.title(title, fontweight='bold')
    plot.xlabel(x_label, labelpad=8, fontweight='bold')
    plot.ylabel(y_label, labelpad=8, fontweight='bold')


def dump_png(plot, output_directory, file_name_prefix):
    if output_directory and file_name_prefix:
        if not Path(output_directory).is_dir():
            Path(output_directory).mkdir(parents=True)
        plot.savefig(output_directory + "/" + file_name_prefix + ".png")