from pathlib import Path
from matplotlib import pyplot as plt


def plot_2d(y_data, x_data=None, y_data_std=None, title="graph", x_label="x", y_label="y",
            graph_labels=None, output_directory=None, file_name_prefix=None, show=False,
            x_font_size=8, x_rotation=0):
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

        plt.xticks(fontsize=x_font_size, rotation=x_rotation)

        if no_legend is False:
            enable_legend(plt)

        if file_name_prefix is None:
            file_name_prefix = title

        dump_png(plt, output_directory, file_name_prefix)

        if show:
            plt.show()

        return plt
    except Exception as e:
        print("plot_2d error:", e)


def plot_bar(y_data, graph_labels, graph_group_names, y_data_std=None, title="graph", x_label="x", y_label="y",
             output_directory=None, file_name_prefix=None, show=False, x_font_size=8, x_rotation=0):
    try:
        init_graph(plt, title, x_label, y_label)

        data_number_of_configurations = len(y_data)

        step = 1
        data_number_of_groups = len(y_data[0])
        space_factor = round(data_number_of_configurations * 0.4)
        bar_width = step/(data_number_of_configurations + space_factor)

        x_data = [list(range(step, step * data_number_of_groups + 1, step))]

        for counter in range(0, data_number_of_configurations - 1):
            x_data.append([x + bar_width for x in x_data[counter]])

        if y_data_std is None:
            y_data_std = []
            for counter in range(data_number_of_configurations):
                y_data_std.append(None)

        for x, y, y_err, graph_label in zip(x_data, y_data, y_data_std, graph_labels):
            plt.bar(x, y, width=bar_width, label=graph_label)
            if y_err:
                plt.errorbar(x, y, fmt="k_", markersize=bar_width*25, yerr=y_err, ecolor="black",
                             elinewidth=bar_width*5, capsize=bar_width*10)

        enable_legend(plt, legend_font_size(data_number_of_configurations))

        if data_number_of_configurations % 2 == 1:
            data_groups_coordinates = x_data[round(data_number_of_configurations/2 + 0.01) - 1]
        else:
            data_groups_coordinates = [x + bar_width/2 for x in x_data[round(data_number_of_configurations / 2) - 1]]

        plt.xticks(data_groups_coordinates, graph_group_names, fontsize=x_font_size, rotation=x_rotation)

        if file_name_prefix is None:
            file_name_prefix = title

        dump_png(plt, output_directory, file_name_prefix)

        if show:
            plt.show()

    except Exception as e:
        print("plot_bar error:", e)


def legend_font_size(points):
    if points <= 3:
        return "small"
    elif points <= 6:
        return "x-small"
    elif points <= 9:
        return "xx-small"
    else:
        return "xx-small"


def enable_legend(plot, fontsize="medium"):
    plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize, labelspacing=1)
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
        plot.savefig(output_directory + "/" + file_name_prefix + ".png", dpi=600)
