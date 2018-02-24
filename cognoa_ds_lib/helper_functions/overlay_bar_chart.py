import numpy as np
import matplotlib.pyplot


def _get_list_of_unique_x_values(list_of_x_data_lists):
    ''' Helper function for functions below. Intended for overlaying bar charts where
    a list of lists or arrays represent the x-values of each dataset that will be overlayed.
    This functions determines the list of the x-values that should be displayed on the
    x-axis '''

    list_of_unique_lists = [np.unique(dataset) for dataset in list_of_x_data_lists]
    combined_list_with_duplicates = [item for sublist in list_of_unique_lists for item in sublist]
    list_of_unique_x_values = np.unique(combined_list_with_duplicates)
    return list_of_unique_x_values


def overlay_bar_charts_from_numeric_arrays(list_of_x_data_lists, legend_label_list, plot_options_dict):
    ''' Intended for situations where there is a small number of often repeated values that
    most of the data might have and you want to compare distributions without them obscuring
    one another

    x_values_list: a list of data lists or arrays
    '''
    list_of_unique_x_values = _get_list_of_unique_x_values(list_of_x_data_lists)
    list_of_x_data_bars = []
    list_of_y_data_bars = []
    for dataset in list_of_x_data_lists:
        x_data_bars = []
        y_data_bars = []
        for x_value in list_of_unique_x_values:
            y_value = len(dataset[dataset == x_value])
            x_data_bars.append(x_value)
            y_data_bars.append(y_value)
        list_of_x_data_bars.append(np.array(x_data_bars))
        list_of_y_data_bars.append(np.array(y_data_bars))

    overlay_bar_charts(list_of_x_data_bars, list_of_y_data_bars, legend_label_list, x_values_are_categorical=False,
                       plot_options_dict=plot_options_dict)


def overlay_bar_charts(list_of_x_data_bars, list_of_y_data_bars, legend_label_list=[''], x_values_are_categorical=True,
                       plot_options_dict=None):
    ''' Overlay some number of bar charts with values (or categories) list_of_x_data_bars, and y_values list_of_y_data_bars
	... This can run on categorical or numeric data. If numeric x-axis becomes the value of the numbers, and the bars will
	probably not be equally spaced. The chart can easily get overwhelmed with many bins. This function is mostly useful if only
	a small number of often repeated numeric values are present in the data.
	... If your x data is categorical then you should have it remapped to equally spaced bars: set x_values_are_categorical to True
	... The overlaying of many different data bars is accomplished by injecting coordinated offsets into the x-values of different elements
	of the x lists so that they can be seen side-by-side. When interpreting the results (especially if the x-data is numerical rather than
	categorical) it should be remembered that this offset is a necessary artifact of plotting in an understandable way and not indicative of
	a true numerical offset in the data. '''

    def get_bar_width(n_bars, x_range):
        bar_density = float(n_bars) / float(x_range[1] - x_range[0])
        bar_width = 0.5 / bar_density  ### want roughly this percent of visual screen to be taken up by bars
        return bar_width

    if 'figshape' in plot_options_dict.keys():
        matplotlib.pyplot.figure(figsize=plot_options_dict['figshape'])
    else:
        matplotlib.pyplot.figure(figsize=(12, 8))
    if 'grid' in plot_options_dict.keys() and plot_options_dict['grid'] == True:
        matplotlib.pyplot.grid(True)
    n_datasets = len(list_of_x_data_bars)
    assert n_datasets == len(list_of_y_data_bars)
    assert n_datasets == len(legend_label_list)

    xtick_labels = None
    if x_values_are_categorical:
        #### In this case x-values need to be identical in every array, otherwise
        #### Plotted bins will not match up
        for x_values_case in list_of_x_data_bars:
            assert np.array_equal(list_of_x_data_bars[0], x_values_case)
        xtick_labels = list_of_x_data_bars[0]
        x_range = [0, len(xtick_labels)]
        list_of_x_values_for_plotting = [np.arange(len(xtick_labels))] * n_datasets
    else:
        x_range = [min([min(x_data) for x_data in list_of_x_data_bars]),
                   max([max(x_data) for x_data in list_of_x_data_bars])]
        list_of_x_values_for_plotting = list_of_x_data_bars

    n_bars = sum([len(x_data) for x_data in list_of_x_data_bars])
    bar_width = get_bar_width(n_bars, x_range)
    if 'color_ordering' in plot_options_dict:
        colors_list = plot_options_dict['color_ordering']
    else:
        colors_list = ['black', 'red', 'blue', 'yellow', 'green', 'purple', 'orange']
    for plot_index, (x_data_bars, y_data_bars, legend_label, color) in enumerate(
            zip(list_of_x_values_for_plotting, list_of_y_data_bars, legend_label_list, colors_list)):
        x_offset = bar_width * ((float(plot_index)) - (0.5 * (n_datasets)))
        this_legend_label = legend_label
        if 'means_in_legend' in plot_options_dict.keys() and plot_options_dict['means_in_legend'] == True:
            this_legend_label += ', mean=' + str(
                round(np.average(list_of_x_data_bars[plot_index], weights=list_of_y_data_bars[plot_index]), 3))
            matplotlib.pyplot.bar(left=x_data_bars + x_offset, height=y_data_bars, width=bar_width, color=color, alpha=0.5,
                label=this_legend_label)
    if legend_label_list != ['']:
        matplotlib.pyplot.legend(fontsize=plot_options_dict['legend_fontsize'])
        matplotlib.pyplot.xlabel(plot_options_dict['xlabel'], fontsize=plot_options_dict['xlabel_fontsize'])
        matplotlib.pyplot.ylabel(plot_options_dict['ylabel'], fontsize=plot_options_dict['ylabel_fontsize'])
        matplotlib.pyplot.title(plot_options_dict['title'], fontsize=plot_options_dict['title_fontsize'])
    if x_values_are_categorical:
        ### Increase bottom margin for readability
        matplotlib.pyplot.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=50, fontsize=8)
        matplotlib.pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

