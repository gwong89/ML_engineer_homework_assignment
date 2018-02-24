import matplotlib
import pandas as pd
import numpy as np

# handy function to plot some histograms of DataFrame columns
def plot_histogram(df, column_name, sort=False):
    histo = df[column_name].value_counts()
    if (sort):
        histo = histo.sort_index()
    X = np.array(histo.keys())
    Y = histo.values
    matplotlib.pylot.bar(np.arange(len(X)), Y, align='center')
    matplotlib.pylot.xticks(np.arange(len(X)), X)
    matplotlib.pylot.title("Histogram of " + column_name + " values")
    matplotlib.pylot.xlabel(column_name)
    matplotlib.pylot.ylabel('Frequency')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    matplotlib.pylot.show()


# plot correlation of categorical feature with outcome variable
def plot_feature_correlation(df, feature_column_name, sort=False):
    c = 1.0 - df.groupby(feature_column_name)['outcome'].mean()
    if (sort):
        c = c.sort_index()
    X = np.array(c.keys())
    Y = c.values
    matplotlib.pylot.bar(np.arange(len(X)), Y, align='center')
    matplotlib.pylot.xticks(np.arange(len(X)), X)
    matplotlib.pylot.title("Correlation of outcome variable with " + feature_column_name + " categories")
    matplotlib.pylot.xlabel(feature_column_name)
    matplotlib.pylot.ylabel('Percent non spectrum')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    matplotlib.pylot.show()


def plot_classifier_profiles(bunch_of_classifier_data, plot_title, default_coverage_to_plot=0.75,
                             specificity_bin_width=0.025, ylim=(0., 1.), legend_font_size=16,
                             shaded_sensitivity_zones=True):

    fig = matplotlib.pylot.figure(figsize=(20, 6))

    # setup axes
    matplotlib.pylot.xlabel('specificity', fontsize=28)
    matplotlib.pylot.xticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    matplotlib.pylot.xlim(0.0, 1.0)
    matplotlib.pylot.ylabel('sensitivity', fontsize=28)
    matplotlib.pylot.yticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    matplotlib.pylot.ylim(ylim)

    # add shaded sensitivity zones if required
    if (shaded_sensitivity_zones):
        matplotlib.pylot.axhspan(0.7, 0.8, edgecolor='none', facecolor='lightyellow', alpha=1.0, zorder=1)
        matplotlib.pylot.axhspan(0.8, 0.9, edgecolor='none', facecolor='orange', alpha=0.3, zorder=1)

    # plot data
    for (classifier_info, sensitivity_specificity_dataframe) in bunch_of_classifier_data:
        print 'Plot for classifier info: ', classifier_info

        # if we're being asked to plot the optimal point only (as opposed to an ROC curve)
        if ('type' in classifier_info and classifier_info['type'] == 'optimal_point'):

            label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'
            if sensitivity_specificity_dataframe['coverage'] < 1.0:
                label = label + ' @ ' + "{0:.0f}%".format(
                    100 * sensitivity_specificity_dataframe['coverage']) + ' coverage'
            size = classifier_info['size'] if 'size' in classifier_info else 400
            linestyle = classifier_info['linestyle'] if 'linestyle' in classifier_info else '-'
            alpha = classifier_info['alpha'] if 'alpha' in classifier_info else 0.75
            fill = classifier_info['fill'] if 'fill' in classifier_info else True
            edgecolors = classifier_info['color'] if 'color' in classifier_info else None
            if (fill):
                facecolors = classifier_info['color'] if 'color' in classifier_info else None
            else:
                facecolors = 'none'

            label = label + " [ {0:.0f}%".format(100 * sensitivity_specificity_dataframe['sensitivity']) + ' sens, '
            label = label + "{0:.0f}%".format(100 * sensitivity_specificity_dataframe['specificity']) + ' spec]'

            matplotlib.pylot.scatter([sensitivity_specificity_dataframe['specificity']],
                        [sensitivity_specificity_dataframe['sensitivity']], s=size, alpha=alpha, facecolors=facecolors,
                        edgecolors=edgecolors, label=label, zorder=10)

        # we default to plotting curves
        else:

            min_acceptable_coverage = classifier_info[
                'coverage'] if 'coverage' in classifier_info else default_coverage_to_plot
            specificity_sensitivity_values = [(spec, sen) for spec, sen in
                                              zip(sensitivity_specificity_dataframe['specificity'].values,
                                                  sensitivity_specificity_dataframe['sensitivity'].values)]
            plot_color = classifier_info['color'] if 'color' in classifier_info else None
            label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'
            linewidth = classifier_info['linewidth'] if 'linewidth' in classifier_info else 3
            linestyle = classifier_info['linestyle'] if 'linestyle' in classifier_info else '-'

            if 'coverage' not in sensitivity_specificity_dataframe:
                matplotlib.pylot.plot(sensitivity_specificity_dataframe['specificity'],
                         sensitivity_specificity_dataframe['sensitivity'], marker=None, linewidth=linewidth,
                         label=label, color=plot_color, linestyle=linestyle)

            else:

                sensitivity_specificity_dataframe['rounded_specificity'] = sensitivity_specificity_dataframe[
                    'specificity'].apply(
                    lambda x: 0 if np.isnan(x) else specificity_bin_width * (int(x / specificity_bin_width)))

                acceptable_coverage_sensitivity_specificity_dataframe = sensitivity_specificity_dataframe[
                    sensitivity_specificity_dataframe.coverage >= min_acceptable_coverage]
                min_sensitivity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')[
                    'sensitivity'].min()
                max_sensitivity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')[
                    'sensitivity'].max()

                specificity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')[
                    'rounded_specificity'].max()

                matplotlib.pylot.plot(specificity, max_sensitivity, linewidth=linewidth,
                         label=label + ' @ ' + "{0:.0f}%".format(100 * min_acceptable_coverage) + '+ coverage',
                         color=plot_color, linestyle=linestyle)

    # add legend
    matplotlib.pylot.legend(loc="lower left", prop={'size': legend_font_size})

    # add title
    matplotlib.pylot.title(plot_title, fontsize=20, fontweight='bold')

    # let's do it!
    matplotlib.pylot.show()
    return matplotlib.pylot, fig


# same as above but plots a simple bar chart instead of complicated ROC curves
def barplot_classifier_profiles(bunch_of_classifier_data, plot_title, sensitivity_low=0.75, sensitivity_high=0.85,
                                min_coverage=0.7):
    barplot_data = []

    for (classifier_info, sensitivity_specificity_dataframe) in bunch_of_classifier_data:
        label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'

        if 'coverage' in sensitivity_specificity_dataframe.columns:
            sensitivity_specificity_dataframe = sensitivity_specificity_dataframe[
                (sensitivity_specificity_dataframe['coverage'] >= min_coverage)]

            sensitivity = sensitivity_specificity_dataframe.groupby('rounded_specificity')['sensitivity'].max()
            specificity = sensitivity_specificity_dataframe.groupby('rounded_specificity')['rounded_specificity'].max()
        else:
            sensitivity = sensitivity_specificity_dataframe.groupby('specificity')['sensitivity'].max()
            specificity = sensitivity_specificity_dataframe.groupby('specificity')['specificity'].max()

        temp = pd.DataFrame(zip(specificity, sensitivity), columns=['specificity', 'sensitivity'])
        temp2 = temp[(temp['sensitivity'] >= sensitivity_low) & (temp['sensitivity'] <= sensitivity_high)]
        bar_height = temp2['specificity'].mean()

        barplot_data += [(classifier_info['label'], bar_height)]

    fig = matplotlib.pylot.figure(figsize=(20, 10))

    barlist = matplotlib.pylot.barh(range(len(barplot_data)), [x[1] for x in barplot_data], align='center', edgecolor="black",
                       alpha=0.8)
    matplotlib.pylot.yticks(range(len(barplot_data)), [x[0] for x in barplot_data])

    # setup value labels
    for i, v in enumerate([x[1] for x in barplot_data]):
        matplotlib.pylot.text(v - 0.05, i - 0.1, "{0:.0f}%".format(100 * v), color='black', fontsize=24)

    # setup name labels
    for i, v in enumerate([x[0]['label'] for x in bunch_of_classifier_data]):
        matplotlib.pylot.text(0.02, i - 0.1, v, color='black', fontsize=18)

    # setup colors
    for i in range(0, len(barlist)):
        classifier_info = bunch_of_classifier_data[i][0]
        color = classifier_info['color'] if 'color' in classifier_info else None
        barlist[i].set(facecolor=color)

    # setup axes
    matplotlib.pylot.ylabel('algorithm', fontsize=28)
    matplotlib.pylot.yticks([])

    matplotlib.pylot.xlabel('specificity', fontsize=28)
    matplotlib.pylot.xticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    matplotlib.pylot.xlim(0.0, 1.0)

    # add title
    matplotlib.pylot.title(plot_title, fontsize=20, fontweight='bold')

    # let's do it!
    print 'show figure with title ', plot_title
    matplotlib.pylot.show()
    return matplotlib.pylot, fig