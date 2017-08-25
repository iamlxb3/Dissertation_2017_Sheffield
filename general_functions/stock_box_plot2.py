import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import plot, show, savefig, xlim, figure, \
    hold, ylim, legend, boxplot, setp, axes

def setBoxColors(bp,metrics_name_list):
    colour_list = ['blue', 'red', 'green', 'cyan', 'yellow', 'magenta']
    box_count = 0
    caps_count = 0
    whiskers_count = 0
    medians_count = 0
    fliers_count = 0

    for i, metric in enumerate(metrics_name_list):
        plt.setp(bp['boxes'][box_count], color=colour_list[i])
        box_count += 1
        plt.setp(bp['caps'][caps_count], color=colour_list[i])
        caps_count += 1
        plt.setp(bp['caps'][caps_count], color=colour_list[i])
        caps_count += 1
        plt.setp(bp['whiskers'][whiskers_count], color=colour_list[i])
        whiskers_count += 1
        plt.setp(bp['whiskers'][whiskers_count], color=colour_list[i])
        whiskers_count += 1
        plt.setp(bp['medians'][medians_count], color=colour_list[i])
        medians_count += 1
        plt.setp(bp['fliers'][fliers_count], markeredgecolor=colour_list[i])
        fliers_count += 1


def get_positions(metrics_name_list, position_now, box_gap):
    positions_list = []
    for i,_ in enumerate(metrics_name_list):
        positions_list.append(position_now)
        position_now += box_gap
    return positions_list, position_now

def stock_metrics_result_box_plot(metrics_result_dict, trail_number_list, metrics_name_list, title = ''):
    box_widths = 0.3
    box_gap = 0.5
    category_gap = 0.7
    position_now = 0
    category_pos_list = []
    fig = figure()
    ax = axes()
    hold(True)

    # Some fake data to plot
    for trail_number in trail_number_list:
        X = []
        for metrics in metrics_name_list:
            X.append(metrics_result_dict[trail_number][metrics])

        # first boxplot pair
        position_now += category_gap
        positions_list,position_now = get_positions(metrics_name_list, position_now, box_gap)
        category_pos_list.append(np.average(positions_list))
        bp = boxplot(X, positions=positions_list, widths=box_widths, sym='+')
        setBoxColors(bp,metrics_name_list)


    # set axes limits and labels
    xlim(0, 12)
    ylim(0.28, 0.6)
    ax.set_xticklabels(trail_number_list)
    ax.set_xticks(category_pos_list)
    ax.set_xlabel('Number of trails in 1 experiment')
    ax.set_title(title)

    # draw temporary red and blue lines and use them to create a legend
    h_list = []
    shape = '-'
    legend_list = ['b{}'.format(shape),
                   'r{}'.format(shape),
                   'g{}'.format(shape),
                   'c{}'.format(shape),
                   'y{}'.format(shape),
                   'm{}'.format(shape)]

    for i,_ in enumerate(metrics_name_list):
        h, = plot([1, 1], legend_list[i])
        h_list.append(h)
        #h.set_visible(False)

    # hB, = plot([1, 1], 'b-')
    # hR, = plot([1, 1], 'r-')
    legend(h_list, metrics_name_list)
    for h in h_list:
        h.set_visible(False)
    # hB.set_visible(False)
    # hR.set_visible(False)
    show()

