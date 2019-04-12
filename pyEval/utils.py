# -*- coding: utf-8 -*-

""" Created on 5:49 PM, 4/10/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

import matplotlib.pyplot as plt
import logging
import numpy as np
import os


def read_dictionary(dict_path):
    temp = {}
    with open(dict_path, "r") as f:
        for line in f.readlines():
            line_split = line.strip().split(" ")
            temp[line_split[0]] = line_split[2:]

    return temp


def plot(type_of_plot, data, figure, xlabel=None, ylabel=None, **kwargs):
    """
        FORGET IT
        Drawing plot, support multiple types
    :param type_of_plot:
    :param data:
    :param kwargs:
    :return:
    """
    logger_eval = logging.getLogger("eval")

    if type_of_plot == 'boxplot':
        axes = figure.gca()
        axes.boxplot(data)

        if xlabel:
            figure.xlabel(xlabel)
        if ylabel:
            axes.ylabel(ylabel)

        return axes
    else:
        logger_eval.critical("The {} plot is not supported yet".format(type_of_plot))
        return None


def label_subplot_with_a_big_subplot(_fig, xlabel=None, ylabel=None):
    """
        To name a big figure with a lot of subplot, by adding a big subplot over it and name its x/y axis
    :param ylabel:
    :param xlabel:
    :param _fig:
    :return:
    """
    ax_big = _fig.add_subplot(111, frameon=False)
    ax_big.spines['top'].set_color('none')
    ax_big.spines['bottom'].set_color('none')
    ax_big.spines['left'].set_color('none')
    ax_big.spines['right'].set_color('none')
    ax_big.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    if xlabel:
        ax_big.set_xlabel(xlabel)
    if ylabel:
        ax_big.set_ylabel(ylabel)

    return ax_big


def get_data(path, is_toy=False):
    if is_toy:
        textpath = os.path.join(path, 'txt/text_toy')
    else:
        textpath = os.path.join(path, 'txt/text')

    logging.info("Reading data from {}".format(textpath))
    wavs = []
    sr = None

    with open(textpath, 'r') as f:
        for line in tqdm(f.readlines()):
            filename, text = line.strip().split("|")
            y_temp, sr = lbr.load(os.path.join(os.path.join(path, 'wav'), filename + ".wav"), sr=None, dtype='double')

            wavs.append([filename, text, y_temp])

    return wavs, sr


if __name__ == "__main__":
    # First create some toy data:
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    # Creates just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)
    #
    # Creates four polar axes, and accesses them through the returned array
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
    axes[0, 0].plot(x, y)
    axes[1, 1].scatter(x, y)
    #
    # #Share a X axis with each column of subplots
    # plt.subplots(2, 2, sharex='col')
    #
    # #Share a Y axis with each row of subplots
    # plt.subplots(2, 2, sharey='row')
    #
    # #Share both X and Y axes with all subplots
    # plt.subplots(2, 2, sharex='all', sharey='all')
    #
    # #Note that this is the same as
    # plt.subplots(2, 2, sharex=True, sharey=True)
    #
    # #Creates figure number 10 with a single subplot
    # #and clears it if it already exists.
    # fig, ax=plt.subplots(num=10, clear=True)

    plt.show()
