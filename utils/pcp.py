"""
https://github.com/gregornickel/pcp
"""

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import deepcopy


def load_csv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data_list = list(reader)

    results = []
    for i in range(1, len(data_list)):
        result = []
        for j in data_list[i]:
            if "." in j or "e" in j:
                try:
                    result.append(float(j))
                except ValueError:
                    result.append(j)
            else:
                try:
                    result.append(int(j))
                except ValueError:
                    result.append(j)

        results.append(result)

    return results, data_list[0]


def check_data(data, labels):
    for i in range(len(data)):
        assert len(data[i]) is len(labels), "data dimension (%d) does not " \
            "match with labels (%d)" % (len(data[i]), len(labels))


def check_formatting(yattribute, labels):
    if yattribute:
        assert len(yattribute) is len(labels), "dimension (%d) does not " \
            "match with labels (%d)" % (len(yattribute), len(labels))
    else:
        yattribute = [[]] * len(labels)
    return yattribute


def set_ytype(ytype, data, colorbar):
    for i in range(len(ytype)):
        if not ytype[i]:
            if type(data[0][i]) is str:
                ytype[i] = "categorial"
            else:
                ytype[i] = "linear"
    if colorbar: 
        assert ytype[len(ytype) - 1] == "linear", "colorbar axis needs to " \
            "be linear"
    return ytype


def set_ylabels(ylabels, data, ytype):
    for i in range(len(ylabels)): 
        # Generate ylabels for string values
        if not ylabels[i] and ytype[i] == "categorial":
            ylabel = []
            for j in range(len(data)):
                if data[j][i] not in ylabel:
                    ylabel.append(data[j][i])
            ylabel.sort()
            if len(ylabel) == 1:
                ylabel.append("")
            ylabels[i] = ylabel
    return ylabels


def replace_str_values(data, ytype, ylabels):
    for i in range(len(ytype)):
        if ytype[i] == "categorial":
            for j in range(len(data)):
                data[j][i] = ylabels[i].index(data[j][i])
    return np.array(data).transpose()


def set_ylim(ylim, data):
    for i in range(len(ylim)):
        if not ylim[i]:
            ylim[i] = [np.min(data[i, :]), np.max(data[i, :])]
            if ylim[i][0] == ylim[i][1]:
                ylim[i] = [ylim[i][0] * 0.95, ylim[i][1] * 1.05]
            if ylim[i] == [0.0, 0.0]:
                ylim[i] = [0.0, 1.0]
    return ylim


def get_score(data, ylim):
    ymin = ylim[len(ylim) - 1][0]
    ymax = ylim[len(ylim) - 1][1]
    score = (np.copy(data[len(ylim) - 1, :]) - ymin) / (ymax - ymin)
    return score


# Rescale data of secondary y-axes to scale of first y-axis
def rescale_data(data, ytype, ylim):
    min0 = ylim[0][0]
    max0 = ylim[0][1]
    scale = max0 - min0
    for i in range(1, len(ylim)):
        mini = ylim[i][0]
        maxi = ylim[i][1]
        if ytype[i] == "log":
            logmin = np.log10(mini)
            logmax = np.log10(maxi)
            span = logmax - logmin
            data[i, :] = ((np.log10(data[i, :]) - logmin) / span) * scale + min0
        else:
            data[i, :] = ((data[i, :] - mini) / (maxi - mini)) * scale + min0
    return data


def get_path(data, i):
    n = data.shape[0] # number of y-axes
    verts = list(zip([x for x in np.linspace(0, n - 1, n * 3 - 2)], 
        np.repeat(data[:, i], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    return path


def pcp(data, 
        labels, 
        ytype=None, 
        ylim=None, 
        ylabels=None, 
        figsize=(10, 5), 
        rect=[0.125, 0.1, 0.75, 0.8], 
        curves=True,
        alpha=1.0,
        colorbar=True, 
        colorbar_width=0.02,
        cmap=plt.get_cmap("inferno")
        ):
    """
    Parallel Coordinates Plot 

    Parameters
    ----------
    data: nested array
        Inner arrays containing data for each curve.
    labels: list
        Labels for y-axes.
    ytype: list, optional
        Default "None" allows linear axes for numerical values and categorial 
        axes for data of type string. If ytype is passed, logarithmic axes are 
        also possible, e.g.  ["categorial", "linear", "log", [], ...]. Vacant 
        fields must be filled with an empty list []. 
    ylim: list, optional
        Custom min and max values for y-axes, e.g. [[0, 1], [], ...].
    ylabels: list, optional (not recommended)
        Only use this option if you want to print more categories than you have
        in your dataset for categorial axes. You also have to set the right 
        ylim for this option to work correct.
    figsize: (float, float), optional
        Width, height in inches.
    rect: array, optional
        [left, bottom, width, height], defines the position of the figure on
        the canvas. 
    curves: bool, optional
        If True, B-spline curve is drawn.
    alpha: float, optional
        Alpha value for blending the curves.
    colorbar: bool, optional
        If True, colorbar is drawn.
    colorbar_width: float, optional
        Defines the width of the colorbar.
    cmap: matplotlib.colors.Colormap, optional
        Specify colors for colorbar.
    
    Returns
    -------
    `~matplotlib.figure.Figure`
    """
    
    [left, bottom, width, height] = rect
    data = deepcopy(data)
    
    # Check data
    check_data(data, labels)
    ytype = check_formatting(ytype, labels)
    ylim = check_formatting(ylim, labels)
    ylabels = check_formatting(ylabels, labels)

    # Setup data
    ytype = set_ytype(ytype, data, colorbar) 
    ylabels = set_ylabels(ylabels, data, ytype)
    data = replace_str_values(data, ytype, ylabels)
    ylim = set_ylim(ylim, data)
    score = get_score(data, ylim)
    data = rescale_data(data, ytype, ylim)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_axes([left, bottom, width, height])
    axes = [ax0] + [ax0.twinx() for i in range(data.shape[0] - 1)]

    # Plot curves
    for i in range(data.shape[1]):
        if colorbar:
            color = cmap(score[i])
        else:
            color = "blue"

        if curves:
            path = get_path(data, i)
            patch = PathPatch(path, facecolor="None", lw=1.5, alpha=alpha, 
                    edgecolor=color, clip_on=False)
            ax0.add_patch(patch)
        else:
            ax0.plot(data[:, i], color=color, alpha=alpha, clip_on=False)

    # Format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(labels)

    # Format y-axis
    for i, ax in enumerate(axes):
        ax.spines["left"].set_position(("axes", 1 / (len(labels) - 1) * i))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        ax.yaxis.set_ticks_position("left")
        ax.set_ylim(ylim[i])
        if ytype[i] == "log":
            ax.set_yscale("log")
        if ytype[i] == "categorial":
            ax.set_yticks(range(len(ylabels[i])))
        if ylabels[i]:
            ax.set_yticklabels(ylabels[i])
        
    if colorbar:
        bar = fig.add_axes([left + width, bottom, colorbar_width, height])
        norm = mpl.colors.Normalize(vmin=ylim[i][0], vmax=ylim[i][1])
        mpl.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, 
            orientation="vertical")
        bar.tick_params(size=0)
        bar.set_yticklabels([])

    return fig


if __name__ == "__main__":
    # Minimal working example
    results = [["ResNet", 0.0001, 4, 0.2],
               ["ResNet", 0.0003, 8, 1.0],
               ["DenseNet", 0.0005, 4, 0.65],
               ["DenseNet", 0.0007, 8, 0.45],
               ["DenseNet", 0.001, 2, 0.8]]
    labels = ["Network", "Learning rate", "Batchsize", "F-Score"]
    pcp(results, labels)
    plt.show()
