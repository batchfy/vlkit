import matplotlib
import numpy as np


def clear_xticks(axes):
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            ax.set_xticks([])
    elif isinstance(axes, matplotlib.axes._axes.Axes):
        axes.set_xticks([])
    else:
        raise TypeError(type(axes))


def clear_yticks(axes):
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            ax.set_yticks([])
    elif isinstance(axes, matplotlib.axes._axes.Axes):
        axes.set_yticks([])
    else:
        raise ValueError


def clear_ticks(axes):
    clear_xticks(axes)
    clear_yticks(axes)
