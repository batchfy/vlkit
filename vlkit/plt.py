from collections.abc import Iterable
import matplotlib
import numpy as np


def clear_xticks(axes):
    if isinstance(axes, Iterable):
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        for ax in axes:
            ax.set_xticks([])
    elif isinstance(axes, matplotlib.axes._axes.Axes):
        axes.set_xticks([])
    else:
        raise ValueError


def clear_yticks(axes):
    if isinstance(axes, Iterable):
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        for ax in axes:
            ax.set_yticks([])
    elif isinstance(axes, matplotlib.axes._axes.Axes):
        axes.set_yticks([])
    else:
        raise ValueError


def clear_ticks(axes):
    clear_xticks(axes)
    clear_yticks(axes)
