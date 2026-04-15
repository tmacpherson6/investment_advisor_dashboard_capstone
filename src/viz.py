#123456789012345678901234567890123456789012345678901234567890123456789012345678
"""This module contains data visualization helper functions.

Author: Pete King
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt


def elbow_plot(
    data: np.ndarray,
    title='Elbow Plot',
    xlabel='Score',
    ylabel='Number'
):
    """Creates an elbow plot from a set of coordinates (data).

    Keyword Arguments:
      data -- ndarray of coordinates (x, y) to plot
      title -- title for the plot
      xlabel -- label for the x-axis
      ylabel -- label for the y-axis
    """
    fig = plt.figure(figsize=(4, 3))
    ax = fig.subplots()
    ax.plot(data[:, 0], data[:, 1])
    ax.set_title(title)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    fig.tight_layout()
    plt.show()
    return None