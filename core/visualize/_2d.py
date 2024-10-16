from typing import List
import numpy as np
import matplotlib.pyplot as plt


def ShowGeometrys(geometrysList: List[List[np.ndarray]], fig_title=None, titles=None,
                  spliter=2, points_grids: List[List[np.ndarray]] = None):
    n = len(geometrysList)
    marker_size = 1
    line_ = "o-"
    if titles:
        assert len(titles) == n
    rows = n//spliter + n % spliter
    _, axs = plt.subplots(rows, spliter)
    for ii, geometrys in enumerate(geometrysList):
        for geometry in geometrys:
            if (rows) > 1:
                axs[ii//spliter, ii %
                    spliter].plot(geometry[:, 0], geometry[:, 1], line_, markersize=marker_size)

            elif rows == 1 and spliter == 1:
                axs.plot(geometry[:, 0], geometry[:, 1],
                         line_, markersize=marker_size)
            else:
                axs[ii % spliter].plot(
                    geometry[:, 0], geometry[:, 1], line_, markersize=marker_size)
        if points_grids is not None and ii < len(points_grids):
            if (rows) > 1:
                axs[ii//spliter, ii %
                spliter].plot(points_grids[ii, :, 0], points_grids[ii, :, 1], 'o', markersize=1)
            elif rows == 1 and spliter == 1:
                axs.plot(points_grids[ii, :, 0], points_grids[ii, :, 1], 'o', markersize=1)
            else:
                axs[ii % spliter].plot(points_grids[ii][:, 0], points_grids[ii][:, 1], 'o', markersize=1)

        if titles:
            axs[ii//spliter, ii % spliter].set_title(titles[ii])
    if fig_title:
        plt.suptitle(fig_title, fontsize=22, weight='bold')
    plt.show()
