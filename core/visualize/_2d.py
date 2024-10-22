from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_PFIXES = ["dark", "deep", "medium"]
_AVAILABLE_COLORS = list(mcolors.CSS4_COLORS.keys())
_AVAILABLE_COLORS = [color for color in _AVAILABLE_COLORS if (
                     any((a in color) for a in _PFIXES))]
_AVAILABLE_IDXS = np.arange(stop=len(_AVAILABLE_COLORS))


def ShowGeometrys(geometrysList: List[List[np.ndarray]], fig_title=None, titles=None,
                  spliter=2, points_grids: List[List[np.ndarray]] = None,
                  points_grids_color_idx_map: List[np.ndarray] = None,
                  points_grids_clusters_centers: List[np.ndarray] = None,
                  points_grids_vector_idx_map: List[np.ndarray] = None,
                  background_color="white",
                  file_name: str = None, show_plot=True):
    n = len(geometrysList)
    marker_size = 1
    line_ = "o-"
    if points_grids_color_idx_map is not None:
        idxs_to_colors_maps = []
        for points_grid_color_idx_map in points_grids_color_idx_map:
            needed_color_len = np.max(points_grid_color_idx_map)+1
            r_idxs = np.random.choice(
                _AVAILABLE_IDXS, needed_color_len, replace=False)
            idxs_to_colors_maps.append(np.array(_AVAILABLE_COLORS)[r_idxs])

    if titles:
        assert len(titles) == n
    rows = n // spliter + n % spliter

    fig, axs = plt.subplots(rows, spliter)
    for ii, geometrys in enumerate(geometrysList):
        if (rows) > 1:
            _plotter = axs[ii // spliter, ii % spliter]
        elif rows == 1 and spliter == 1:
            _plotter = axs
        else:
            _plotter = axs[ii % spliter]

        _plotter.set_facecolor(background_color)

        for geometry in geometrys:
            _plotter.plot(geometry[:, 0], geometry[:, 1],
                          line_, markersize=marker_size)
        if points_grids is not None and ii < len(points_grids):

            if (ii+1) > len(points_grids_color_idx_map) or points_grids_color_idx_map[ii] is None:
                _plotter.plot(
                    points_grids[ii][:, 0], points_grids[ii][:, 1], 'o', markersize=1)

            else:
                for idx, point in enumerate(points_grids[ii]):
                    _plotter.plot(point[0], point[1], 'o', markersize=1,
                                  color=idxs_to_colors_maps[ii][points_grids_color_idx_map[ii][idx]])
                if points_grids_clusters_centers is not None and points_grids_clusters_centers[ii] is not None:
                    for idx, c_point in enumerate(points_grids_clusters_centers[ii]):
                        _plotter.plot(
                            c_point[0], c_point[1], 'o', markersize=4,
                            color="black")
                        _plotter.plot(
                            c_point[0], c_point[1], 'o', markersize=3,
                            color=idxs_to_colors_maps[ii][idx])
            if points_grids_vector_idx_map and (ii) < len(points_grids_vector_idx_map) and points_grids_vector_idx_map[ii] is not None:
                vector_map = points_grids_vector_idx_map[ii] 
                for idx in range(len(vector_map)-1):
                    x0, y0 = points_grids[ii][vector_map[idx]]
                    x1, y1 = points_grids[ii][vector_map[idx+1]]
                    _plotter.arrow(x0,y0,x1-x0,y1-y0,head_width=4,head_length=1,color="red")

        if titles:
            axs[ii // spliter, ii % spliter].set_title(titles[ii])

    if fig_title:
        plt.suptitle(fig_title, fontsize=22, weight='bold')
    if file_name:
        print(f"Saving {file_name}")
        try:
            plt.savefig(file_name, dpi=250, bbox_inches='tight')
        except Exception as e:
            print(f"An Error Ocurred {e}")
        finally:
            print(f"File saved at : {file_name}")

    if show_plot:
        plt.show()
