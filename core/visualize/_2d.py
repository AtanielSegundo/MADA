from typing import List, Tuple,  Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
import matplotlib.colors as mcolors


_PFIXES = ["dark", "deep", "medium"]
_AVAILABLE_COLORS = list(mcolors.CSS4_COLORS.keys())
_AVAILABLE_COLORS = [color for color in _AVAILABLE_COLORS if (
    any((a in color) for a in _PFIXES))]
_AVAILABLE_IDXS = np.arange(stop=len(_AVAILABLE_COLORS))


class TiledAxesIter:
    def __vertical_tile_iter__(self, idx: int):
        if self.iter_dim_nums == 0:
            return self.axes
        if self.iter_dim_nums == 1:
            return self.axes[idx]
        if self.iter_dim_nums == 2:
            return self.axes[idx % self.split_in][idx//self.split_in]

    def __horizontal_tile_iter__(self, idx: int):
        if self.iter_dim_nums == 0:
            return self.axes
        if self.iter_dim_nums == 1:
            return self.axes[idx]
        if self.iter_dim_nums == 2:
            return self.axes[idx//self.split_in][idx % self.split_in]

    def __init__(self, axes: Axes, slices_len: int, tile_direction: str, split_in: int, iter_dim_nums: Literal[0, 1, 2]):
        self.tile_direction = tile_direction
        self.iter_dim_nums = iter_dim_nums
        self.axes = axes
        self.slices_len = slices_len
        self.split_in = split_in
        self.axes_getter = self.__vertical_tile_iter__ if self.tile_direction == 'vertical' else self.__horizontal_tile_iter__

    def __getitem__(self, idx: int):
        return self.axes_getter(idx % self.slices_len)


class SlicesPlotter:
    @staticmethod
    def __get_axes_dimension(_dim_flated, slices_len, split_in):
        _axes_dim = 0
        if slices_len == 1:
            _axes_dim = 0
        elif _dim_flated > 1 and split_in > 1:
            _axes_dim = 2
        elif _dim_flated > 1 or split_in > 1:
            _axes_dim = 1
        elif _dim_flated > 1 and split_in == 1:
            _axes_dim = 1
        return _axes_dim

    def __create_fig_axes__(self):
        if self.slices is None:
            raise Exception("Cant Create Figure And Axes For 'None' Slices")
        else:
            slices_len = len(self.slices)
            _dim_flated = slices_len // self.split_in + slices_len % self.split_in
            _axes_dim = self.__get_axes_dimension(
                _dim_flated, slices_len, self.split_in)
            if self.tile_direction == 'vertical':
                rows = _dim_flated
                fig, axs = plt.subplots(rows, self.split_in)
                self.fig = fig
                self.axs = TiledAxesIter(
                    axs, slices_len, 'vertical', self.split_in, _axes_dim)
            elif self.tile_direction == 'horizontal':
                columns = _dim_flated
                fig, axs = plt.subplots(self.split_in, columns)
                self.fig = fig
                self.axs = TiledAxesIter(
                    axs, slices_len, 'horizontal', self.split_in, _axes_dim)
            else:
                raise Exception(
                    f"Invalid Tile Direction: '{self.tile_direction}'")

    def __init__(self, slices: List[np.ndarray] = None, split_in=1,
                 tile_direction: Literal['vertical',
                                         'horizontal'] = 'vertical',
                 title=None,
                 subplots_instance: Tuple[Figure, Axes] = None,
                 use_tight=True):
        self.tile_direction = tile_direction
        self.split_in = split_in
        self.figs_titles = None
        self.fig_title = None
        self.slices = slices
        self.usable_colors = None
        self.slices_len = len(slices)
        self.slices_drawed = False
        if subplots_instance is None:
            if self.slices is not None:
                self.__create_fig_axes__()
        else:
            self.fig, axes = subplots_instance
            _dim_flated = self.slices_len // self.split_in + self.slices_len % self.split_in
            _axes_dim = self.__get_axes_dimension(
                _dim_flated, self.slices_len, self.split_in)
            self.axs = TiledAxesIter(
                axes, self.slices_len, tile_direction, self.split_in, _axes_dim)
        if title:
            self.fig.suptitle(title, fontsize=22, weight='bold')
        self.fig.set_tight_layout(use_tight)

    def draw_fig_title(self, title:str):
        self.fig.suptitle(title, fontsize=22, weight='bold')
        return self
    
    def draw_titles(self, titles: List[str]):
        assert len(titles) == self.slices_len
        for ii in range(self.slices_len):
            canvas = self.axs[ii]
            canvas.set_title(titles[ii])
        return self
    
    def set_background_colors(self, bg_colors: List[str]):
        assert len(bg_colors) == self.slices_len
        for ii in range(self.slices_len):
            canvas = self.axs[ii]
            canvas.set_facecolor(bg_colors[ii])
        return self

    def set_usable_colors(self,colors:List[str]):
        if all([(color in _AVAILABLE_COLORS) for color in colors]):
            self.usable_colors = colors
        return self

    def set_random_usable_colors(self,amount:int):
         amount += 1
         r_idxs = np.random.choice(_AVAILABLE_IDXS, amount, replace=False)
         self.usable_colors = np.array(_AVAILABLE_COLORS)[r_idxs]
         return self

    IDX = int
    def draw_points(self,points:List[np.ndarray],markersize=2,
                    colors_maps:List[List[IDX]]=None,
                    edgesize=0, edgecolor='black'):
        if self.usable_colors is None:
            color_amount = np.max(colors_maps) if colors_maps is not None else 10
            self.set_random_usable_colors(color_amount)
        for ii in range(len(points)):
            if points[ii] is not None:
                canvas = self.axs[ii]
                c_map = colors_maps[ii] if (colors_maps is not None and ii < len(colors_maps)) else None
                for idx,point in enumerate(points[ii]):
                    if edgesize > 0:
                        canvas.plot(point[0], point[1], 'o', markersize=(markersize+edgesize), color=edgecolor) 
                    color = self.usable_colors[c_map[idx]] if c_map is not None else self.usable_colors[0]
                    canvas.plot(point[0], point[1], 'o', markersize=markersize, color=color)
        return self

    def draw_vectors(self,points:List[np.ndarray],vectors_map:List[np.ndarray],
                     thick:int=1,color='red'):
        assert len(points) == len(vectors_map)
        for ii in range(len(vectors_map)):
                vector_map = vectors_map[ii]
                if vector_map is not None:
                    canvas = self.axs[ii]
                    point_grid = points[ii]
                    for idx in range(len(vector_map)-1):
                        x0, y0 = point_grid[vector_map[idx]]
                        x1, y1 = point_grid[vector_map[idx+1]]
                        # Vector Head Compensation
                        a = np.arctan2((y1-y0),(x1-x0))
                        cx,cy = (thick/2)*np.cos(a) , (thick/2)*np.sin(a)  
                        canvas.arrow(x0+cx, y0+cy, (x1-x0)-2.5*cx, (y1-y0)-2.5*cy,
                                     width=thick/4,head_width=thick,head_length=thick, color=color)
        return self


    def draw_slices(self, marker_size=1, line_kind="o-", bg_color='white'):
        for ii in range(self.slices_len):
            if self.slices[ii] is not None:
                canvas = self.axs[ii]
                for slice in self.slices[ii]:
                    canvas.plot(slice[:, 0], slice[:, 1], line_kind, markersize=marker_size)
        self.slices_drawed = True
        return self

    def show(self):
        if not self.slices_drawed:
            self.draw_slices()
        plt.show()
        
    def save(self,file_path:str,dpi=250):
        if not self.slices_drawed:
            self.draw_slices()
        try:
            self.fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"An Error Ocurred {e}")
        finally:
            print(f"File saved at : {file_path}")



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

            if points_grids_color_idx_map is None or (ii+1) > len(points_grids_color_idx_map) or points_grids_color_idx_map[ii] is None:
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
                    _plotter.arrow(x0, y0, x1-x0, y1-y0,
                                   head_width=4, head_length=1, color="red")

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
