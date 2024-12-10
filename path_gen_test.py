from typing import Tuple,List
from core.visualize._2d import SlicesPlotter
from core.Layer import Layer
from lkh_clusters import generateCH,generateCH_with_dummy,generateDummyTour
from kmeans_tests import generatePointsAndClusters,fill_geometrys_with_points

if __name__ == "__main__":
    SEED     = 777
    DISTANCE = 5
    BORDERS_DISTANCE = 0
    fliped_y = False
    FILE = "assets/svg/rabbit.svg"
    layer = Layer.From(FILE, scale=0.5)
    forma = layer.data
    _plt = SlicesPlotter([None,None,None],tile_direction="horizontal")
    _plt.set_random_usable_colors(6)
    _plt.set_background_colors(['black','black','black'])
    grid = fill_geometrys_with_points(forma,delta=DISTANCE,figure_sep=BORDERS_DISTANCE,fliped_y=layer.is_y_flipped)
    path_dummy = generateDummyTour(0,20, len(forma[0])+1)
    path_ch_dummy = generateCH_with_dummy(grid, 0,20)
    path_ch = generateCH(grid)
    _plt.draw_vectors([grid,grid,grid],[path_ch,path_dummy[:-1],path_ch_dummy[:-1]])
    _plt.draw_points([[grid[path_ch[0]], grid[path_ch[-1]]],
                      [grid[path_dummy[:-1][0]], grid[path_dummy[:-1][-1]]],
                      [grid[path_ch_dummy[:-1][0]], grid[path_ch_dummy[:-1][-1]]]],
                     colors_maps=[[0,1],[0,1],[0,1]])
    _plt.show()
