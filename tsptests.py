import numpy as np
import os
from numba import njit, prange
from kmeans_tests import generatePointsAndClusters, geometrys_from_txt_nan_separeted, ShowGeometrys
from py2opt.routefinder import RouteFinder


@njit(cache=True, parallel=True)
def compute_distance_matrix_numba_parallel(pointsA,pointsB):
    ''' rows: idxs points A, columns: idxs points B '''
    row_points = pointsA.shape[0]
    col_points = pointsB.shape[0]
    distance_matrix = np.zeros((row_points, col_points), dtype=np.float32)
    for i in prange(row_points):
        for j in range(col_points):
            if i != j:
                distance_matrix[i, j] = np.sqrt((pointsA[i, 0] - pointsB[j, 0]) ** 2 +
                                                (pointsA[i, 1] - pointsB[j, 1]) ** 2)

    return distance_matrix


if __name__ == "__main__":
    test_file = "assets/txt/formas/rabbit.txt"
    DISTANCE = 10
    CLUSTER_N = 2
    ITERATIONS = 20
    forma = geometrys_from_txt_nan_separeted(test_file)
    grid, pred, centers = generatePointsAndClusters(
        forma, distance=DISTANCE, clusters_n=CLUSTER_N)
    grid_clusters = [list() for _ in range(np.max(pred)+1)]
    for jj, p in enumerate(pred):
        grid_clusters[p].append(grid[jj])
    distance_matrices = [compute_distance_matrix_numba_parallel(
        np.array(cluster)) for cluster in grid_clusters]
    best_routes = []
    for ii, distance_matrix in enumerate(distance_matrices):
        points_idxs = [i for i in range(distance_matrix.shape[0])]
        route_finder = RouteFinder(
            distance_matrix, points_idxs, iterations=ITERATIONS, verbose=False)
        distance, best_route = route_finder.solve()
        OUTPUT_PATH = f"outputs/tsp/{os.path.basename(test_file).split('.')[0]}_{ii}_tsp.png"
        ShowGeometrys([forma], points_grids=[grid_clusters[ii]],
                      points_grids_color_idx_map=[pred],
                      points_grids_clusters_centers=[
                          [grid_clusters[ii][best_route[0]], grid_clusters[ii][best_route[-1]]]],
                      points_grids_vector_idx_map=[best_route],
                      background_color="black",
                      file_name=OUTPUT_PATH,
                      spliter=1,
                      fig_title=f"distance: {distance}",
                      show_plot=False
                      )
