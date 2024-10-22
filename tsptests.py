import numpy as np
import os
from numba import njit, prange
from kmeans_tests import generatePointsAndClusters, geometrys_from_txt_nan_separeted, ShowGeometrys
from py2opt.routefinder import RouteFinder


@njit(cache=True, parallel=True)
def compute_distance_matrix_numba_parallel(points):
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points), dtype=np.float64)
    for i in prange(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i, j] = np.sqrt((points[i, 0] - points[j, 0]) ** 2 +
                                                (points[i, 1] - points[j, 1]) ** 2)

    return distance_matrix


if __name__ == "__main__":
    test_file = "assets/txt/formas/rabbit.txt"
    OUTPUT_PATH = f"outputs/tsp/{os.path.basename(test_file).split('.')[0]}"
    DISTANCE = 20
    CLUSTER_N = 3
    ITERATIONS = 10
    forma = geometrys_from_txt_nan_separeted(test_file)
    grid, pred, centers = generatePointsAndClusters(
        forma, distance=DISTANCE, clusters_n=CLUSTER_N)
    grid_clusters = [list() for _ in range(np.max(pred)+1)]
    for jj, p in enumerate(pred):
        grid_clusters[p].append(grid[jj])
    distance_matrices = [compute_distance_matrix_numba_parallel(
        np.array(cluster)) for cluster in grid_clusters]
    best_routes = []
    for distance_matrix in distance_matrices:
        points_idxs = [i for i in range(distance_matrix.shape[0])]
        route_finder = RouteFinder(
            distance_matrix, points_idxs, iterations=ITERATIONS, verbose=False)
        _, best_route = route_finder.solve()
        best_routes.append(best_route)
    ShowGeometrys([forma]*CLUSTER_N, points_grids=grid_clusters,
                  points_grids_color_idx_map=[pred]*CLUSTER_N,
                  points_grids_clusters_centers=[
                      [grid[best_routes[idx][0]], grid[best_routes[idx][-1]]]
                      for idx, grid in enumerate(grid_clusters)],
                  points_grids_vector_idx_map=best_routes,
                  background_color="black")
