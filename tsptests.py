import numpy as np
import os
from typing import List
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
            distance_matrix[i, j] = np.sqrt((pointsA[i, 0] - pointsB[j, 0]) ** 2 +
                                                (pointsA[i, 1] - pointsB[j, 1]) ** 2)

    return distance_matrix


@njit(cache=True, parallel=True)
def compute_angle_delta_mean(points: np.ndarray, route: List[int]) -> float:
    n = len(route)
    if n < 3: return 0.0
    angle_deltas = np.zeros(n - 2, dtype=np.float64)
    for i in range(1, n - 1):
        p1 = points[route[i - 1]]
        p2 = points[route[i]]
        p3 = points[route[i + 1]]
        v1 = p2 - p1
        v2 = p3 - p2
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        delta = np.abs(angle2 - angle1)
        if delta > np.pi:
            delta = 2 * np.pi - delta
        angle_deltas[i - 1] = delta
    return np.mean(angle_deltas)

def sort_points_up_right(points: np.ndarray) -> np.ndarray:
    # Sort points by y descending and then by x ascending
    _remap = np.lexsort((-points[:, 1], points[:, 0]))
    sorted_points = points[_remap]
    return _remap,sorted_points

def generateGreedyPath(centers: np.ndarray) -> np.ndarray:
    _remap,sorted_centers = sort_points_up_right(centers)
    n_centers = len(sorted_centers)
    path_idxs = [0]  
    visited = np.zeros(n_centers, dtype=bool)  
    visited[0] = True  
    for _ in range(1, n_centers):
        base_point = sorted_centers[path_idxs[-1]]
        shortest_dist = np.inf
        shortest_idx = None
        for idx in range(n_centers):
            if not visited[idx]:
                euc_dist = np.linalg.norm(sorted_centers[idx] - base_point)
                if euc_dist < shortest_dist and euc_dist != 0.0:
                    shortest_dist = euc_dist
                    shortest_idx = idx
        path_idxs.append(shortest_idx)
        visited[shortest_idx] = True
    return np.array([_remap[e] for e in path_idxs])

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
