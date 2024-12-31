from typing import List
from sklearn.cluster import KMeans
from core.geometry import fill_geometrys_with_points
from numba import njit,prange
import numpy as np


def sort_points_up_right(points: np.ndarray) -> np.ndarray:
    # Sort points by y descending and then by x ascending
    _remap = np.lexsort((-points[:, 1], points[:, 0]))
    sorted_points = points[_remap]
    return _remap,sorted_points


def generatePointsAndClusters(forma: List[np.ndarray], clusters_n=6, seed=None,
                              clusters_iters=600, distance=5, fliped_y=False,figure_sep=None):
    figure_sep = figure_sep or 0.5
    pointgrid = fill_geometrys_with_points(forma, distance, figure_sep=figure_sep, fliped_y=fliped_y)
    if len(pointgrid) < clusters_n:
        print("Not enough points in pointgrid")
        return None, None, None
    k_means = KMeans(n_clusters=clusters_n, max_iter=clusters_iters,random_state=seed)
    k_means.fit(pointgrid)
    clusters_centers = k_means.cluster_centers_
    predictions = k_means.predict(pointgrid)
    return pointgrid, predictions, clusters_centers


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
