from typing import List
from numba import njit,prange
import numpy as np


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
