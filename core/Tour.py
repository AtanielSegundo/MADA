from numba import njit,prange
import numpy as np


@njit(cache=True)
def generateCH(points):
    n = points.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    path = np.empty(n, dtype=np.int32)
    current_node = 0
    path[0] = current_node
    visited[current_node] = True
    for step in range(1, n):
        next_node = -1
        coords = points[current_node]
        dx = coords[0] - points[:, 0]
        dy = coords[1] - points[:, 1]
        distances_squared = dx**2 + dy**2
        for i in range(n):
            if not visited[i] and (next_node == -1 or distances_squared[i] < distances_squared[next_node] or (distances_squared[i] == distances_squared[next_node] and i > next_node)):
                next_node = i
        path[step] = next_node
        visited[next_node] = True
        current_node = next_node
    return path


def generateDummyTour(start_node: int, end_node: int, tour_len: int):
    """
    Generates a tour array for LKH that starts at `start_node` and ends at `end_node`.
    The dummy node is the last node in the tour.
    """
    tour = np.zeros((tour_len), dtype=np.int32)
    tour[0] = start_node  
    tour[-1] = tour_len - 1  
    tour[-2] = end_node  
    remaining_nodes = [i for i in range(tour_len - 1) if i not in {start_node, end_node}]
    np.random.shuffle(remaining_nodes) 
    tour[1:-2] = remaining_nodes
    return tour

@njit(cache=True)
def generateCH_with_dummy(points, start_node, end_node):
    n = points.shape[0]
    visited = np.zeros(n+1, dtype=np.bool_)
    path = np.zeros(n+1, dtype=np.int32)
    path[-1] = n
    current_node = start_node
    path[0] = current_node
    visited[current_node] = True
    visited[n] = True
    for step in range(1,n):
        next_node = -1
        coords = points[current_node]
        dx = coords[0] - points[:, 0]
        dy = coords[1] - points[:, 1]
        distances_squared = dx**2 + dy**2
        for i in range(n):
            if not visited[i] and (next_node == -1 or distances_squared[i] < distances_squared[next_node] or (distances_squared[i] == distances_squared[next_node] and i > next_node)):
                next_node = i
        path[step] = next_node
        visited[next_node] = True
        current_node = next_node

    for k in range(n):
        if path[k] == end_node:
            path[k], path[-2] = path[-2], path[k]
            break
        
    return path
