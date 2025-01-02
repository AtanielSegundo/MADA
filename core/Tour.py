import numpy as np
from core.transform import sort_points_up_right
from numba import njit
from core.TSP.solver import Solver
from core.Points.Grid import Clusters,Cluster
from core.Points.operations import compute_distance_matrix_numba_parallel
from typing import Type

class Tour:
    def __init__(self, path: np.ndarray):
        assert len(path.shape) == 1, "Path must be a 1D array"
        self.path = path
        self.len = path.shape[0]
    def remap(self, remap_by_idx: np.ndarray):
        self.path = remap_by_idx[self.path]

class TourEnd:
    def __init__(self,temporary_folder:str):
        self.clusters_big_path = None
        self.clusters_centers_tour = None
        self.temporary_folder  = temporary_folder
        self.relative_lenght = 0
        self.current_start_end_idx = 0
        self.start_end = ...
    
    def set_clusters_big_path(self,clusters:Clusters,Solver:Type[Solver],runs:int=2):
        pass
    
    def get_current_start_end(self,clusters:Clusters):
        pass
    
    def remap_tour_path(self,tour:Tour):
        pass  


class closeEnd(TourEnd):
    def __init__(self, temporary_folder):
        super().__init__(temporary_folder)
    def set_clusters_big_path(self, clusters, Solver, runs = 2):
        if clusters.set is not None:
            self.clusters_centers_tour = Tour(np.arange(clusters.n_clusters))
    def get_current_start_end(self, clusters):
        self.start_end = None
        if clusters.centers is None and clusters.set is None:
            now_start_end_idx = 0
            self.current_start_end_idx = None
        else:
            now_start_end_idx = self.current_start_end_idx 
            self.current_start_end_idx += 1
            if self.current_start_end_idx >= clusters.n_clusters:
                self.current_start_end_idx = None
        
        return now_start_end_idx,now_start_end_idx
            

class openEnd(TourEnd):
    def __init__(self, temporary_folder):
        super().__init__(temporary_folder)

    def set_clusters_big_path(self, clusters:Clusters, Solver, runs = 2):
        if clusters.centers is None and clusters.set is None:
            self.start_end = None
        else:
            _remap,centers = sort_points_up_right(clusters.centers)
            tsp_solver = Solver(self.temporary_folder)
            _,centers_tour = tsp_solver.solve(centers,runs=runs)
            centers_tour.remap(_remap)
            # Determinando o ponto de inicio e fim por clusters mais proximos
            # Utiliza os indices do point grid para poupar espaço
            # [[0,1],[2,3]]
            clusters_total_lenght = 0
            self.clusters_big_path = [[0,1]]
            for ii in range(centers_tour.len - 1) :
                cluster = clusters.set[centers_tour.path[ii]]
                next_cluster = clusters.set[centers_tour.path[ii+1]]
                distance_matrix = compute_distance_matrix_numba_parallel(next_cluster.cluster,cluster.cluster)
                start_next_cluster,end_current_cluster = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                clusters_total_lenght += distance_matrix[start_next_cluster,end_current_cluster] 
                self.clusters_big_path[ii][1] = end_current_cluster
                self.clusters_big_path.append([start_next_cluster,1])
            #print(self.clusters_big_path)
            self.relative_lenght += clusters_total_lenght
            self.clusters_centers_tour = centers_tour
            #print(self.clusters_centers_tour.path)
    
    def get_current_start_end(self,clusters:Clusters):
        # GREEDY IDX => centers_tour.path
        current_cluster_idx = 0
        now_start_end_idx = 0
        if self.current_start_end_idx is None:
            self.start_end = None
        elif self.start_end is None:
            self.current_start_end_idx = None
        else:
            now_start_end_idx = self.current_start_end_idx
            current_cluster_idx = self.clusters_centers_tour.path[self.current_start_end_idx]
            cluster:Cluster = clusters.set[current_cluster_idx]
            start_node, end_node = self.clusters_big_path[self.current_start_end_idx]
            #print(current_cluster_idx," | ",start_node,end_node)
            if start_node == end_node :
                start_node = end_node-1 if end_node-1 > 0 else end_node+1
            if start_node is not None and end_node is not None:
                temp_ = cluster.remap_idxs[0] 
                cluster.remap_idxs[0] = cluster.remap_idxs[start_node] 
                cluster.remap_idxs[start_node] = temp_
                temp_ = np.array([0,0])
                temp_[0] = cluster.cluster[start_node][0]
                temp_[1] = cluster.cluster[start_node][1]
                cluster.cluster[start_node][0] = cluster.cluster[0][0] 
                cluster.cluster[start_node][1] = cluster.cluster[0][1]
                cluster.cluster[0][0] = temp_[0]
                cluster.cluster[0][1] = temp_[1]
            self.start_end = [0,end_node]
            if end_node == 0:
                self.start_end = [0,start_node]
            self.current_start_end_idx += 1
            if self.current_start_end_idx >= clusters.n_clusters:
                self.current_start_end_idx = None
        return current_cluster_idx,now_start_end_idx
    
    def remap_tour_path(self,tour:Tour):
        tour.path = tour.path[:-1]        


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
