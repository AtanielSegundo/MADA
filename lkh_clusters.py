import os
import shutil
import time
from numba import njit
import numpy as np
import lkh
from typing import Tuple,List
from tqdm import tqdm
from core.visualize._2d import SlicesPlotter
from commons.utils.clipper import readPathSVG
from kmeans_tests import generatePointsAndClusters,fill_geometrys_with_points
from tsptests import compute_distance_matrix_numba_parallel,compute_angle_delta_mean,sort_points_up_right

class Cluster:
    def __init__(self):
        self.cluster = []
        self.route = []
        self.remap_idxs = []

def writeDistanceMatrixProblemFile(distance_matrix: np.ndarray, file_name: str, dummy_edge: Tuple[int, int] = None):
    min_dist = 1e-4
    max_dist = 1e4
    with open(file_name, "w") as f:
        f.write("NAME: DistanceMatrixProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")

        start_node, end_node = dummy_edge if dummy_edge else (None, None)
        for i, row in enumerate(distance_matrix):
            formatted_row = [
                str(min_dist) if j == dimension - 1 and (i == start_node or i == end_node) else f"{val:.3f}" 
                for j, val in enumerate(np.append(row, [max_dist] if dummy_edge else []))
            ]
            f.write(" ".join(formatted_row) + "\n")
        if dummy_edge:
            dummy_row = [str(min_dist) if j == start_node or j == end_node or j == dimension - 1 else str(max_dist) for j in range(dimension)]
            f.write(" ".join(dummy_row) + "\n")

def writePointsProblemFile(points:np.ndarray,file_name:str) :
    dimension = points.shape[0]
    with open(file_name,"w") as f:
        f.write("NAME: PointsProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for idx,point in enumerate(points):
            f.write(f"{idx+1} ")
            f.write(" ".join(map(lambda e: f"{e:.3e}",point)))
            f.write("\n")

            
def writeInitialTourFile(tour:np.ndarray,file_name:str) :
    dimension = len(tour)
    with open(file_name,"w") as f:
        f.write("NAME: InitialTourFile\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("TOUR_SECTION\n")
        f.write("\n".join(map(lambda e : str(e+1),tour)))
        f.write("\n-1")

def writeDistanceMatrixProblemFile(distance_matrix: np.ndarray, file_name: str, dummy_edge: Tuple[int, int] = None):
    with open(file_name, "w") as f:
        dimension = distance_matrix.shape[0] + (1 if dummy_edge else 0)
        f.write("NAME: DistanceMatrixProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        start_node, end_node = dummy_edge if dummy_edge else (None, None)
        for i, row in enumerate(distance_matrix):
            if dummy_edge:
                formatted_row = [
                    "1e-6" if j == dimension - 1 and (i == start_node or i == end_node) else f"{val:.3f}" 
                    for j, val in enumerate(np.append(row, [1e6] if dummy_edge else []))
                ]
            else:
                formatted_row = [f"{val:.3f}" for val in row]
            f.write(" ".join(formatted_row) + "\n")

        if dummy_edge:
            dummy_row = ["1e-6" if j == start_node or j == end_node else "1e6" for j in range(dimension)]
            f.write(" ".join(dummy_row) + "\n")
            
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

angle_delta = float
path_lenght = float
exec_time = float
number_of_points = int
Grid = np.ndarray

def generateClustersPath(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,save_fig_path=None,runs:int=5,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    CLUSTER_N = n_clusters
    LKH_RUNS = runs
    temporary_folder = "outputs/temp/generate_clusters_path/"
    grid, pred, _ = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED,fliped_y=fliped_y)
    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster,)
    
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(CLUSTER_N)
        plotter.draw_points([grid],colors_maps=[pred])
    
    os.makedirs(temporary_folder,exist_ok=True)
    
    total_lenght = 0
    _exec_time = 0

    for idx, cluster in enumerate(grid_clusters):    
        _ta = time.time()
        dst_mat = compute_distance_matrix_numba_parallel(cluster.cluster,cluster.cluster)
        mat_file_name = f"_c{idx}.txt"
        mat_file_name = os.path.join(temporary_folder, mat_file_name)
        writeDistanceMatrixProblemFile(dst_mat,mat_file_name)
        lenght,best_route = lkh.solve(LKH_PATH, problem_file=mat_file_name, runs=LKH_RUNS,seed=SEED)
        best_route = np.array(best_route[0]) - 1
        cluster.route = best_route
        total_lenght += lenght
        _exec_time += time.time() - _ta
        if save_fig_path is not None:
            plotter.set_background_colors(['black'])
            plotter.draw_points([[cluster.cluster[best_route[0]],
                                cluster.cluster[best_route[-1]]]],
                                colors_maps=[[0,1,2]],
                                markersize=3,edgesize=1)
            plotter.draw_vectors([cluster.cluster],[best_route],thick=1.25)

    b_cluster = Cluster()
    b_cluster.cluster = grid
    route_remaped = []

    for _idx,_cluster in enumerate(grid_clusters):
        remap = [_cluster.remap_idxs[idx] for idx in _cluster.route]
        if _idx > 0 :
            first_new_point = grid[remap[0]]
            last_point = grid[route_remaped[-1]]
            total_lenght += np.linalg.norm(np.array(last_point)-np.array(first_new_point))
            if save_fig_path is not None:
                plotter.draw_vectors([np.array([last_point,first_new_point])],[[0,1]],thick=1.25,color="green")
        route_remaped.extend(remap)
    angle_delta_mean = compute_angle_delta_mean(grid,route_remaped)
    if save_fig_path is not None:
        plotter.draw_fig_title(total_lenght.__ceil__())
        plotter.save(save_fig_path)
    shutil.rmtree(temporary_folder)
    return _exec_time,total_lenght,angle_delta_mean

def generateClustersMergedPath(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,save_fig_path=None,runs:int=5,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    CLUSTER_N = n_clusters
    LKH_RUNS = runs
    temporary_folder = "outputs/temp/merged_clusters_path/"
    grid, pred, _ = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED,fliped_y=fliped_y)
    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster,)
    
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(CLUSTER_N)
        plotter.draw_points([grid],colors_maps=[pred])
    os.makedirs(temporary_folder,exist_ok=True)
    _exec_time = 0
    _total_lenght = 0
    for idx, cluster in enumerate(grid_clusters):    
        _ta = time.time()
        dst_mat = compute_distance_matrix_numba_parallel(cluster.cluster,cluster.cluster)
        mat_file_name = f"_c{idx}.txt"
        mat_file_name = os.path.join(temporary_folder, mat_file_name)
        writeDistanceMatrixProblemFile(dst_mat,mat_file_name)
        lenght,best_route = lkh.solve(LKH_PATH, problem_file=mat_file_name, runs=LKH_RUNS,seed=SEED)
        _total_lenght += lenght
        best_route = np.array(best_route[0]) - 1
        cluster.route = best_route
        _exec_time += time.time() - _ta
    b_cluster = Cluster()
    b_cluster.cluster = grid
    routes_remaped = []
    for _cluster in grid_clusters:
        remap = [_cluster.remap_idxs[idx] for idx in _cluster.route]
        routes_remaped.extend(remap)
    _ta = time.time()
    dst_mat = compute_distance_matrix_numba_parallel(b_cluster.cluster,b_cluster.cluster)
    mat_file_name = "c_merged.txt"
    mat_file_name = os.path.join(temporary_folder, mat_file_name)
    writeDistanceMatrixProblemFile(dst_mat, mat_file_name)
    merge_tour_file = mat_file_name.replace(".txt", f"_mergetour_.txt")
    writeInitialTourFile(routes_remaped, merge_tour_file)
    lenght,best_route = lkh.solve(LKH_PATH,runs=1,
                                    time_limit=(_exec_time.__ceil__()),
                                    optimum=((_total_lenght*(1.5)).__ceil__()),
                                    max_breadth=10,
                                    problem_file=mat_file_name,
                                    initial_tour_file=merge_tour_file,
                                    merge_tour_files=[merge_tour_file])
    _exec_time += time.time() - _ta
    best_route = np.array(best_route[0]) - 1
    angle_delta_mean = compute_angle_delta_mean(grid,best_route)
    if not save_fig_path is None:
        plotter.set_background_colors(['black'])
        plotter.draw_points([[grid[best_route[0]],grid[best_route[-1]]]],colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid],[best_route],thick=1.25)
        plotter.draw_fig_title(lenght.__ceil__())
        plotter.save(save_fig_path)

    shutil.rmtree(temporary_folder)
    return _exec_time,lenght,angle_delta_mean


def generateCHRaw(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,runs:int=5,save_fig_path=None,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    LKH_RUNS = runs
    temporary_folder = "outputs/temp/generate_path_ch_raw/"
    grid = fill_geometrys_with_points(forma,delta=DISTANCE,figure_sep=BORDERS_DISTANCE,fliped_y=fliped_y)
    ch_tour = generateCH(grid)
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(3)
        plotter.draw_points([None])
    os.makedirs(temporary_folder,exist_ok=True)
    _ta = time.time()
    dst_mat = compute_distance_matrix_numba_parallel(grid,grid)
    mat_file_name = f"_ch_raw.txt"
    mat_file_name = os.path.join(temporary_folder, mat_file_name)
    _initial_tour_name = "_ch_tour_raw.txt"
    _initial_tour_file = os.path.join(temporary_folder, _initial_tour_name)
    writeInitialTourFile(ch_tour, _initial_tour_file)
    writeDistanceMatrixProblemFile(dst_mat,mat_file_name)
    total_lenght,best_route = lkh.solve(LKH_PATH, initial_tour_file=_initial_tour_file,
                                        problem_file=mat_file_name, runs=LKH_RUNS, seed=SEED)
    best_route = np.array(best_route[0]) - 1
    _exec_time = time.time() - _ta
    if not save_fig_path is None:
        plotter.set_background_colors(['black'])
        plotter.draw_points([[grid[best_route[0]],grid[best_route[-1]]]],colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid],[best_route],thick=1.25)
    angle_delta_mean = compute_angle_delta_mean(grid,best_route)
    if save_fig_path is not None:
        plotter.draw_fig_title(total_lenght.__ceil__())
        plotter.save(save_fig_path)
    shutil.rmtree(temporary_folder)
    return _exec_time,total_lenght,angle_delta_mean

    
def generatePathRaw(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,runs:int=5,save_fig_path=None,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta,number_of_points,Grid]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    LKH_RUNS = runs
    temporary_folder = "outputs/temp/generate_path_raw/"
    grid = fill_geometrys_with_points(forma,delta=DISTANCE,figure_sep=BORDERS_DISTANCE,fliped_y=fliped_y)
    number_of_points = len(grid)
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(3)
        plotter.draw_points([None])
    os.makedirs(temporary_folder,exist_ok=True)
    _ta = time.time()
    dst_mat = compute_distance_matrix_numba_parallel(grid,grid)
    mat_file_name = f"_raw.txt"
    mat_file_name = os.path.join(temporary_folder, mat_file_name)
    writeDistanceMatrixProblemFile(dst_mat,mat_file_name)
    total_lenght,best_route = lkh.solve(LKH_PATH, problem_file=mat_file_name, runs=LKH_RUNS, seed=SEED)
    best_route = np.array(best_route[0]) - 1
    _exec_time = time.time() - _ta
    if not save_fig_path is None:
        plotter.set_background_colors(['black'])
        plotter.draw_points([[grid[best_route[0]],grid[best_route[-1]]]],colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid],[best_route],thick=1.25)
    angle_delta_mean = compute_angle_delta_mean(grid,best_route)
    if save_fig_path is not None:
        plotter.draw_fig_title(total_lenght.__ceil__())
        plotter.save(save_fig_path)
    shutil.rmtree(temporary_folder)
    return _exec_time,total_lenght,angle_delta_mean,number_of_points,grid
        

def generatePathOpenClusters(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,runs:int=5,save_fig_path=None,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    CLUSTER_N = n_clusters
    LKH_RUNS = runs
    _total_lenght = 0
    _exec_time = 0
    temporary_folder = "outputs/temp/open_clusters_path/"
    grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED,fliped_y=fliped_y)
    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster,)
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(CLUSTER_N)
        plotter.draw_points([None],colors_maps=[pred])
    os.makedirs(temporary_folder,exist_ok=True)
    # Organizando os clusters usando lkh
    _ta = time.time()
    #greedy_idxs = generateGreedyPath(centers)
    _remap,centers = sort_points_up_right(centers)
    centers_prb_f = os.path.join(temporary_folder,"centers_dst.txt")
    writeDistanceMatrixProblemFile(compute_distance_matrix_numba_parallel(centers,centers),centers_prb_f)
    _,centers_route = lkh.solve(LKH_PATH,problem_file=centers_prb_f,runs=LKH_RUNS,seed=SEED)
    greedy_idxs = [_remap[e-1] for e in centers_route[0]]
    # Determinando o ponto de inicio e fim por clusters mais proximos
    # Utiliza os indices do point grid para poupar espaço
    # [[0,1],[2,3]]
    clusters_start_ends = [[0,1]]
    for ii in range(len(greedy_idxs) - 1) :
        cluster = grid_clusters[greedy_idxs[ii]]
        next_cluster = grid_clusters[greedy_idxs[ii+1]]
        distance_matrix = compute_distance_matrix_numba_parallel(next_cluster.cluster,cluster.cluster)
        start_next_cluster,end_current_cluster = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        _total_lenght += distance_matrix[start_next_cluster,end_current_cluster] 
        clusters_start_ends[ii][1] = end_current_cluster
        clusters_start_ends.append([start_next_cluster,1])
    _exec_time += time.time() - _ta
    
    for idx, greedy_idx in enumerate(greedy_idxs):
        _ta = time.time()
        cluster = grid_clusters[greedy_idx]
        start_node, end_node = clusters_start_ends[idx]
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
        dummy_edge = (0,end_node)
        if end_node == 0:
            dummy_edge = (0,start_node)
            
        dst_mat = compute_distance_matrix_numba_parallel(cluster.cluster,cluster.cluster)
        mat_file_name = f"_c{idx}.txt"
        mat_file_name = os.path.join(temporary_folder, mat_file_name)
        dummy_tour = generateDummyTour(start_node,end_node,cluster.cluster.shape[0]+1)
        initial_tour_file = mat_file_name.replace(".txt", "_dummy_tour.txt")
        writeInitialTourFile(dummy_tour,initial_tour_file)
        writeDistanceMatrixProblemFile(dst_mat, mat_file_name,dummy_edge=dummy_edge)
        lenght,best_route = lkh.solve(LKH_PATH, initial_tour_file=initial_tour_file, problem_file=mat_file_name, runs=LKH_RUNS)
        best_route = np.array(best_route[0])[:-1] - 1
        _total_lenght += lenght
        cluster.route = best_route
        _exec_time += time.time() - _ta
    
    open_route_merged = []
    for greedy_idx in greedy_idxs:
        cluster = grid_clusters[greedy_idx]
        for idx in cluster.route:
            open_route_merged.append(cluster.remap_idxs[idx])
    angle_delta_mean = compute_angle_delta_mean(grid,open_route_merged)
    if not save_fig_path is None:
        plotter.set_background_colors(['black'])
        plotter.draw_points([[grid[open_route_merged[0]],grid[open_route_merged[-1]]]],
                            colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid],[open_route_merged],thick=1.25)
        plotter.draw_fig_title(_total_lenght.__ceil__())
        plotter.save(save_fig_path)
    
    shutil.rmtree(temporary_folder)
    return _exec_time,_total_lenght,angle_delta_mean


def generatePathCHOpenClusters(n_clusters:int,forma:List[np.ndarray],distance:float,seed:bool,runs:int=5,save_fig_path=None,fliped_y=False) -> Tuple[exec_time,path_lenght,angle_delta]:
    LKH_PATH = "LKH.exe"
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    CLUSTER_N = n_clusters
    LKH_RUNS = runs
    _total_lenght = 0
    _exec_time = 0
    temporary_folder = "outputs/temp/open_ch_clusters_path/"
    grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED,fliped_y=fliped_y)
    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster,)
    if not save_fig_path is None: 
        base_path = os.path.dirname(save_fig_path)
        os.makedirs(base_path,exist_ok=True)
        plotter = SlicesPlotter([None],tile_direction='horizontal')
        plotter.set_random_usable_colors(CLUSTER_N)
        plotter.draw_points([None],colors_maps=[pred])
    os.makedirs(temporary_folder,exist_ok=True)
    # Organizando os clusters usando lkh
    _ta = time.time()
    #greedy_idxs = generateGreedyPath(centers)
    _remap,centers = sort_points_up_right(centers)
    centers_prb_f = os.path.join(temporary_folder,"centers_dst.txt")
    writeDistanceMatrixProblemFile(compute_distance_matrix_numba_parallel(centers,centers),centers_prb_f)
    _,centers_route = lkh.solve(LKH_PATH,problem_file=centers_prb_f,runs=LKH_RUNS,seed=SEED)
    greedy_idxs = [_remap[e-1] for e in centers_route[0]]
    # Determinando o ponto de inicio e fim por clusters mais proximos
    # Utiliza os indices do point grid para poupar espaço
    # [[0,1],[2,3]]
    clusters_start_ends = [[0,1]]
    for ii in range(len(greedy_idxs) - 1) :
        cluster = grid_clusters[greedy_idxs[ii]]
        next_cluster = grid_clusters[greedy_idxs[ii+1]]
        distance_matrix = compute_distance_matrix_numba_parallel(next_cluster.cluster,cluster.cluster)
        start_next_cluster,end_current_cluster = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        _total_lenght += distance_matrix[start_next_cluster,end_current_cluster] 
        clusters_start_ends[ii][1] = end_current_cluster
        clusters_start_ends.append([start_next_cluster,1])
    _exec_time += time.time() - _ta
    
    for idx, greedy_idx in enumerate(greedy_idxs):
        _ta = time.time()
        cluster = grid_clusters[greedy_idx]
        start_node, end_node = clusters_start_ends[idx]
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
        dummy_edge = (0,end_node)
        if end_node == 0:
            dummy_edge = (0,start_node)
            
        dst_mat = compute_distance_matrix_numba_parallel(cluster.cluster,cluster.cluster)
        mat_file_name = f"_c{idx}.txt"
        mat_file_name = os.path.join(temporary_folder, mat_file_name)
        dummy_tour = generateCH_with_dummy(cluster.cluster,start_node,end_node)
        initial_tour_file = mat_file_name.replace(".txt", "_dummy_ch_tour.txt")
        writeInitialTourFile(dummy_tour,initial_tour_file)
        writeDistanceMatrixProblemFile(dst_mat, mat_file_name,dummy_edge=dummy_edge)
        lenght,best_route = lkh.solve(LKH_PATH, initial_tour_file=initial_tour_file, problem_file=mat_file_name, runs=LKH_RUNS)
        best_route = np.array(best_route[0])[:-1] - 1
        _total_lenght += lenght
        cluster.route = best_route
        _exec_time += time.time() - _ta
    
    open_route_merged = []
    for greedy_idx in greedy_idxs:
        cluster = grid_clusters[greedy_idx]
        for idx in cluster.route:
            open_route_merged.append(cluster.remap_idxs[idx])
    angle_delta_mean = compute_angle_delta_mean(grid,open_route_merged)
    if not save_fig_path is None:
        plotter.set_background_colors(['black'])
        plotter.draw_points([[grid[open_route_merged[0]],grid[open_route_merged[-1]]]],
                            colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid],[open_route_merged],thick=1.25)
        plotter.draw_fig_title(_total_lenght.__ceil__())
        plotter.save(save_fig_path)
    
    shutil.rmtree(temporary_folder)
    return _exec_time,_total_lenght,angle_delta_mean


if __name__ == "__main__":
    LKH_PATH = "LKH.exe"
    SEED     = None
    DISTANCE = 5
    BORDERS_DISTANCE = 0
    CLUSTER_N = 16
    LKH_RUNS = 2
    MERGE_CLUSTERS  = False
    MERGE_TOLERANCE = 0.4 
    FILE = "assets/svg/rabbit.svg"

    rabbit = readPathSVG(FILE, scale=1)
    forma = [rabbit/2]

    # print(generateClustersPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/gcp/test.png"))
    # print(generateClustersMergedPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/merged/test.png"))
    # print(generatePathOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open/test.png"))
    # print(generatePathCHOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open_ch/test.png"))
    print(generateCHRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/ch_raw/test.png")) 
    print(generatePathRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/raw/test.2png"))
    exit()

    grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED)
    n_clusters = CLUSTER_N

    out_dir = "outputs/lkh_tests"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)
        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster)

    plotter = SlicesPlotter([None,forma],tile_direction='horizontal')
    plotter.set_random_usable_colors(CLUSTER_N)
    total_lenght = 0
    for idx, cluster in tqdm(enumerate(grid_clusters), total=len(grid_clusters), desc="Processing Clusters"):
        print(" ")
        start_node, end_node = 0,1
        print(cluster.cluster[start_node])    
        # start node aqui nao importa, pois se nao for 0 o end node buga 
        dummy_edge = (0,end_node) if end_node is not None else None
        if start_node is not None and end_node is not None:
            temp_ = np.array([0,0])
            temp_[0] = cluster.cluster[start_node][0]
            temp_[1] = cluster.cluster[start_node][1]
            cluster.cluster[start_node][0] = cluster.cluster[0][0] 
            cluster.cluster[start_node][1] = cluster.cluster[0][1]
            cluster.cluster[0][0] = temp_[0]
            cluster.cluster[0][1] = temp_[1]
            
        dst_mat = compute_distance_matrix_numba_parallel(cluster.cluster,cluster.cluster)
        mat_file_name = os.path.basename(FILE).replace(".svg",f"_c{idx}.txt")
        mat_file_name = os.path.join(out_dir, mat_file_name)
        if dummy_edge:
            dummy_tour = generateDummyTour(start_node,end_node,cluster.cluster.shape[0]+1)
            initial_tour_file = mat_file_name.replace(".txt", "_dummy_tour.txt")
            writeInitialTourFile(dummy_tour,initial_tour_file)
            writeDistanceMatrixProblemFile(dst_mat, mat_file_name,dummy_edge=dummy_edge)
        if dummy_edge:
            lenght,best_route = lkh.solve(LKH_PATH, initial_tour_file=initial_tour_file, problem_file=mat_file_name, runs=LKH_RUNS)
            best_route = np.array(best_route[0])[:-1] - 1
        else:
            lenght,best_route = lkh.solve(LKH_PATH, problem_file=mat_file_name, runs=LKH_RUNS)
            best_route = np.array(best_route[0]) - 1
            cluster.route = best_route
        total_lenght += lenght
        if not MERGE_CLUSTERS:
            cluster_cmap = pred[[cluster.remap_idxs[idx] for idx in cluster.route]]
            target_node = best_route[-1] if end_node is None else dummy_edge[1]
            plotter.draw_points([[cluster.cluster[best_route[0]],
                                cluster.cluster[best_route[-1]],
                                cluster.cluster[target_node]]],
                                colors_maps=[[0,1,2]],
                                markersize=4,edgesize=3)
            plotter.set_background_colors(['black','black'])
            plotter.draw_vectors([cluster.cluster],[best_route])
        if not MERGE_CLUSTERS:
            plotter.draw_fig_title(f"{total_lenght}")
            plotter.show()
            #plotter.save(os.path.join(out_dir,os.path.basename(FILE).replace(".svg",f"_{CLUSTER_N}_{DISTANCE}_clusters.png")))

        if MERGE_CLUSTERS:
            print("Merging Clusters In Single One")
            b_cluster = Cluster()
            b_cluster.cluster = grid
            routes_remaped = []
            for _cluster in grid_clusters:
                remap = [_cluster.remap_idxs[idx] for idx in _cluster.route]
                routes_remaped.extend(remap)

        dst_mat = compute_distance_matrix_numba_parallel(b_cluster.cluster,b_cluster.cluster)

        mat_file_name = os.path.basename(FILE).replace(".svg",f"_cMerged.txt")
        mat_file_name = os.path.join(out_dir, mat_file_name)
        writeDistanceMatrixProblemFile(dst_mat, mat_file_name)

        merge_tour_file = mat_file_name.replace(".txt", f"_mergetour_{idx+1}_.txt")
        writeInitialTourFile(routes_remaped, merge_tour_file)

        lenght,best_route = lkh.solve(LKH_PATH,runs=1,
                                    time_limit=20.0,
                                    optimum=(total_lenght*(1+MERGE_TOLERANCE)),
                                    problem_file=mat_file_name,
                                    merge_tour_files=[merge_tour_file])
        best_route = np.array(best_route[0]) - 1
        plotter.draw_points([b_cluster.cluster],colors_maps=[pred])
        plotter.draw_points([[cluster.cluster[best_route[0]],cluster.cluster[best_route[-1]]]],colors_maps=[[0,1]],
                            markersize=4,edgesize=3)
        plotter.set_background_colors(['black'])
        plotter.draw_vectors([b_cluster.cluster],[best_route])
        plotter.draw_fig_title(f"{lenght}")
        #plotter.show()
        plotter.save(os.path.join(out_dir,os.path.basename(FILE).replace(".svg",f"_{CLUSTER_N}_{DISTANCE}_cmerged.png")))
