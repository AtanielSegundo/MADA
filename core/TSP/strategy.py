import os
import shutil
import time
import numpy as np
import lkh
from typing import Tuple,Type,List
from core.visualize._2d import SlicesPlotter
from core.geometry import fill_geometrys_with_points
from core.Points.Grid import generatePointsAndClusters,generateGridAndClusters,Cluster,Clusters
from core.Points.operations import compute_distance_matrix_numba_parallel,compute_angle_delta_mean,sort_points_up_right
from core.TSP.solver import Solver
from core.Tour import generateCH,generateDummyTour,generateCH_with_dummy,Tour,openEnd,closeEnd,TourEnd
from core.Layer import Layer
from core.TSP.LKH.LKH import LKH

angle_delta = float
path_lenght = float
exec_time = float
number_of_points = int
Grid = np.ndarray

class Metrics:
    def __init__(self,execution_time,tour_lenght,angle_delta_mean):
        self.execution_time = execution_time
        self.tour_lenght = tour_lenght
        self.angle_delta_mean = angle_delta_mean


def continuous_trajectory_heuristic(points:np.ndarray,start_end:Tuple[int, int]=None) -> np.ndarray:
    if start_end is None:
        return generateCH(points)
    else:
        return generateCH_with_dummy(points,*start_end)
        

def standard_trajectory_heuristic(points:np.ndarray,start_end:Tuple[int, int]=None) -> np.ndarray:
    if start_end is None:
        return None
    else:
        return generateDummyTour(*start_end,points.shape[0]+1)

# key:str,value:Callable
AVAILABLE_INITIAL_HEURISTICS = {
    "continuous" : continuous_trajectory_heuristic,
    "std": standard_trajectory_heuristic
}

def rawGenerator(tag:str,solver:Type[Solver],layer:Layer,strategy) -> Tuple[Grid,Tour,Metrics]:
    temporary_folder = os.path.join(strategy.temp_dir,f"{tag}/rawGenerator/")
    os.makedirs(temporary_folder,exist_ok=True)
    grid,clusters = generateGridAndClusters(layer,strategy,gen_clusters=False)
    endType_manager:TourEnd = strategy.end_type(temporary_folder)
    initial_heuristic_handler = strategy.initial_heuristic
    tsp_solver = solver(temporary_folder) 
    start_exec_time = time.time()
    tour_lenght = 0
    best_path  = None
    endType_manager.set_clusters_big_path(clusters, tsp_solver)
    while (endType_manager.current_start_end_idx is not None):
        endType_manager.get_current_start_end(clusters)
        suffix = f"_dm{(endType_manager.current_start_end_idx or 0)+1}"
        initial_tour_path = initial_heuristic_handler(grid.points)
        tour_lenght,best_tour = tsp_solver.solve(
                                grid.points,
                                strategy.runs,
                                strategy.seed,
                                initial_path=initial_tour_path,
                                start_end=endType_manager.start_end,
                                suffix=suffix
                                )
        endType_manager.remap_tour_path(best_tour)
        best_path = best_tour.path
    end_exec_time = time.time()
    angle_delta_mean = compute_angle_delta_mean(grid.points,best_path)
    metrics = Metrics(end_exec_time-start_exec_time,
                      tour_lenght,
                      angle_delta_mean)
    return grid,Tour(best_path),metrics


def clustersGenerator(tag:str,solver:Type[Solver],layer:Layer,strategy) -> Tuple[Grid,Tour]:
    temporary_folder = os.path.join(strategy.temp_dir,f"{tag}/clustersGenerator/")
    os.makedirs(temporary_folder,exist_ok=True)
    grid,clusters = generateGridAndClusters(layer,strategy)
    endType_manager:TourEnd = strategy.end_type(temporary_folder)
    initial_heuristic_handler = strategy.initial_heuristic
    tsp_solver = solver(temporary_folder) 
    start_exec_time = time.time()
    total_tour_lenght = 0
    endType_manager.set_clusters_big_path(clusters, solver)
    while (endType_manager.current_start_end_idx is not None):
        current_cluster_idx,start_end_idx = endType_manager.get_current_start_end(clusters)
        suffix = f"_dm{(start_end_idx or 0)+1}"
        cluster:Cluster = clusters.set[current_cluster_idx]
        initial_tour_path = initial_heuristic_handler(cluster.cluster,endType_manager.start_end)
        cluster_tour_lenght,cluster_best_tour = tsp_solver.solve(
                                cluster.cluster,
                                strategy.runs,
                                strategy.seed,
                                initial_path=initial_tour_path,
                                start_end=endType_manager.start_end,
                                suffix=suffix
                                )
        endType_manager.remap_tour_path(cluster_best_tour)
        best_path = cluster_best_tour.path
        cluster.route = best_path
        total_tour_lenght += cluster_tour_lenght
    
    open_route_merged = []
    for cluster_idx in endType_manager.clusters_centers_tour.path:
        last_point_idx = new_start_point_idx = None
        cluster = clusters.set[cluster_idx]
        if cluster_idx > 0 : last_point_idx = open_route_merged[-1]
        for idx in cluster.route:
            if idx == 0 : new_start_point_idx = cluster.remap_idxs[idx]
            open_route_merged.append(cluster.remap_idxs[idx])
        if (last_point_idx is not None) and (new_start_point_idx is not None): 
            last_point = grid.points[last_point_idx]
            new_start_point = grid.points[new_start_point_idx]
            total_tour_lenght += np.linalg.norm(np.array(last_point)-np.array(new_start_point))
        

    end_exec_time = time.time()
    angle_delta_mean = compute_angle_delta_mean(grid.points,open_route_merged)
    metrics = Metrics(end_exec_time-start_exec_time,
                      total_tour_lenght,
                      angle_delta_mean)
    return grid,Tour(np.array(open_route_merged)),metrics

def mergedGenerator(tag:str,solver:Type[Solver],layer:Layer,strategy) -> Tuple[Grid,Tour]:
    temporary_folder = os.path.join(strategy.temp_dir,f"{tag}/mergedGenerator/")
    os.makedirs(temporary_folder,exist_ok=True)
    grid,clusters = generateGridAndClusters(layer,strategy)
    endType_manager:TourEnd = strategy.end_type(temporary_folder)
    initial_heuristic_handler = strategy.initial_heuristic
    tsp_solver = solver(temporary_folder) 
    start_exec_time = time.time()
    
    endType_manager.set_clusters_big_path(clusters, solver)
    while (endType_manager.current_start_end_idx is not None):
        current_cluster_idx,start_end_idx = endType_manager.get_current_start_end(clusters)
        suffix = f"_dm{(start_end_idx or 0)+1}"
        cluster:Cluster = clusters.set[current_cluster_idx]
        initial_tour_path = initial_heuristic_handler(cluster.cluster,endType_manager.start_end)
        _,cluster_best_tour = tsp_solver.solve(
                                cluster.cluster,
                                strategy.runs,
                                strategy.seed,
                                initial_path=initial_tour_path,
                                start_end=endType_manager.start_end,
                                suffix=suffix
                                )
        endType_manager.remap_tour_path(cluster_best_tour)
        best_path = cluster_best_tour.path
        cluster.route = best_path
    
    open_route_merged = []
    for cluster_idx in endType_manager.clusters_centers_tour.path:
        cluster = clusters.set[cluster_idx]
        for idx in cluster.route:
            open_route_merged.append(cluster.remap_idxs[idx])
    
    suffix = f"_merged_tour"
    #CONVERTING OPEN ROUTE MERGED TO MATCH START END CRITERIA
    converted_open_route = open_route_merged[:-1] + [len(open_route_merged)] + [open_route_merged[-1]]
    #print(len(open_route_merged),len(converted_open_route))
    start_node, end_node = open_route_merged[0], open_route_merged[-1]
    if start_node == end_node :
        start_node = end_node-1 if end_node-1 > 0 else end_node+1
    if start_node is not None and end_node is not None:
        temp_ = grid.points[0]
        grid.points[0] = grid.points[start_node] 
        grid.points[start_node] = temp_
        temp_ = np.array([0,0])
        temp_[0] = grid.points[start_node][0]
        temp_[1] = grid.points[start_node][1]
        grid.points[start_node][0] = grid.points[0][0] 
        grid.points[start_node][1] = grid.points[0][1]
        grid.points[0][0] = temp_[0]
        grid.points[0][1] = temp_[1]
    
    start_end= [0,end_node]
    if end_node == 0:
        start_end = [0,start_node]

    tour_lenght,best_merged_tour = tsp_solver.solve(
                                    grid.points,
                                    strategy.runs,
                                    strategy.seed,
                                    initial_path=converted_open_route,
                                    start_end=start_end,
                                    suffix=suffix
                                    )
    best_path = [e for e in best_merged_tour.path if e!=len(open_route_merged)]
    #print(best_path)
    end_exec_time = time.time()
    angle_delta_mean = compute_angle_delta_mean(grid.points,best_path)
    metrics = Metrics(end_exec_time-start_exec_time,
                      tour_lenght,
                      angle_delta_mean)
    return grid,Tour(np.array(best_path)),metrics


AVAILABLE_GENERATORS = {
    "raw" : rawGenerator,
    "clusters" : clustersGenerator,
    "merged" : mergedGenerator
}


# key:str, value: 
AVAILABLE_END_TYPES = {
    "open"   : openEnd ,
    "closed" : closeEnd
}

AVAILABLE_SOLVERS = {
    "lkh" : LKH
}

class Strategy:
    def __init__(self, output_dir="./outputs", n_clusters=6, distance=7,
                 border_distance=0, seed=None, save_fig=False, runs: int = 5):
        self.n_cluster = n_clusters
        self.distance = distance
        self.generator = None
        self.end_type = None
        self.initial_heuristic = None
        self.seed = seed or int(time.time())
        self.save_fig = save_fig
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, "temp/")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.runs: int = runs
        self.border_distance = border_distance

    def clearTemporary(self):
        shutil.rmtree(self.temp_dir)

    def setHandlers(self, generator: str, end_type: str, initial_heuristic: str):
        self.generator = AVAILABLE_GENERATORS.get(generator)
        self.end_type:TourEnd = AVAILABLE_END_TYPES.get(end_type)
        self.initial_heuristic = AVAILABLE_INITIAL_HEURISTICS.get(initial_heuristic)

        invalid_handlers = {
            "generator": generator if self.generator is None else None,
            "end_type": end_type if self.end_type is None else None,
            "initial_heuristic": initial_heuristic if self.initial_heuristic is None else None
        }
        invalid_names = [name for name, value in invalid_handlers.items() if value is not None]

        if invalid_names:
            raise ValueError(f"The following handlers are invalid: {', '.join(invalid_names)}")

    def solve(self, layer: Layer, tsp_solver: str, generator: str,
              tag: str = None, end_type: str = "closed", initial_heuristic: str = "std"):
        tag = tag or f"{layer.tag}_{generator}_{end_type}_{initial_heuristic}"
        self.setHandlers(generator, end_type, initial_heuristic)
        solver_str = tsp_solver
        tsp_solver:Type[Solver] = AVAILABLE_SOLVERS.get(solver_str,None)
        if tsp_solver is None:
            raise f"{solver_str} not available"
        if self.save_fig:
            save_fig_path = os.path.join(self.output_dir, tag)
            plotter = SlicesPlotter([None], tile_direction='horizontal')
            plotter.set_random_usable_colors(self.n_cluster)
        grid, best_tour, metrics = self.generator(tag, tsp_solver, layer, self)
        if self.save_fig:
            plotter.set_background_colors(['black'])
            start_point = grid.points[best_tour.path[0]]
            end_point = grid.points[best_tour.path[-1]]
            plotter.draw_points([[start_point,end_point]],
                                colors_maps=[[1,2]],markersize=3,edgesize=1)
            plotter.draw_vectors([grid.points],[best_tour.path],thick=1.25)
        if self.save_fig:
            plotter.draw_fig_title(metrics.tour_lenght.__ceil__())
            plotter.save(save_fig_path)
        return grid,best_tour,metrics
    
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
    total_lenght,best_route = lkh.solve(solver=LKH_PATH, initial_tour_file=_initial_tour_file,
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
    SEED     = seed
    DISTANCE = distance
    BORDERS_DISTANCE = 0
    CLUSTER_N = n_clusters
    LKH_RUNS = runs
    LKH_PATH = "core/TSP/LKH/LKH.exe"
    from core.TSP.LKH.LKH import writeDistanceMatrixProblemFile,writeInitialTourFile
    _total_lenght = 0
    _exec_time = 0
    temporary_folder = "outputs/temp/open_clusters_path/"
    grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED,fliped_y=fliped_y)
    grid_clusters = [Cluster([]) for _ in range(n_clusters)]
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
    
    print(clusters_start_ends)
    print(greedy_idxs)
    for idx, greedy_idx in enumerate(greedy_idxs):
        _ta = time.time()
        cluster = grid_clusters[greedy_idx]
        start_node, end_node = clusters_start_ends[idx]
        print(greedy_idx," | ",start_node,end_node)
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
