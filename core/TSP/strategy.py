import os
import shutil
import time
import numpy as np
import lkh
from typing import Tuple,Type
from core.visualize._2d import SlicesPlotter
from core.Points.Grid import generateGridAndClusters,Cluster
from core.Points.operations import compute_angle_delta_mean
from core.TSP.solver import Solver,NoSolver
from core.Tour import generateCH,generateDummyTour,generateCH_with_dummy,Tour,openEnd,closeEnd,TourEnd
from core.Layer import Layer
from core.TSP.LKH.LKH import LKH

angle_delta = float
path_lenght = float
exec_time = float
number_of_points = int
Grid = np.ndarray

class Metrics:
    def __init__(self,execution_time,tour_lenght,angle_delta_mean,nodes_count=0):
        self.nodes_count      = nodes_count
        self.execution_time   = execution_time
        self.tour_lenght      = tour_lenght
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
                      angle_delta_mean,
                      grid.len)
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
    for cidx,cluster_idx in enumerate(endType_manager.clusters_centers_tour.path):
        last_point_idx = new_start_point_idx = None
        cluster = clusters.set[cluster_idx]
        if cidx > 0 : last_point_idx = open_route_merged[-1]
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
                      angle_delta_mean,
                      grid.len)
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
    end_exec_time = time.time()
    angle_delta_mean = compute_angle_delta_mean(grid.points,best_path)
    metrics = Metrics(end_exec_time-start_exec_time,
                      tour_lenght,
                      angle_delta_mean,
                      grid.len)
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
    "lkh" : LKH,
    "no_solver" : NoSolver
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
        self.tag = None

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
        tag = tag or f"{layer.tag}_{generator}_{end_type}_{initial_heuristic}.png"
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
                                colors_maps=[[0,self.n_cluster]],markersize=3,edgesize=1)
            _c_sqrt_2 = 1.4142135624
            wire_arc_distance_lim = self.distance * _c_sqrt_2 
            plotter.draw_vectors([grid.points],[best_tour.path],thick=1.25,
                                 d_lim=wire_arc_distance_lim)
        if self.save_fig:
            plotter.draw_fig_title(metrics.tour_lenght.__ceil__())
            plotter.save(save_fig_path)
        return grid,best_tour,metrics