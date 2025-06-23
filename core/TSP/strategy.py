import os
import shutil
import time
import numpy as np
import lkh
from typing import Tuple,Type
from core.visualize._2d import SlicesPlotter
from core.Points.Grid import generateGridAndClusters,Cluster
from core.Points.Grid import Grid as GridObj
from core.Points.operations import compute_angle_delta_mean
from core.TSP.solver import Solver,NoSolver
from core.Tour import generateCH,generateDummyTour,generateCH_with_dummy,Tour,openEnd,closeEnd,TourEnd
from core.Layer import Layer
from core.TSP.LKH.LKH import LKH
from core.Points.operations import compute_distance_matrix_numba_parallel

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


def pick_first_border_contour(layer_data, center, first_point):
    best_layer_idx = None
    best_point_idx = None
    best_d_fp      = np.inf
    best_r_cent    = -np.inf

    for k, contour in enumerate(layer_data):
        # 1) distances to first_point
        dists_fp = np.linalg.norm(contour - first_point, axis=1)
        i_min    = dists_fp.argmin()
        d_fp     = dists_fp[i_min]

        # 2) radii from center
        radii  = np.linalg.norm(contour - center, axis=1)
        r_cent = radii.max()

        # 3) pick by (smallest d_fp, tie-break: largest r_cent)
        if (d_fp < best_d_fp) or (d_fp == best_d_fp and r_cent > best_r_cent):
            best_layer_idx = k
            best_point_idx = i_min
            best_d_fp      = d_fp
            best_r_cent    = r_cent

    return best_layer_idx, best_point_idx, best_d_fp, best_r_cent

def interpolate_contour(contour, step):
    """Interpolate a closed contour to ensure max segment length <= step"""
    if len(contour) == 0:
        return contour
    poly = np.vstack([contour, contour[0]])
    new_poly = []
    for i in range(len(poly) - 1):
        p0 = poly[i]
        p1 = poly[i+1]
        new_poly.append(p0)
        delta = p1 - p0
        d = np.linalg.norm(delta)
        if d > step:
            num_segments = int(np.ceil(d / step))
            for j in range(1, num_segments):
                t = j / num_segments
                new_poly.append(p0 + t * delta)
    new_poly = np.array(new_poly)
    return new_poly

def compute_tour_length(points: np.ndarray) -> float:
    """
    Given an (N,2) array of sequential points, returns the sum of
    Euclidean distances between each consecutive pair.
    """
    diffs = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return seg_lengths.sum()

def merge_border_contours(distance: float,
                          layer: Layer,
                          grid: Grid,
                          best_tour: Tour,
                          metrics: Metrics):
    start_exec_time = time.time()
    new_grid = []
    total_pts = grid.len 
    center = np.mean(grid.points, axis=0)
    first_pt = grid.points[best_tour.path[0]]
    
    f_border_idx, f_border_pt_idx, _, _ = pick_first_border_contour(
        layer.data, center, first_pt)
    
    # Interpolate and rotate first border
    f_border_interp = interpolate_contour(layer.data[f_border_idx], distance)
    dists = np.linalg.norm(f_border_interp - first_pt, axis=1)
    f_i0 = dists.argmin()
    f_border_rotated = np.roll(f_border_interp, -f_i0, axis=0)
    new_grid.extend(f_border_rotated)  
    total_pts += len(f_border_interp)

    # 2. Map other borders to grid points
    grid_to_border_idxs = []
    borders_idxs = []
    for k, border in enumerate(layer.data):
        if k == f_border_idx:
            continue
        dm = compute_distance_matrix_numba_parallel(
            grid.points.astype(np.float32),
            border.astype(np.float32)
        )
        flat_idx = dm.argmin()
        i_grid, i_border = np.unravel_index(flat_idx, dm.shape)
        grid_to_border_idxs.append(i_grid)
        borders_idxs.append((k, i_border))
    
    g2b_map = {g: (b, p) for g, (b, p) in zip(grid_to_border_idxs, borders_idxs)}
    grid_hits = set(grid_to_border_idxs)
    grid_hits.add(best_tour.path[0]) 

    # 3. Build new path: grid points + interpolated borders
    for idx in best_tour.path:
        if idx in grid_hits:
            if idx == best_tour.path[0]:
                border_idx, border_pt_idx = f_border_idx, f_border_pt_idx
            else:
                border_idx, border_pt_idx = g2b_map[idx]
            attach_pt = layer.data[border_idx][border_pt_idx]
            border_interp = interpolate_contour(layer.data[border_idx], distance)
            dists = np.linalg.norm(border_interp - attach_pt, axis=1)
            i0 = dists.argmin()
            border_rotated = np.roll(border_interp, -i0, axis=0)
            new_grid.extend(border_rotated)
            total_pts += len(border_interp)
        new_grid.append(grid.points[idx])
    
    # 4. Prepare output
    pts = np.array(new_grid, dtype=np.float64).reshape(-1, 2)
    new_best_tour = Tour(np.arange(total_pts))
    new_grid = GridObj(pts)
    angle_delta_mean = compute_angle_delta_mean(new_grid.points,new_best_tour.path)
    end_exec_time = time.time()
    new_metrics = Metrics(
        nodes_count=total_pts,
        execution_time=metrics.execution_time + (end_exec_time - start_exec_time),
        tour_lenght=compute_tour_length(new_grid.points),
        angle_delta_mean=angle_delta_mean
    )
    return new_grid, new_best_tour, new_metrics

AVAILABLE_GENERATORS = {
    "clusters" : clustersGenerator,
    "raw" : rawGenerator,
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
    def __init__(self, output_dir="./outputs", 
                 n_clusters=6, distance=7,
                 border_distance=0, seed=None, 
                 save_fig=False, runs: int = 5,
                 use_border_contours=False):
        self.n_cluster = n_clusters
        self.distance = distance
        self.generator = None
        self.end_type = None
        self.use_border_contours = use_border_contours
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
        
        if self.use_border_contours:
            grid, best_tour, metrics = merge_border_contours(self.distance,layer,grid, best_tour, metrics)
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