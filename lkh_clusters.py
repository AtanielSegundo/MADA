import os 
import numpy as np
import lkh
from typing import Tuple 
from core.geometry import nthgone, CLOCKWISE
from core.visualize._2d import SlicesPlotter
from commons.utils.clipper import readPathSVG
from kmeans_tests import generatePointsAndClusters
from tsptests import compute_distance_matrix_numba_parallel

def writeDistanceMatrixProblemFile(distance_matrix: np.ndarray, file_name: str, dummy_edge: Tuple[int, int] = None):
    with open(file_name, "w") as f:
        dimension = distance_matrix.shape[0] + (1 if dummy_edge else 0)
        f.write("NAME: DistanceMatrixProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")

        start_node, end_node = dummy_edge if dummy_edge else (None, None)
        for i, row in enumerate(distance_matrix):
            formatted_row = [
                "1e-6" if j == dimension - 1 and (i == start_node or i == end_node) else f"{val:.3f}" 
                for j, val in enumerate(np.append(row, [1e6] if dummy_edge else []))
            ]
            f.write(" ".join(formatted_row) + "\n")
        if dummy_edge:
            dummy_row = ["1e-6" if j == start_node or j == end_node or j == dimension - 1 else "1e6" for j in range(dimension)]
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
    
    tour[1:-2] = [i for i in range(tour_len - 1) if i not in {start_node, end_node}]
    return tour

LKH_PATH = "LKH.exe"
SEED     = None
DISTANCE = 5
BORDERS_DISTANCE = 0
CLUSTER_N = 6
LKH_RUNS = 2
MERGE_CLUSTERS  = False
MERGE_TOLERANCE = 0.4 
FILE = "assets/svg/rabbit.svg"

rabbit = readPathSVG(FILE, scale=1)
forma = [rabbit]
grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE,figure_sep=BORDERS_DISTANCE,seed=SEED)
n_clusters = CLUSTER_N
grid_clusters = np.empty(n_clusters, dtype=object)

out_dir = "outputs/lkh_tests"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

class Cluster:
    def __init__(self):
        self.cluster = []
        self.route = []
        self.remap_idxs = []

grid_clusters = [Cluster() for _ in range(np.max(pred)+1)]
for jj, p in enumerate(pred):
    grid_clusters[p].cluster.append(grid[jj])
    grid_clusters[p].remap_idxs.append(jj)
    
for cluster in grid_clusters:
    cluster.cluster = np.array(cluster.cluster)

plotter = SlicesPlotter([None,forma],tile_direction='horizontal')
plotter.set_random_usable_colors(CLUSTER_N)
from tqdm import tqdm
total_lenght = 0
for idx, cluster in tqdm(enumerate(grid_clusters), total=len(grid_clusters), desc="Processing Clusters"):
    print(" ")
    start_node, end_node = 20,100
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
    total_lenght += lenght
    cluster.route = best_route
    print(cluster.cluster[best_route[0]])
    print(best_route)
    input()
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
