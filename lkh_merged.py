import os 
import numpy as np
import lkh
from typing import Tuple 
from core.geometry import nthgone, CLOCKWISE
from core.visualize._2d import SlicesPlotter
from commons.utils.clipper import readPathSVG
from kmeans_tests import generatePointsAndClusters
from tsptests import compute_distance_matrix_numba_parallel
from lkh_clusters import writeDistanceMatrixProblemFile,writeInitialTourFile,writeDistanceMatrixProblemFile,\
                         generateDummyTour,Cluster
from tqdm import tqdm

LKH_PATH = "LKH.exe"
SEED     = None
DISTANCE = 5
CLUSTER_N = 6
LKH_RUNS = 2
FILE = "assets/svg/rabbit.svg"

rabbit = readPathSVG(FILE, scale=1)
forma = [rabbit]
grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE, seed=SEED)
n_clusters = CLUSTER_N
grid_clusters = np.empty(n_clusters, dtype=object)

# Getting centers routes 
 
out_dir = "outputs/lkh_tests"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

grid_clusters = [Cluster() for _ in range(np.max(pred)+1)]
for jj, p in enumerate(pred):
    grid_clusters[p].cluster.append(grid[jj])
    grid_clusters[p].remap_idxs.append(jj)
    
for cluster in grid_clusters:
    cluster.cluster = np.array(cluster.cluster)

plotter = SlicesPlotter([forma])
plotter.set_random_usable_colors(CLUSTER_N)
total_lenght = 0
for idx, cluster in tqdm(enumerate(grid_clusters), total=len(grid_clusters), desc="Processing Clusters"):
    print(" ")
    start_node, end_node = 100,None
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
