import os 
import time
import numpy as np
import lkh
import shutil
from typing import Tuple,List
from core.geometry import nthgone, CLOCKWISE
from core.visualize._2d import SlicesPlotter
from core.clipper import readPathSVG
from kmeans_tests import generatePointsAndClusters
from tsptests import compute_distance_matrix_numba_parallel,generateGreedyPath,sort_points_up_right,compute_angle_delta_mean
from lkh_clusters import writeDistanceMatrixProblemFile,writeInitialTourFile,writeDistanceMatrixProblemFile,\
                         generateDummyTour,Cluster
from core.transform import geometrys_from_txt_nan_separeted
from core.slicing import getSliceStl
from tqdm import tqdm


if __name__ == "__main__":
    LKH_PATH = "LKH.exe"
    SEED     = 777
    DISTANCE = 7
    CLUSTER_N = 8
    LKH_RUNS = 10
    #FILE = "assets/svg/rabbit.svg"
    FILE  = "assets/3d/flange16furos.stl"
    FLIPED_Y = False
    if (ext_kwrd:=".svg") in FILE:
        forma = [readPathSVG(FILE, scale=1)]
    elif (ext_kwrd:=".txt") in FILE:
        forma = [geometrys_from_txt_nan_separeted(FILE)]
    elif (ext_kwrd:=".stl") in FILE:
        forma = getSliceStl(FILE,z=1,scaleFactor=0.5)
        FLIPED_Y = True

    grid, pred, centers = generatePointsAndClusters(forma, clusters_n=CLUSTER_N, distance=DISTANCE, seed=SEED,figure_sep=0,fliped_y=FLIPED_Y)
    n_clusters = CLUSTER_N

    out_dir = "outputs/lkh_tests"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # esta organizado de 0 a n_clusters
    grid_clusters = [Cluster() for _ in range(n_clusters)]
    for jj, p in enumerate(pred):
        grid_clusters[p].cluster.append(grid[jj])
        grid_clusters[p].remap_idxs.append(jj)
        
    for cluster in grid_clusters:
        cluster.cluster = np.array(cluster.cluster)

    # Organizando os clusters de forma greedy
    #greedy_idxs = generateGreedyPath(centers)
    _remap,centers = sort_points_up_right(centers)
    centers_prb_f = os.path.join(out_dir,"centers_dst.txt")
    writeDistanceMatrixProblemFile(compute_distance_matrix_numba_parallel(centers,centers),centers_prb_f)
    _,centers_route = lkh.solve(LKH_PATH,problem_file=centers_prb_f,runs=5)
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
        clusters_start_ends[ii][1] = end_current_cluster
        clusters_start_ends.append([start_next_cluster,1])

    plotter = SlicesPlotter([forma,forma],tile_direction='horizontal')
    plotter.set_random_usable_colors(CLUSTER_N)
    total_lenght = 0

    print(clusters_start_ends)
    for idx, greedy_idx in tqdm(enumerate(greedy_idxs), total=len(greedy_idxs), desc="Processing Clusters"):
        print(" ")
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
        mat_file_name = os.path.basename(FILE).replace(ext_kwrd,f"_c{idx}.txt")
        mat_file_name = os.path.join(out_dir, mat_file_name)
        
        dummy_tour = generateDummyTour(start_node,end_node,cluster.cluster.shape[0]+1)
        initial_tour_file = mat_file_name.replace(".txt", "_dummy_tour.txt")
        writeInitialTourFile(dummy_tour,initial_tour_file)
        writeDistanceMatrixProblemFile(dst_mat, mat_file_name,dummy_edge=dummy_edge)
        lenght,best_route = lkh.solve(LKH_PATH, initial_tour_file=initial_tour_file, problem_file=mat_file_name, runs=LKH_RUNS)
        best_route = np.array(best_route[0])[:-1] - 1
        total_lenght += lenght
        cluster.route = best_route
        
        cluster_cmap = pred[[cluster.remap_idxs[idx] for idx in cluster.route]]
        target_node = best_route[-1] if end_node is None else dummy_edge[1]
        print(plotter.usable_colors)
        plotter.draw_points([[cluster.cluster[best_route[0]],
                                cluster.cluster[best_route[-1]],
                                cluster.cluster[target_node]]],
                                colors_maps=[[0,1,2]],
                                markersize=4,edgesize=3)
        plotter.set_background_colors(['black','black'])
        plotter.draw_vectors([cluster.cluster],[best_route])
    
    # Unindo as rotas em uma só
    open_route_merged = []
    points_crt = []
    for greedy_idx in greedy_idxs:
        cluster = grid_clusters[greedy_idx]
        for idx in cluster.route:
            open_route_merged.append(cluster.remap_idxs[idx])
    plotter.draw_points([None,None])
    plotter.draw_points([None,
                         [grid[open_route_merged[0]],grid[open_route_merged[-1]]]],
                         colors_maps=[None,[0,2]],markersize=4,edgesize=3)
    plotter.draw_vectors([None,grid],[None,open_route_merged],thick=2)
    plotter.show()