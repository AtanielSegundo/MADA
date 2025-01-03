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
