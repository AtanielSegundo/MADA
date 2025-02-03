from core.TSP.strategy import Strategy,AVAILABLE_INITIAL_HEURISTICS
from core.visualize import SlicesPlotter
from core.Points.Grid import generateGridAndClusters
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.Layer import Layer
import os

def plot_metric(out_dir,outs_results, metric_index, metric_name, forma_name):
    plt.figure(figsize=(10, 6))
    for pwd_tag in outs_results[next(iter(outs_results))]:
        x = []
        y = []
        for n_cluster in outs_results:
            x.append(n_cluster)
            y.append(outs_results[n_cluster][pwd_tag][metric_index])
        plt.plot(x, y, label=pwd_tag, marker='o')
    plt.title(f"{metric_name} vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel(metric_name)
    plt.legend(loc="upper left")
    plt.savefig(out_dir+f"/{forma_name}_{metric_name}.png")

if __name__ == "__main__":
    SEED     = 3676
    DISTANCE = 4
    LKH_RUNS = 2
    Z_LAYER  = 1

    SOLVER    = "lkh"
    GENERATOR = "clusters"
    END_TYPE  = "open"
    
    strategy_ctx = Strategy(distance=DISTANCE,seed=SEED,save_fig=True,runs=LKH_RUNS)
    
    target_clusters_n = [1,2,4,6,8,12]

    # pwd_generators = [
    #               ("Clusters Unidos TSP", generateClustersMergedPath),
    #               ("Raw com CH", generateCHRaw),
    #               ("Cluster Separados", generateClustersPath),
    #               ("Clusters Caminho Aberto", generatePathOpenClusters),
    #               ("Clusters Caminho Aberto CH", generatePathCHOpenClusters)
    #               ]
    f_names = ["individual/Part_Wang_transformed.stl",
               "individual/Cframe.stl",
               "individual/flange16furos.stl",
               "individual/gripperroboticmaior.stl",
               "assets/txt/formas/rabbit.txt"
               ]
    
    formas = [Layer.From("individual/Part_Wang_transformed.stl"),
              Layer.From("individual/Cframe.stl"),
              Layer.From("individual/flange16furos.stl",scale=0.75),
              Layer.From("individual/gripperroboticmaior.stl"),
              Layer.From("assets/txt/formas/rabbit.txt"),
              ]

    for forma_idx,forma in enumerate(formas):
        forma_name = os.path.basename(f_names[forma_idx]).split(".")[0]
        forma_out_dir = f"./outputs/{forma_name}"
        os.makedirs(forma_out_dir,exist_ok=True)
        strategy_ctx.output_dir = forma_out_dir 
        grid,_ = generateGridAndClusters(forma,strategy_ctx,False)
        
        _img_plt = SlicesPlotter([None])
        _img_plt.draw_points([grid.points])
        _img_plt.draw_fig_title(f"Points:{grid.len}")
        _img_plt.save(forma_out_dir+"/grid.png")
        
        outs_results = {n_cluster: {h:[0, 0, 0] for h in AVAILABLE_INITIAL_HEURISTICS}  
                        for n_cluster in target_clusters_n}
        for n_cluster in tqdm(target_clusters_n, desc=f"Processing clusters for {forma_name}"):
            strategy_ctx.n_cluster = n_cluster
            for heuristic in AVAILABLE_INITIAL_HEURISTICS:
                _,_,metrics = strategy_ctx.solve(forma,SOLVER,GENERATOR,f"_{n_cluster}_{heuristic}",
                                                 END_TYPE,heuristic)
                outs_results[n_cluster][heuristic][0] = metrics.execution_time
                outs_results[n_cluster][heuristic][1] = metrics.tour_lenght
                outs_results[n_cluster][heuristic][2] = metrics.angle_delta_mean
        
        plot_metric(forma_out_dir,outs_results, metric_index=0, metric_name="Execution_Time", forma_name=forma_name)
        plot_metric(forma_out_dir,outs_results, metric_index=1, metric_name="Total_Length", forma_name=forma_name)
        plot_metric(forma_out_dir,outs_results, metric_index=2, metric_name="Angle_Delta_Mean", forma_name=forma_name)
