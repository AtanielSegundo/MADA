from lkh_clusters import generateClustersMergedPath,generateClustersPath,generatePathRaw,generatePathOpenClusters,generateCHRaw,generatePathCHOpenClusters
from core.visualize import ShowGeometrys,SlicesPlotter
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.Layer import Layer

def plot_metric(outs_results, metric_index, metric_name, forma_name):
    plt.figure(figsize=(10, 6))
    for pwd_tag in outs_results[next(iter(outs_results))]:
        x = []
        y = []
        for n_cluster in outs_results:
            if n_cluster == "raw":
                continue
            x.append(n_cluster)
            y.append(outs_results[n_cluster][pwd_tag][metric_index])
        plt.plot(x, y, label=pwd_tag, marker='o')
    raw_value = outs_results["raw"]["Raw Path"][metric_index]
    plt.axhline(y=raw_value, color='r', linestyle='--', label="Raw Path")
    plt.title(f"{metric_name} vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel(metric_name)
    plt.legend(loc="upper left")
    plt.savefig(f"outputs/metricas_plot/{forma_name}_{metric_name}.png")

if __name__ == "__main__":
    SEED = 777
    DISTANCE = 10
    LKH_RUNS = 2
    target_clusters_n = [3,4,6,8,12,16]

    pwd_generators = [
                  ("Clusters Unidos TSP", generateClustersMergedPath),
                  ("Raw com CH", generateCHRaw),
                  ("Cluster Separados", generateClustersPath),
                  ("Clusters Caminho Aberto", generatePathOpenClusters),
                  ("Clusters Caminho Aberto CH", generatePathCHOpenClusters)
                  ]

    formas = [Layer.From("assets/txt/formas/circulo_furo.txt"),
              Layer.From("assets/svg/rabbit.svg"),
              Layer.From("assets/3d/flange16furos.stl",z=1,scaleFactor=0.75),
              Layer.From("assets/3d/Flange_inventor.stl",z=2),
              Layer.From("assets/3d/Petro_foice.stl",z=2)]

    for forma_idx,forma in enumerate(formas):
        forma_name = f"Forma_{forma_idx + 1}"
        raw_exec_time, raw_total_lenght, raw_angle_delta_mean, number_of_points, grid= generatePathRaw(2, forma.data, DISTANCE, seed=SEED, runs=LKH_RUNS,fliped_y=forma.is_y_flipped)
        _img_plt = SlicesPlotter([None])
        _img_plt.draw_points([grid])
        _img_plt.draw_fig_title(f"Points:{number_of_points}")
        _img_plt.save(f"./outputs/metricas_plot/Forma_{forma_idx + 1}.png")
        outs_results = {n_cluster: {pwd_tag: [0, 0, 0] for pwd_tag, _ in pwd_generators} for n_cluster in target_clusters_n}
        outs_results["raw"] = {"Raw Path": [raw_exec_time, raw_total_lenght, raw_angle_delta_mean]}
        for n_cluster in tqdm(target_clusters_n, desc=f"Processing clusters for {forma_name}"):
            for pwd_tag, pwd_fn in pwd_generators:
                _exec_time, _total_lenght, _angle_delta_mean = pwd_fn(
                n_cluster, forma.data, DISTANCE, seed=SEED, runs=LKH_RUNS, fliped_y=forma.is_y_flipped
                )
                outs_results[n_cluster][pwd_tag][0] = _exec_time
                outs_results[n_cluster][pwd_tag][1] = _total_lenght
                outs_results[n_cluster][pwd_tag][2] = _angle_delta_mean
        plot_metric(outs_results, metric_index=0, metric_name="Execution_Time", forma_name=forma_name)
        plot_metric(outs_results, metric_index=1, metric_name="Total_Length", forma_name=forma_name)
        plot_metric(outs_results, metric_index=2, metric_name="Angle_Delta_Mean", forma_name=forma_name)
