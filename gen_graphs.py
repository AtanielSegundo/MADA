from lkh_clusters import generateClustersMergedPath,generateClustersPath,generatePathRaw,generatePathOpenClusters
from core.transform import geometrys_from_txt_nan_separeted
from commons.utils.clipper import readPathSVG
from core.slicing import getSliceStl
from core.visualize import ShowGeometrys
import matplotlib.pyplot as plt

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
    DISTANCE = 7
    LKH_RUNS = 2
    target_clusters_n = [3,4,6,8,12,16]

    pwd_generators = [("Clusters Unidos TSP",generateClustersMergedPath),
                  ("Cluster Separados",generateClustersPath),
                  ("Clusters Caminho Aberto",generatePathOpenClusters)]
    
    non_fliped_formas = [geometrys_from_txt_nan_separeted("assets/txt/formas/circulo_furo.txt"),[readPathSVG("assets/svg/rabbit.svg")]]


    fliped_formas = [getSliceStl("assets/3d/flange16furos.stl",z=1,scaleFactor=0.75),
                  getSliceStl("assets/3d/Flange_inventor.stl",z=2),
                  getSliceStl("assets/3d/Petro_foice.stl",z=2)]

    for forma in non_fliped_formas+fliped_formas:
        ShowGeometrys([forma],spliter=1)

    exit()

    for forma_idx,forma in enumerate(fliped_formas):
        raw_exec_time, raw_total_lenght, raw_angle_delta_mean = generatePathRaw(2, forma, DISTANCE, seed=SEED, runs=LKH_RUNS,fliped_y=True)
        outs_results = {n_cluster: {pwd_tag: [0, 0, 0] for pwd_tag, _ in pwd_generators} for n_cluster in target_clusters_n}
        outs_results["raw"] = {"Raw Path": [raw_exec_time, raw_total_lenght, raw_angle_delta_mean]}
        for n_cluster in target_clusters_n:
            for pwd_tag, pwd_fn in pwd_generators:
                _exec_time, _total_lenght, _angle_delta_mean = pwd_fn(
                n_cluster, forma, DISTANCE, seed=SEED, runs=LKH_RUNS, fliped_y=True
                )
                outs_results[n_cluster][pwd_tag][0] = _exec_time
                outs_results[n_cluster][pwd_tag][1] = _total_lenght
                outs_results[n_cluster][pwd_tag][2] = _angle_delta_mean
        forma_name = f"Forma_fp_{forma_idx + 1}"  
        plot_metric(outs_results, metric_index=0, metric_name="Execution_Time", forma_name=forma_name)
        plot_metric(outs_results, metric_index=1, metric_name="Total_Length", forma_name=forma_name)
        plot_metric(outs_results, metric_index=2, metric_name="Angle_Delta_Mean", forma_name=forma_name)


    for forma_idx,forma in enumerate(non_fliped_formas):
        raw_exec_time, raw_total_lenght, raw_angle_delta_mean = generatePathRaw(2, forma, DISTANCE, seed=SEED, runs=LKH_RUNS)
        outs_results = {n_cluster: {pwd_tag: [0, 0, 0] for pwd_tag, _ in pwd_generators} for n_cluster in target_clusters_n}
        outs_results["raw"] = {"Raw Path": [raw_exec_time, raw_total_lenght, raw_angle_delta_mean]}
        for n_cluster in target_clusters_n:
            for pwd_tag, pwd_fn in pwd_generators:
                _exec_time, _total_lenght, _angle_delta_mean = pwd_fn(
                n_cluster, forma, DISTANCE, seed=SEED, runs=LKH_RUNS
                )
                outs_results[n_cluster][pwd_tag][0] = _exec_time
                outs_results[n_cluster][pwd_tag][1] = _total_lenght
                outs_results[n_cluster][pwd_tag][2] = _angle_delta_mean
        forma_name = f"Forma_nf_{forma_idx + 1}"  
        plot_metric(outs_results, metric_index=0, metric_name="Execution_Time", forma_name=forma_name)
        plot_metric(outs_results, metric_index=1, metric_name="Total_Length", forma_name=forma_name)
        plot_metric(outs_results, metric_index=2, metric_name="Angle_Delta_Mean", forma_name=forma_name)