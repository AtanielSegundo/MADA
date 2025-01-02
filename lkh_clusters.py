from core.Layer import Layer
from core.TSP.strategy import Strategy
            
if __name__ == "__main__":
    SEED     = None
    DISTANCE = 10
    BORDERS_DISTANCE = 0
    CLUSTER_N = 16
    LKH_RUNS = 2
    MERGE_CLUSTERS = False
    MERGE_TOLERANCE = 0.4
    FILE = "assets/svg/rabbit.svg"

    layer = Layer.From(FILE,scale=0.5)
    strategy = Strategy(save_fig=True)
    strategy.solve(layer,
                   "lkh",
                   "clusters",
                   "testclosedclusters",
                   end_type="open",
                   initial_heuristic="std")
    input("BRK")
    
    print(generateClustersPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/gcp/test.png"))
    print(generateClustersMergedPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/merged/test.png"))
    print(generatePathOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open/test.png"))
    print(generatePathCHOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open_ch/test.png"))
    print(generateCHRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/ch_raw/test.png")) 
    print(generatePathRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/raw/test.2png"))
    exit()
