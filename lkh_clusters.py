from core.Layer import Layer
from core.TSP.strategy import Strategy,generatePathOpenClusters
            
if __name__ == "__main__":
    SEED     = 13
    DISTANCE = 7
    BORDERS_DISTANCE = 0
    CLUSTER_N = 16
    LKH_RUNS = 2
    MERGE_CLUSTERS = False
    MERGE_TOLERANCE = 0.4
    FILE = "assets/svg/rabbit.svg"

    layer = Layer.From(FILE,scale=0.5)
    strategy = Strategy(save_fig=True,distance=5,seed=SEED)
    solver_str = "lkh"
    generator_str = "merged"
    end_type_str = "closed"
    initial_heuristic_str = "continuous"
    test_name = "test_"+generator_str+"_"+end_type_str+"_"+initial_heuristic_str
    strategy.solve(layer,
                 solver_str,
                 generator_str,
                 test_name,
                 end_type=end_type_str,
                 initial_heuristic=initial_heuristic_str)
    
    exit()
    print(generatePathOpenClusters(6,layer.data,DISTANCE,seed=SEED,save_fig_path=f"outputs/old_{test_name}.png"))
    
    print(generateClustersPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/gcp/test.png"))
    print(generateClustersMergedPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/merged/test.png"))
    print(generatePathCHOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open_ch/test.png"))
    print(generateCHRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/ch_raw/test.png")) 
    print(generatePathRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/raw/test.2png"))
    
    exit()
