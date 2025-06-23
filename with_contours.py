import numpy as np

from os.path import exists,basename,join
from sys import argv

from core.TSP.strategy import Strategy
from core.Layer import Layer
from core.visualize import SlicesPlotter
OUTPUT_DIR = "outputs/contours_test"

# Parâmetros globais para a solução
SOLVER = "lkh"
GENERATOR = "clusters"
END_TYPE = "open"
HEURISTIC = "continuous"

# Parâmetros globais para a solução
DISTANCE        = 4
SEED            = 627
RUNS            = 1
N_CLUSTERS      = 5  
Z_INDEX         = 10
BORDER_DISTANCE = DISTANCE 

strategy = Strategy(output_dir=OUTPUT_DIR,
                    distance=DISTANCE,
                    seed=SEED,
                    runs=RUNS,
                    n_clusters=N_CLUSTERS,
                    save_fig=True,
                    use_border_contours = True
                    )

if __name__ == "__main__":
    file_path = argv.pop(-1)
    if not exists(file_path):
        print(f"[ERROR] {file_path} not found")
    layer = Layer.From(file_path,z=Z_INDEX)
    grid,best_tour,metrics = strategy.solve(layer,
                                            tsp_solver=SOLVER,
                                            generator=GENERATOR,
                                            end_type=END_TYPE,
                                            initial_heuristic=HEURISTIC
                                            )