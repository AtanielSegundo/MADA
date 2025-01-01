import os
import shutil
import time
from numba import njit
import numpy as np
import lkh
from typing import Tuple,List
from tqdm import tqdm
from core.visualize._2d import SlicesPlotter
from core.clipper import readPathSVG
from core.Tour import generateCH

LKH_PATH = "core/TSP/LKH/LKH.exe"
            
if __name__ == "__main__":
    SEED     = None
    DISTANCE = 10
    BORDERS_DISTANCE = 0
    CLUSTER_N = 16
    LKH_RUNS = 2
    MERGE_CLUSTERS = False
    MERGE_TOLERANCE = 0.4
    FILE = "assets/svg/rabbit.svg"

    rabbit = readPathSVG(FILE, scale=1)
    forma = [rabbit/2]

    print(generateClustersPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/gcp/test.png"))
    print(generateClustersMergedPath(6,forma,DISTANCE,seed=45,save_fig_path="outputs/merged/test.png"))
    print(generatePathOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open/test.png"))
    print(generatePathCHOpenClusters(6,forma,DISTANCE,seed=45,save_fig_path="outputs/open_ch/test.png"))
    print(generateCHRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/ch_raw/test.png")) 
    print(generatePathRaw(6,forma,DISTANCE,seed=45,save_fig_path="outputs/raw/test.2png"))
    exit()
