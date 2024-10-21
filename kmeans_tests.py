import numpy as np
import os
from typing import *
from sklearn.cluster import KMeans
from core.geometry import nthgone, CLOCKWISE
from gridtests import fill_geometrys_with_points
from commons.utils.clipper import readPathSVG, center_point
from core.transform import geometrys_from_txt_nan_separeted
from core.visualize import ShowGeometrys
from core.slicing import getSliceStl


def generatePointsAndClusters(forma: List[np.ndarray], file_path: str, clusters_n=6,
                              clusters_iters=600, distance=5, bg_color="black",fliped_y=False):
    pointgrid = fill_geometrys_with_points(forma, distance, fliped_y=fliped_y)
    if len(pointgrid) < clusters_n:
        print("Not enough points in pointgrid")
        return
    k_means = KMeans(n_clusters=clusters_n, max_iter=clusters_iters)
    k_means.fit(pointgrid)
    clusters_centers = k_means.cluster_centers_
    predictions = k_means.predict(pointgrid)
    ShowGeometrys([forma, forma],
                  points_grids=[pointgrid,
                                pointgrid], background_color=bg_color,
                  points_grids_color_idx_map=[predictions],
                  points_grids_clusters_centers=[clusters_centers],
                  show_plot=False, file_name=file_path
                  )


#rabbit = readPathSVG("assets/svg/rabbit.svg", scale=1)
#hole = nthgone(1000, 50, center_p=center_point(rabbit), dir=CLOCKWISE)
#forma = [rabbit, hole]
#generatePointsAndClusters(forma, "outputs\\rabbit.png")

PATH = "assets/txt/formas"

print("Testando K-Means em peças no formato txt".upper())
print()
for arquivo in os.listdir(PATH):
    print(f"Lendo {arquivo}")
    forma = geometrys_from_txt_nan_separeted(os.path.join(PATH, arquivo))
    file_name = arquivo.replace(".txt", "_pg.png")
    generatePointsAndClusters(forma, os.path.join("outputs", file_name))
    print()

PATH = "assets/3d"
print("Testando K-Means em slices dos arquivos stl".upper())
print()
for arquivo in os.listdir(PATH):
    print(f"Lendo {arquivo}")
    if "truss" not in arquivo:
        forma = getSliceStl(os.path.join(PATH, arquivo), z=1)
    else :
        forma = getSliceStl(os.path.join(PATH, arquivo), z=1,scaleFactor=0.25)
    file_name = arquivo.replace(".stl", "_pg.png")
    generatePointsAndClusters(forma, os.path.join("outputs", file_name), distance=5, fliped_y=True)
