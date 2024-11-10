import numpy as np
import os
from typing import *
from sklearn.cluster import KMeans
from core.geometry import nthgone, CLOCKWISE
from gridtests import fill_geometrys_with_points,is_polygon_counterclockwise
from commons.utils.clipper import readPathSVG, center_point
from ttftests import str2Polygons, generate_square_box_by_lenght,getPolygonsCenter
from core.transform import geometrys_from_txt_nan_separeted
from core.visualize import ShowGeometrys
from core.slicing import getSliceStl


def savePointsAndClusters(file_path: str, forma, pointgrid,
                          predictions, clusters_centers, bg_color="black"):
    if all([p is not None for p in [forma, predictions, clusters_centers]]):
        ShowGeometrys([[np.array([[0,0]])],forma],
                      points_grids=[pointgrid,
                                    pointgrid], background_color=bg_color,
                      points_grids_color_idx_map=[predictions],
                      points_grids_clusters_centers=[clusters_centers],
                      show_plot=False, file_name=file_path
                      )


def generatePointsAndClusters(forma: List[np.ndarray], clusters_n=6, seed=None,
                              clusters_iters=600, distance=5, fliped_y=False,figure_sep=None):
    figure_sep = figure_sep or 0.5
    pointgrid = fill_geometrys_with_points(forma, distance, figure_sep=figure_sep, fliped_y=fliped_y)
    print(f"{len(pointgrid)} points created inside geometry")
    if len(pointgrid) < clusters_n:
        print("Not enough points in pointgrid")
        return None, None, None
    k_means = KMeans(n_clusters=clusters_n, max_iter=clusters_iters,random_state=seed)
    k_means.fit(pointgrid)
    clusters_centers = k_means.cluster_centers_
    predictions = k_means.predict(pointgrid)
    return pointgrid, predictions, clusters_centers

if __name__ == "__main__":
    #rabbit = readPathSVG("assets/svg/rabbit.svg", scale=1)
    #hole = nthgone(1000, 50, center_p=center_point(rabbit), dir=CLOCKWISE)
    #forma = [rabbit, hole]
    #generatePointsAndClusters(forma, "outputs\\rabbit.png")
    
    PATH = "assets/txt/formas"
    DISTANCE = 5
    CLUSTER_N = 6
    OUTPUT_PATH = f"outputs/d_{DISTANCE}_cn_{CLUSTER_N}"
    try:
        os.makedirs(OUTPUT_PATH)
    except Exception as e:
        print(e)
    
    print("Testando em MADA")
    waam_p, offsetx, offsety = str2Polygons("WAAM\nMADA", "assets\\ttf\\arial.ttf",200, _return_offsets=True)
    square = generate_square_box_by_lenght(
        max(abs(offsetx), abs(offsety)), getPolygonsCenter(waam_p))
    forma = [square]
    forma.extend(waam_p)
    grid, pred, centers = generatePointsAndClusters(
        forma, distance=DISTANCE, clusters_n=CLUSTER_N)
    savePointsAndClusters(os.path.join(
        OUTPUT_PATH, "MADA.png"), [square], grid, pred, centers)
    print()
    
    
    print("Testando K-Means em peças no formato txt".upper())
    print()
    for arquivo in os.listdir(PATH):
        print(f"Lendo {arquivo}")
        forma = geometrys_from_txt_nan_separeted(os.path.join(PATH, arquivo))
        file_name = arquivo.replace(".txt", "_pg.png")
        grid, pred, centers = generatePointsAndClusters(
            forma, distance=DISTANCE, clusters_n=CLUSTER_N)
        savePointsAndClusters(os.path.join(
            OUTPUT_PATH, file_name), forma, grid, pred, centers)
        print()
    
    PATH = "assets/3d"
    print("Testando K-Means em slices dos arquivos stl".upper())
    print()
    for arquivo in os.listdir(PATH):
        print(f"Lendo {arquivo}")
        if "truss" not in arquivo:
            forma = getSliceStl(os.path.join(PATH, arquivo), z=1)
        else:
            forma = getSliceStl(os.path.join(PATH, arquivo), z=1, scaleFactor=0.25)
        file_name = arquivo.replace(".stl", "_pg.png")
        grid, pred, centers = generatePointsAndClusters(forma,  distance=DISTANCE, clusters_n=CLUSTER_N, fliped_y=True)
        savePointsAndClusters(os.path.join(OUTPUT_PATH, file_name), forma, grid, pred, centers)
        print()
    
