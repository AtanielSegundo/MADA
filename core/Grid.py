from typing import List
from sklearn.cluster import KMeans
from core.geometry import fill_geometrys_with_points
import numpy as np

def generatePointsAndClusters(forma: List[np.ndarray], clusters_n=6, seed=None,
                              clusters_iters=600, distance=5, fliped_y=False,figure_sep=None):
    figure_sep = figure_sep or 0.5
    pointgrid = fill_geometrys_with_points(forma, distance, figure_sep=figure_sep, fliped_y=fliped_y)
    if len(pointgrid) < clusters_n:
        print("Not enough points in pointgrid")
        return None, None, None
    k_means = KMeans(n_clusters=clusters_n, max_iter=clusters_iters,random_state=seed)
    k_means.fit(pointgrid)
    clusters_centers = k_means.cluster_centers_
    predictions = k_means.predict(pointgrid)
    return pointgrid, predictions, clusters_centers
