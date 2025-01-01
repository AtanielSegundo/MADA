from typing import List, Tuple
from sklearn.cluster import KMeans
from core.geometry import fill_geometrys_with_points
from core.Layer import Layer
from core.TSP.strategy import Strategy
import numpy as np

def generatePointsAndClusters(forma: List[np.ndarray], clusters_n=6, seed=None,
                              clusters_iters=600, distance=5, fliped_y=False, figure_sep=None):
    figure_sep = figure_sep or 0.5
    pointgrid = fill_geometrys_with_points(forma, distance, figure_sep=figure_sep, fliped_y=fliped_y)
    
    if len(pointgrid) < clusters_n:
        print("Not enough points in pointgrid")
        return None, None, None

    k_means = KMeans(
        n_clusters=clusters_n,
        max_iter=clusters_iters,
        random_state=seed,
        n_init='auto'
    )
    k_means.fit(pointgrid)
    clusters_centers = k_means.cluster_centers_
    predictions = k_means.labels_
    return pointgrid, predictions, clusters_centers

class Grid:
    def __init__(self, points: np.ndarray):
        assert points.shape[1] == 2, "Points must be a 2D array with shape (len, 2)"
        self.points: np.ndarray = points
        self.len = points.shape[0]

class Cluster:
    def __init__(self):
        self.cluster = None
        self.route = []
        self.remap_idxs = []

class Clusters:
    def __init__(self,set,centers):
        self.set = set
        self.n_clusters = centers.shape[0]
        self.centers = None
        

def generateGridAndClusters(layer: Layer, strategy: Strategy, gen_clusters=True) -> Tuple[Grid, List[Cluster]]:
    forma = layer.data
    points, pred, centers = generatePointsAndClusters(
        forma,
        clusters_n=strategy.n_cluster,
        distance=strategy.distance,
        figure_sep=strategy.border_distance,
        seed=strategy.seed,
        fliped_y=layer.is_y_flipped
    )
    
    if points is None or pred is None:
        return Grid(np.empty((0, 2))), []
    
    grid_clusters = None
    if gen_clusters :
        grid_clusters = [Cluster() for _ in range(strategy.n_cluster)]
        for cluster_idx in range(strategy.n_cluster):
            mask = pred == cluster_idx
            cluster_points = points[mask]
            grid_clusters[cluster_idx].cluster = cluster_points
            grid_clusters[cluster_idx].remap_idxs = np.flatnonzero(mask).tolist()
            
    return Grid(points), Clusters(grid_clusters,centers)
