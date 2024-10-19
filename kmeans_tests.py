import numpy as np
from sklearn.cluster import KMeans
from core.geometry import nthgone,CLOCKWISE
from gridtests import fill_geometrys_with_points
from commons.utils.clipper import offsetSVG, offsetTXT, readPathSVG,center_point
from core.visualize import ShowGeometrys

CLUSTERS_N = 5
DISTANCE = 3
ITER = 2

rabbit = readPathSVG("assets/svg/rabbit.svg", scale=1)
hole   = nthgone(100,50,center_p=center_point(rabbit),dir=CLOCKWISE) 

pointgrid = fill_geometrys_with_points([rabbit,hole],5)

k_means = KMeans(n_clusters=CLUSTERS_N,max_iter=600)
k_means.fit(pointgrid)
predictions = k_means.predict(pointgrid)
ShowGeometrys([[rabbit,hole],[rabbit,hole]],
              points_grids=[pointgrid,pointgrid],background_color="black",
              points_grids_color_idx_map=[predictions,None])
