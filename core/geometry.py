import numpy as np
from numba import njit,prange
from typing import *
from matplotlib import pyplot as plt

ANTICLOCKWISE  = 1.0
EXTERNAL       = -1.0 
CLOCKWISE      = -1.0
INTERNAL       = 1.0

def nthgone(n:int,ray:float,center_p=(0.,0.),pad=1/2,dir=ANTICLOCKWISE,closed=True) :
    '''
    **nthgone Function Documentation**
    =====================================

    ### Overview

    The `nthgone` function generates an n-sided polygon with a specified center point, radius, and direction.

    ### Parameters

    * `n`: The number of sides of the polygon. (int)
    * `ray`: The radius of the polygon. (float)
    * `center_p`: The coordinates of the center point of the polygon. Default is (0.0, 0.0). (tuple of floats)
    * `pad`: A padding factor to adjust the starting point of the polygon. Default is 1/2. (float)
    * `dir`: The direction of the polygon (1 for clockwise, -1 for counter-clockwise). Default is 1. (int)
    * `closed`: Repeat the first element in the end to close the geometry (int)
    ### Returns

    * A 2D numpy array representing the vertices of the polygon.

    ### Description

    The function calculates the angle `phi` between each vertex of the polygon using the formula `dir * (2 * np.pi) / n`. The `dir` parameter determines the direction of the polygon, with 1 resulting in a clockwise polygon and -1 resulting in a counter-clockwise polygon. It then generates the vertices of the polygon using a list comprehension, applying the parametric equation of a circle (`x = r * cos(theta)`, `y = r * sin(theta)`) with the calculated angle and radius. The `pad` parameter is used to adjust the starting point of the polygon. Finally, the function appends the first vertex to the end of the array to close the polygon and returns the result as a numpy array.

    ### Example Usage
    '''
    phi = dir*((2*np.pi)/n)
    arr = [[center_p[0]+ray*np.cos(phi*(i+pad)),center_p[1]+ray*np.sin(phi*(i+pad))] for i in range(n)]
    if closed : arr.append(arr[0])
    return np.array(arr)

@njit(cache=True, parallel=True)
def ds_line_chunked(ds: float, p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[int, np.ndarray]]:
    """
    Generate an array of points along a line segment.

    Parameters:
    ds (float): distance between points
    p1 (np.ndarray): starting point of the line segment
    p2 (np.ndarray): ending point of the line segment

    Returns:
    Tuple[int, np.ndarray]: number of points and array of points along the line segment
    """
    _d_vec = p2 - p1
    _distance_ = np.sqrt(np.sum(np.power(_d_vec, 2)))
    _versor = ds*(_d_vec/_distance_)
    num_points = int(np.floor(_distance_ /ds)) - 1
    points_arr = np.empty((1,2))
    if num_points >= 1 :
        points_arr = np.empty((num_points, 2))
        for i in prange(num_points):
            points_arr[i] = p1 + (i + 1) * _versor
    return num_points, points_arr

def closePolygon(array:np.ndarray) -> np.ndarray:
    arr_len = len(array)
    return  [np.array([array[i%arr_len] for i in range(arr_len+1)])]

center_point = lambda geometry : np.mean(geometry,axis=0)

def fill_geometry_with_points(ds:float,geometry:np.ndarray) -> np.ndarray:
    SLIDING_WINDOW_LEN = 2
    filled_geometry = []
    for i in range(len(geometry) - SLIDING_WINDOW_LEN + 1):
        window = geometry[i:i + SLIDING_WINDOW_LEN]
        filled_geometry.extend(np.array([window[0]]))
        points_generated = ds_line_chunked(ds, window[0], window[1])
        if points_generated[0] >= 1:
            filled_geometry.extend(points_generated[1])
        filled_geometry.extend(np.array([window[1]]))
    return np.array(filled_geometry)