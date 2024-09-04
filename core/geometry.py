import numpy as np
from numba import njit,prange
from typing import *
from matplotlib import pyplot as plt
ANTICLOCKWISE = 1
CLOCKWISE     = -1
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



def offset_point(point: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    x, y = point
    v_angle = np.arctan2(y, x)  
    off_x = direction * magnitude * np.cos(v_angle)
    off_y = direction * magnitude * np.sin(v_angle)
    return np.array([x + off_x, y + off_y])

def raw_offset(geometry: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    new_geometry = np.zeros((len(geometry), 2))  
    for i in range(len(geometry)):
        new_geometry[i] = offset_point(geometry[i], magnitude, direction)
    return new_geometry

@njit(cache=True)
def nb_offset_point(point: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    x, y = point
    v_angle = np.arctan2(y, x)  
    off_x = direction * magnitude * np.cos(v_angle)
    off_y = direction * magnitude * np.sin(v_angle)
    return np.array([x + off_x, y + off_y])

@njit(cache=True, parallel=True)
def nb_offset(geometry: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    new_geometry = np.zeros_like(geometry)  
    for i in prange(len(geometry)):
        new_geometry[i] = nb_offset_point(geometry[i], magnitude, direction)
    return new_geometry

def ShowGeometrys(geometrysList:List[List[np.ndarray]],titles=None) :
    n = len(geometrysList)
    if titles: assert len(titles) == n
    _,axs = plt.subplots(1,n)
    for ii,geometrys in enumerate(geometrysList) :
        for geometry in geometrys :
            if n > 1 :
                axs[ii].plot(geometry[:,0],geometry[:,1],"go-",markersize=1)
            else :
                axs.plot(geometry[:,0],geometry[:,1],"go-",markersize=1)
        if titles : axs[ii].set_title(titles[ii])