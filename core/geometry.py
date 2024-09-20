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

def offset_point(point: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    x, y = point
    v_angle = np.arctan2(y, x)  
    off_x = direction * magnitude * np.cos(v_angle)
    off_y = direction * magnitude * np.sin(v_angle)
    return np.array([x - off_x, y - off_y])

def raw_offset(geometry: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    new_geometry = np.zeros((len(geometry), 2))  
    for i in range(len(geometry)):
        new_geometry[i] = offset_point(geometry[i], magnitude, direction)
    return new_geometry

@njit(cache=True)
def nb_offset_point(point: np.ndarray, magnitude: float, direction: int, center:Tuple[float,float]) -> np.ndarray:
    x, y = point
    v_angle = np.arctan2(y-center[1],x-center[0])  
    off_x = direction * magnitude * np.cos(v_angle)
    off_y = direction * magnitude * np.sin(v_angle)
    return np.array([x - off_x, y - off_y])

@njit(cache=True, parallel=True)
def nb_points(ds: float, p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[int, np.ndarray]]:
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

def fill_geometry_with_points(ds:float,geometry:np.ndarray) -> np.ndarray:
    SLIDING_WINDOW_LEN = 2
    filled_geometry = []
    for i in range(len(geometry) - SLIDING_WINDOW_LEN + 1):
        window = geometry[i:i + SLIDING_WINDOW_LEN]
        filled_geometry.extend(np.array([window[0]]))
        points_generated = nb_points(ds, window[0], window[1])
        if points_generated[0] >= 1:
            filled_geometry.extend(points_generated[1])
        filled_geometry.extend(np.array([window[1]]))
    return np.array(filled_geometry)
    
@njit(cache=True, parallel=True)
def nb_offset(geometry: np.ndarray, magnitude: float, direction: int ,
              center:Tuple[float,float] = (0,0)) -> np.ndarray:
    new_geometry = np.zeros(geometry.shape)  
    for i in prange(len(geometry)):
        new_geometry[i] = nb_offset_point(geometry[i], magnitude, direction, center)
    return new_geometry

def center_point(geometry:np.ndarray) :
    return np.mean(geometry,axis=0)
    
def fold_3d_array_to_2d_using_NaN_separator(_3d_array: np.ndarray) -> np.ndarray:
    _1d_arrays = []
    for array_2d in _3d_array:
        _1d_arrays.extend(array_2d.flatten())
        _1d_arrays.extend([np.nan, np.nan])
    _1d_arrays = _1d_arrays[:-2]
    result = np.array(_1d_arrays).reshape(-1, 2)
    return result

def closePolygon(array:np.ndarray) -> np.ndarray:
    arr_len = len(array)
    return  [np.array([array[i%arr_len] for i in range(arr_len+1)])]

def ShowGeometrys(geometrysList:List[List[np.ndarray]],fig_title=None,titles=None,
                  spliter=2,show_points=False) :
    n = len(geometrysList)
    marker_size = 1 if not show_points else 2
    line_ = "o-" if not show_points else 'o'
    if titles: assert len(titles) == n
    rows = n//spliter + n%spliter 
    _,axs = plt.subplots(rows,spliter)
    for ii,geometrys in enumerate(geometrysList) :
        for geometry in geometrys :
            if (rows) > 1 :
                axs[ii//spliter,ii%spliter].plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
            elif rows == 1 and spliter == 1:
                axs.plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
            else :
                axs[ii%spliter].plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
                
        if titles : axs[ii//spliter,ii%spliter].set_title(titles[ii])
    if fig_title: plt.suptitle(fig_title,fontsize=22,weight='bold')
    plt.show()

def geometrys_from_txt_nan_separeted(txt_path:str,sep=" ") -> np.ndarray:
    geometrys = []
    raw_geometrys  = np.fromfile(txt_path,sep=sep).reshape(-1,2)
    nans_idx = list(np.argwhere(np.isnan(raw_geometrys))[:,0])[::2]
    nans_len = len(nans_idx)
    if nans_len == 0 : return [raw_geometrys]
    ii = -1                 # Comeca em -1 para preservar logica do loop
    for jj in range(nans_len):
        geometrys.append(raw_geometrys[ii+1:nans_idx[jj]])
        ii = nans_idx[jj]
        if jj == nans_len - 1 :
            geometrys.append(raw_geometrys[ii+1:])
    return geometrys