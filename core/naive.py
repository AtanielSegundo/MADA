import numpy as np
from numba import njit,prange
from typing import *

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
def nb_offset(geometry: np.ndarray, magnitude: float, direction: int ,
              center:Tuple[float,float] = (0,0)) -> np.ndarray:
    new_geometry = np.zeros(geometry.shape)  
    for i in prange(len(geometry)):
        new_geometry[i] = nb_offset_point(geometry[i], magnitude, direction, center)
    return new_geometry