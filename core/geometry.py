import numpy as np
from numba import njit,prange
from typing import *

ANTICLOCKWISE  = 1.0
EXTERNAL       = -1.0 
CLOCKWISE      = -1.0
INTERNAL       = 1.0

def online_mean(new_mean: np.ndarray, current_mean: np.ndarray, count):
    return (current_mean+(new_mean-current_mean)/(count+1), (count+1))

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


def generate_square_box_by_area(area: float, center: Tuple[float, float]):
    side_length = np.sqrt(area)
    half_side = side_length / 2
    square = [
        [center[0] - half_side, center[1] - half_side],
        [center[0] + half_side, center[1] - half_side],
        [center[0] + half_side, center[1] + half_side],
        [center[0] - half_side, center[1] + half_side],
        [center[0] - half_side, center[1] - half_side]
    ]
    return np.array(square)


def generate_square_box_by_lenght(side_lenght: float, center: Tuple[float, float]):
    half_side = side_lenght / 2
    #print(center)
    square = [
        [center[0] - half_side, center[1] - half_side],
        [center[0] + half_side, center[1] - half_side],
        [center[0] + half_side, center[1] + half_side],
        [center[0] - half_side, center[1] + half_side],
        [center[0] - half_side, center[1] - half_side]
    ]
    return np.array(square)

def generate_points_inside_square(square: np.ndarray, delta: float) -> np.ndarray:
    """
    Generate points inside the given square with the specified delta distance between x and y coordinates.

    Args:
        square (np.ndarray): A 2D array representing the square, with shape (5, 2).
        delta (float): The distance between x and y coordinates of the points.

    Returns:
        np.ndarray: A 2D array of points inside the square, with shape (n, 2).
    """
    min_x, min_y = np.min(square, axis=0)
    max_x, max_y = np.max(square, axis=0)
    x_coords = np.arange(min_x, max_x, delta)
    y_coords = np.arange(min_y, max_y, delta)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    inside_square = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & (
        points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    points_inside_square = points[inside_square]
    return points_inside_square

def rotate_180(points):
    """Rotates the points by 180 degrees."""
    center_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
    center_y = (np.min(points[:, 1]) + np.max(points[:, 1])) / 2
    translated_points = points - np.array([center_x, center_y])
    rotated_points = -translated_points
    rotated_points += np.array([center_x, center_y])
    return rotated_points


def flip_horizontal(points):
    """Flips the points horizontally."""
    center_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2  
    translated_points = points - np.array([center_x, 0])  
    flipped_points = np.array([-translated_points[:, 0], translated_points[:, 1]]).T  
    flipped_points += np.array([center_x, 0])  
    return flipped_points


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

def getPolygonsCenter(polygons):
    center = np.mean(polygons[0], axis=0)
    for idx, geometry in enumerate(polygons[1:]):
        center, _ = online_mean(center_point(geometry), center, idx+1)
    return center


def closePolygon(array:np.ndarray) -> np.ndarray:
    arr_len = len(array)
    return  [np.array([array[i%arr_len] for i in range(arr_len+1)])]

center_point = lambda geometry : np.mean(geometry,axis=0)

def fill_polyline_with_points(ds:float,polyline:np.ndarray) -> np.ndarray:
    SLIDING_WINDOW_LEN = 2
    filled_geometry = []
    for i in range(len(polyline) - SLIDING_WINDOW_LEN + 1):
        window = polyline[i:i + SLIDING_WINDOW_LEN]
        filled_geometry.extend(np.array([window[0]]))
        points_generated = ds_line_chunked(ds, window[0], window[1])
        if points_generated[0] >= 1:
            filled_geometry.extend(points_generated[1])
        filled_geometry.extend(np.array([window[1]]))
    return np.array(filled_geometry)

def generate_intermediate_points(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    distance = np.linalg.norm(end - start)
    num_steps = int(distance // step)
    intermediate_points = []
    for i in range(1, num_steps + 1):
        t = i / (num_steps + 1)
        intermediate_point = start + t * (end - start)
        intermediate_points.append(intermediate_point)
    return np.array(intermediate_points)


@njit(cache=True)
def point_in_polygon(x, y, polygon):
    """Ray-casting algorithm to check if a point is inside a polygon."""
    # Remove the last point if it duplicates the first point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# OLD IMPLEMENTATION
# def filter_invalid_points(geometry: List[np.ndarray], point_grid: np.ndarray, delta: float = -1) -> np.ndarray:
#     points = np.array([(x, y) for x, y in point_grid])
#     points_inside = np.zeros((len(points),), dtype=bool)
#     for polygon in geometry:
#         as_polygon = shapely.Polygon(polygon)
#         as_ring = shapely.LinearRing(polygon)
#         if as_polygon.exterior.is_ccw:
#             points_inside |= np.array([as_polygon.contains(shapely.Point(x, y)) and (
#                 as_ring.distance(shapely.Point(x, y)) >= delta) for x, y in points])
#         else:
#             points_inside &= np.array([not as_polygon.contains(shapely.Point(x, y)) and (
#                 as_ring.distance(shapely.Point(x, y)) >= delta) for x, y in points])
#     points_inside_array = points[points_inside]
#     return points_inside_array

@njit(cache=True)
def distance_to_polygon_edge(x, y, polygon):
    """Calculate the minimum distance from a point (x, y) to any edge of the polygon."""
    # Remove the last point if it duplicates the first point
    min_dist = np.inf
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        dist = np.sqrt((x - xx) ** 2 + (y - yy) ** 2)
        min_dist = min(min_dist, dist)
    return min_dist


@njit(cache=True)
def is_polygon_counterclockwise(polygon):
    """Calculate the signed area of the polygon to determine its orientation."""
    # Remove the last point if it duplicates the first point
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    return area < 0


@njit(cache=True)
def nb_filter_invalid_points(geometry: List[np.ndarray], point_grid: np.ndarray, delta: float = -1, fliped_y: bool = False) -> np.ndarray:
    points_inside = np.zeros((len(point_grid),), dtype=np.bool_)
    for polygon in geometry:
        if fliped_y ^ is_polygon_counterclockwise(polygon):
            for i in range(len(point_grid)):
                x, y = point_grid[i]
                points_inside[i] |= point_in_polygon(
                    x, y, polygon) and distance_to_polygon_edge(x, y, polygon) >= (delta)
        else:
            for i in range(len(point_grid)):
                x, y = point_grid[i]
                points_inside[i] &= not point_in_polygon(
                    x, y, polygon) and distance_to_polygon_edge(x, y, polygon) >= (delta)

    points_inside_array = point_grid[points_inside]
    return points_inside_array


def fill_geometrys_with_points(geometrys: List[np.ndarray], delta: float, figure_sep:float = 1, correction_factor: int = 4, fliped_y=False):
    geometrys_center = center_point(geometrys[0])
    for idx, geometry in enumerate(geometrys[1:]):
        geometrys_center, _ = online_mean(
            center_point(geometry), geometrys_center, idx+1)

    def get_center_max(geometry, center): return np.max(
        geometry - center, axis=0)

    def norma(arr): return np.sqrt(np.sum(arr*arr))
    max_point = get_center_max(geometrys[0], geometrys_center)
    for geometry in geometrys[1:]:
        curr_max = get_center_max(geometry, geometrys_center)
        if norma(max_point) < norma(curr_max):
            max_point = curr_max
    square_len = correction_factor*norma(max_point)
    square = generate_square_box_by_lenght(square_len, geometrys_center)
    points = generate_points_inside_square(square, delta)
    
    filtered_points = nb_filter_invalid_points(
        geometrys, points, delta=figure_sep, fliped_y=fliped_y)
    return filtered_points
