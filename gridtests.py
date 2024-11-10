import numpy as np
import pyclipr
import shapely
from typing import Tuple, List
from commons.utils.clipper import offsetSVG, offsetTXT
from commons.utils.math import online_mean
from core.geometry import center_point
from core.visualize import showStl, ShowGeometrys
import numba as nb
from numba import njit


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


def filter_invalid_points(geometry: List[np.ndarray], point_grid: np.ndarray, delta: float = -1) -> np.ndarray:
    points = np.array([(x, y) for x, y in point_grid])
    points_inside = np.zeros((len(points),), dtype=bool)
    for polygon in geometry:
        as_polygon = shapely.Polygon(polygon)
        as_ring = shapely.LinearRing(polygon)
        if as_polygon.exterior.is_ccw:
            points_inside |= np.array([as_polygon.contains(shapely.Point(x, y)) and (
                as_ring.distance(shapely.Point(x, y)) >= delta) for x, y in points])
        else:
            points_inside &= np.array([not as_polygon.contains(shapely.Point(x, y)) and (
                as_ring.distance(shapely.Point(x, y)) >= delta) for x, y in points])
    points_inside_array = points[points_inside]
    return points_inside_array


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


if __name__ == "__main__":
    DISTANCE = 3
    ITER = 2
    rabbit, hole, offsets = offsetSVG("assets/svg/rabbit.svg", DISTANCE=-DISTANCE, ITER=ITER,
                                      HOLE_RAY=60, scale=1)
    # ShowGeometrys([[rabbit,hole,fill_geometrys_with_points([rabbit,hole],2)]])
    geometrys, offset = offsetTXT(
        "assets\\txt\\formas\\teste_biela.txt", iter=ITER, offset=-DISTANCE)
    geometrys = [rabbit, hole]
    points = [fill_geometrys_with_points(geometrys, DISTANCE)]
    # ShowGeometrys([geometrys], spliter=2, points_grids=points)
