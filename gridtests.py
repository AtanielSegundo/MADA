from cliper_tests import offsetSVG,center_point,ShowGeometrys
from typing import Tuple,List
import numpy as np
import pyclipr
import shapely

def generate_square_box(area: float, center: Tuple[float, float]):
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
    points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    inside_square = np.logical_and(
        np.logical_and(points[:, 0] >= min_x, points[:, 0] <= max_x),
        np.logical_and(points[:, 1] >= min_y, points[:, 1] <= max_y)
    )
    points_inside_square = points[inside_square]
    return points_inside_square

def filter_points_that_is_not_inside(geometry: List[np.ndarray], point_grid: np.ndarray) -> np.ndarray:
    """
    Filter out points that are not inside the given geometry.

    Args:
        geometry (List[np.ndarray]): A list of 2D arrays representing the geometry, with shape (n, 2).
        point_grid (np.ndarray): A 2D array of points, with shape (m, 2).

    Returns:
        np.ndarray: A 2D array of points that are not inside the geometry, with shape (k, 2).
    """
    points = [shapely.Point(x, y) for x, y in point_grid]
    points_inside = []
    polygons = [shapely.Polygon(polygon) for polygon in geometry]
    for point in points:
        inside = False
        for polygon in polygons:
            if polygon.contains(point):
                if polygon.exterior.is_ccw:  # counter-clockwise is not a hole
                    inside = True
                else:  # clockwise is a hole
                    inside = False
                    break
        if inside:
            points_inside.append(point)
    points_inside_array = np.array([(point.x, point.y) for point in points_inside])
    return points_inside_array

if __name__ == "__main__" :
    rabbit,hole,offsets = offsetSVG("assets/rabbit.svg",HOLE_RAY=60,scale=1)
    form = [rabbit,hole]
    form_center = np.mean([center_point(rabbit),center_point(hole)],axis=0)
    # ShowGeometrys([[rabbit,hole,np.array([form_center,])]])
    pc2 = pyclipr.Clipper()
    pc2.scaleFactor = int(1e3)
    # print(dir(pyclipr))
    pc2.addPaths(form,pyclipr.PathType.Subject)
    pc2.addPaths(form,pyclipr.PathType.Clip)
    result = pc2.execute2(pyclipr.Union,pyclipr.FillRule.EvenOdd)
    area = result.area
    square = generate_square_box(10*area,form_center)
    grid_points = generate_points_inside_square(square,10)
    ShowGeometrys([[rabbit,hole,filter_points_that_is_not_inside([rabbit,hole],grid_points)]],show_points=True)