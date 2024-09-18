import numpy as np

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

from math import cos, sin, radians, sqrt, atan2

def parse_svg_path(path_data):
    """Parses the SVG path data into a list of points, including elliptical arcs."""
    points = []
    current_position = np.array([0.0, 0.0])  
    commands = 'mlMLA'
    
    path_elements = path_data.split()
    
    i = 0
    while i < len(path_elements):
        element = path_elements[i]
        
        if element in commands:
            command = element
            i += 1
        else:
            command = 'l'  

        if command == 'A':
            
            rx, ry = float(path_elements[i]), float(path_elements[i + 1])
            x_axis_rotation = float(path_elements[i + 2])
            large_arc_flag = int(path_elements[i + 3])
            sweep_flag = int(path_elements[i + 4])
            x, y = float(path_elements[i + 5]), float(path_elements[i + 6])
            arc_params = (rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, x, y)
            
            
            arc_points = calculate_elliptical_arc(current_position, arc_params)
            points.extend(arc_points)
            
            
            current_position = np.array([x, y])
            i += 7  

        else:
            coords = path_elements[i].split(',')
            
            if len(coords) == 2:
                x, y = map(float, coords)
                point = np.array([x, y])
                
                if command == 'm':  
                    current_position += point
                elif command == 'M':  
                    current_position = point
                elif command == 'l':  
                    current_position += point
                elif command == 'L':  
                    current_position = point
                
                points.append(current_position.copy())
            i += 1
    
    return np.array(points)

def calculate_elliptical_arc(start_point, arc_params):
    """Approximates the points along an elliptical arc."""
    
    rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y = arc_params

    arc_points = []
    arc_points.append(start_point)
    arc_points.append(np.array([end_x, end_y]))
    
    return arc_points