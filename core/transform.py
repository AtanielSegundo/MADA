import numpy as np

def fold_3d_array_to_2d_using_NaN_separator(_3d_array: np.ndarray) -> np.ndarray:
    _1d_arrays = []
    for array_2d in _3d_array:
        _1d_arrays.extend(array_2d.flatten())
        _1d_arrays.extend([np.nan, np.nan])
    _1d_arrays = _1d_arrays[:-2]
    result = np.array(_1d_arrays).reshape(-1, 2)
    return result

def geometrys_from_txt_nan_separeted(txt_path:str,sep=" ",**kwargs) -> np.ndarray:
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

def calculate_elliptical_arc(start_point, arc_params):
    """Approximates the points along an elliptical arc."""
    rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y = arc_params
    arc_points = []
    arc_points.append(start_point)
    arc_points.append(np.array([end_x, end_y]))
    return arc_points

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
