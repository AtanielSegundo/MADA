import numpy as np
import xml.etree.ElementTree as ET
import pyclipr
from typing import List 
from core.geometry import nthgone,center_point,CLOCKWISE
from core.visualize import ShowGeometrys
from core.geometry import rotate_180
from core.transform import parse_svg_path

def offsetPaths(Paths:List[np.ndarray],distance:float,iter:int,precisao:float=1e3) -> np.ndarray :
    offsetedFlat = Paths
    _Paths = Paths
    for _ in range(iter):
        new_temp = []
        po = pyclipr.ClipperOffset()
        po.scaleFactor = int(precisao)           
        po.addPaths(_Paths,pyclipr.JoinType.Round,pyclipr.EndType.Polygon)
        resultArr = po.execute(distance)
        if resultArr:  
            resultArr = [np.array([result[i%len(result)] for i in range(len(result)+1)]) 
                         for result in resultArr if len(result) > 0]
            if resultArr:
                new_temp.extend(resultArr)  
                offsetedFlat.extend(resultArr)  
        _Paths = new_temp
    return offsetedFlat  

def offsetSVG(svg_path:str,DISTANCE=-5,ITER=40,HOLE_RAY=50,scale=1,show=False):
    tree = ET.parse(svg_path)
    root = tree.getroot()   
    DISTANCE = -5*scale
    HOLE_RAY = HOLE_RAY*scale
    path_element = root.find('.//{http://www.w3.org/2000/svg}path')
    path_data = path_element.get('d')
    path_array = rotate_180(parse_svg_path(path_data))*scale
    if HOLE_RAY > 0 :
        hole = nthgone(100,HOLE_RAY,center_p=center_point(path_array),dir=CLOCKWISE)
        rabbitOffsetedflat = offsetPaths([path_array,hole],DISTANCE,ITER)
        if show : ShowGeometrys([[np.array(path_array),np.array(hole)],rabbitOffsetedflat])
        return np.array(path_array),np.array(hole),rabbitOffsetedflat
    else :
        rabbitOffsetedflat = offsetPaths([path_array],DISTANCE,ITER)
        if show : ShowGeometrys([[np.array(path_array)],rabbitOffsetedflat])
        return np.array(path_array),None,rabbitOffsetedflat