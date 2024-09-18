import pyclipr
import numpy as np
import pyclipr.pyclipr
from commons.figutils import rotate_180,parse_svg_path,flip_horizontal
from core.geometry import ShowGeometrys,nthgone,center_point,CLOCKWISE
import xml.etree.ElementTree as ET
from typing import List

    
def offsetSVG(svg_path:str,DISTANCE=-5,ITER=40,HOLE_RAY=50):
    tree = ET.parse(svg_path)
    root = tree.getroot()   
    path_element = root.find('.//{http://www.w3.org/2000/svg}path')
    path_data = path_element.get('d')
    path_array = rotate_180(parse_svg_path(path_data))
    if HOLE_RAY > 0 :
        hole = nthgone(100,HOLE_RAY,center_p=center_point(path_array),dir=CLOCKWISE)
        rabbitOffsetedflat = offsetPaths([path_array,hole],DISTANCE,ITER)
        ShowGeometrys([[np.array(path_array),np.array(hole)],rabbitOffsetedflat])
    else :
        rabbitOffsetedflat = offsetPaths([path_array],DISTANCE,ITER)
        ShowGeometrys([[np.array(path_array)],rabbitOffsetedflat])

def closePolygon(array:np.ndarray) -> np.ndarray:
    arr_len = len(array)
    return  [np.array([array[i%arr_len] for i in range(arr_len+1)])]

def offsetPaths(Paths:List[np.ndarray],distance:float,iter:int) -> np.ndarray :
    offsetedFlat = Paths
    for _ in range(iter):
        new_temp = []
        po = pyclipr.ClipperOffset()
        po.scaleFactor = int(1000)            
        po.addPaths(Paths, pyclipr.JoinType.Miter, pyclipr.EndType.Polygon)
        resultArr = po.execute(distance)
        if resultArr:  
            resultArr = [np.array([result[i%len(result)] for i in range(len(result)+1)]) 
                         for result in resultArr if len(result) > 0]
            if resultArr:
                new_temp.extend(resultArr)  
                offsetedFlat.extend(resultArr)  
        Paths = new_temp
    return offsetedFlat  
    
def generateBridge(dim_size:float) -> np.ndarray :
    square = nthgone(4,dim_size) 
    circle = nthgone(100,dim_size/2,center_p=(0,3*dim_size/4)) 
    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths([square],pyclipr.Subject)
    pc.addPath(circle,pyclipr.Clip)
    result = pc.execute(pyclipr.Difference, pyclipr.FillRule.EvenOdd)[0]
    return rotate_180(np.array([result[i%len(result)] for i in range(len(result)+1)]))
    
def offsetBridge(bridge_size:float = 4,DISTANCE=-1,ITER=3) :
    bridge   = generateBridge(bridge_size)
    offseted = offsetPaths([bridge],DISTANCE,ITER)
    ShowGeometrys([[bridge],offseted],spliter=2)

def getRedStuffContours() :
    import cv2
    image = cv2.imread('assets/red_stuff.png')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)  
    upper_red1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)  
    upper_red2 = np.array([255, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_list = []
    for contour in contours:
        if len(contour) > 2:  
            contour_list.append(np.squeeze(contour, axis=1)) 
    contour_arr = np.array([contour_list[i%len(contour_list)] for i in range(len(contour_list)+1)])[0]
    return flip_horizontal(rotate_180(contour_arr))

def offsetRedStuff(DISTANCE=-40,ITER=1) :
    redStuff = getRedStuffContours()
    offseted = offsetPaths([redStuff],DISTANCE,ITER)
    ShowGeometrys([offseted],spliter=1)

if __name__ == "__main__" :
    offsetSVG("assets/rabbit.svg",HOLE_RAY=0)
    offsetBridge(DISTANCE=-1,ITER=3)
    offsetRedStuff(DISTANCE=-40,ITER=1)
    offsetRedStuff(DISTANCE=-10,ITER=40)