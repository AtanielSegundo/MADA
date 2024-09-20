import numpy as np
import pyclipr.pyclipr
from commons.clipperutils import offsetPaths
from commons.figutils import rotate_180,parse_svg_path,flip_horizontal
from core.geometry import ShowGeometrys,nthgone,center_point,CLOCKWISE
from core.geometry import fold_3d_array_to_2d_using_NaN_separator
import xml.etree.ElementTree as ET
from typing import List

    
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

    
def generateBridge(dim_size:float) -> np.ndarray :
    square = nthgone(4,dim_size) 
    circle = nthgone(100,dim_size/2,center_p=(0,3*dim_size/4)) 
    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths([square],pyclipr.Subject)
    pc.addPath(circle,pyclipr.Clip)
    result = pc.execute(pyclipr.Difference, pyclipr.FillRule.EvenOdd)[0]
    return rotate_180(np.array([result[i%len(result)] for i in range(len(result)+1)]))
    
def offsetBridge(bridge_size:float = 4,DISTANCE=-1,ITER=3,show=False) :
    bridge   = generateBridge(bridge_size)
    offseted = offsetPaths([bridge],DISTANCE,ITER)
    if show : ShowGeometrys([[bridge],offseted],spliter=2)
    return bridge,offseted

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

def offsetRedStuff(DISTANCE=-40,ITER=1,show=False) :
    redStuff = getRedStuffContours()
    offseted = offsetPaths([redStuff],DISTANCE,ITER)
    if show: ShowGeometrys([offseted],spliter=1)
    return redStuff,offseted

if __name__ == "__main__" :
    # offsetSVG("assets/rabbit.svg",HOLE_RAY=0)
    
    # rabbit,hole,offsets = offsetSVG("assets/rabbit.svg",HOLE_RAY=40,scale=1,show=True)
    # fig = np.concatenate((rabbit,np.array([[np.nan,np.nan]]),hole), axis=0)
    # np.savetxt("assets/rabbit_hole.txt",fig,delimiter=",")
    # np.savetxt("assets/offsets_hole.txt",fold_3d_array_to_2d_using_NaN_separator(offsets),delimiter=",")
    
    # rabbit,hole,offsets = offsetSVG("assets/rabbit.svg",HOLE_RAY=0,scale=1,show=True)
    # np.savetxt("assets/rabbit.txt",rabbit,delimiter=",")
    # np.savetxt("assets/offsets.txt",fold_3d_array_to_2d_using_NaN_separator(offsets),delimiter=",")


    # bridge,offseted   = offsetBridge(DISTANCE=-1,ITER=3,show=True)
    # 
    # np.savetxt("assets/bridge.txt",bridge,delimiter=",")
    # np.savetxt("assets/bridge_offset.txt",fold_3d_array_to_2d_using_NaN_separator(offseted),delimiter=",")
    
    redstuff,offseted = offsetRedStuff(DISTANCE=-40,ITER=1,show=True)
    np.savetxt("assets/global_loop.txt",redstuff,delimiter=",")
    np.savetxt("assets/global_loop_offset.txt",fold_3d_array_to_2d_using_NaN_separator(offseted),delimiter=",")
    

    # offsetRedStuff(DISTANCE=-10,ITER=40)