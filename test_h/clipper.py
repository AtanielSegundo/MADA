import numpy as np
import pyclipr.pyclipr
from core.clipper import offsetPaths
from core.geometry import rotate_180,flip_horizontal,nthgone
from core.visualize import ShowGeometrys
    
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
    image = cv2.imread('assets/png/red_stuff.png')
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
