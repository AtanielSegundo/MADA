import numpy as np
import pyclipr.pyclipr
from commons.clipperutils import offsetPaths
from commons.figutils import rotate_180,flip_horizontal
from core.geometry import ShowGeometrys,nthgone,CLOCKWISE
    
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

if __name__ == "__main__" :
    #offsetSVG("assets/rabbit.svg",HOLE_RAY=0)

    #rabbit,hole,offsets = offsetSVG("assets/rabbit.svg",HOLE_RAY=40,scale=1,show=True)
    #fig = np.concatenate((rabbit,np.array([[np.nan,np.nan]]),hole), axis=0)
    #np.savetxt("assets/rabbit_hole.txt",fig,delimiter=",")
    #np.savetxt("assets/offsets_hole.txt",fold_3d_array_to_2d_using_NaN_separator(offsets),delimiter=",")

    #rabbit,hole,offsets = offsetSVG("assets/rabbit.svg",HOLE_RAY=0,scale=1,show=True)
    #np.savetxt("assets/rabbit.txt",rabbit,delimiter=",")
    #np.savetxt("assets/offsets.txt",fold_3d_array_to_2d_using_NaN_separator(offsets),delimiter=",")
    #bridge,offseted   = offsetBridge(DISTANCE=-1,ITER=3,show=True)
    #
    #np.savetxt("assets/bridge.txt",bridge,delimiter=",")
    #np.savetxt("assets/bridge_offset.txt",fold_3d_array_to_2d_using_NaN_separator(offseted),delimiter=",")

    #redstuff,offseted = offsetRedStuff(DISTANCE=-40,ITER=1,show=True)
    #np.savetxt("assets/global_loop.txt",redstuff,delimiter=",")
    #np.savetxt("assets/global_loop_offset.txt",fold_3d_array_to_2d_using_NaN_separator(offseted),delimiter=",")

    circle = nthgone(10_000,10)
    hole  = nthgone(10_000,1,dir=CLOCKWISE)
    offsets = offsetPaths([circle.copy(),hole.copy()],-1,40)
    ShowGeometrys([[circle,hole],offsets])
    # offsetRedStuff(DISTANCE=-10,ITER=40)