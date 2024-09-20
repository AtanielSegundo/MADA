import numpy as np
from typing import List 
import pyclipr

def offsetPaths(Paths:List[np.ndarray],distance:float,iter:int) -> np.ndarray :
    offsetedFlat = Paths
    _Paths = Paths
    for _ in range(iter):
        new_temp = []
        po = pyclipr.ClipperOffset()
        po.scaleFactor = int(1e3)            
        po.addPaths(_Paths, pyclipr.JoinType.Miter,pyclipr.EndType.Polygon)
        resultArr = po.execute(distance)
        if resultArr:  
            resultArr = [np.array([result[i%len(result)] for i in range(len(result)+1)]) 
                         for result in resultArr if len(result) > 0]
            if resultArr:
                new_temp.extend(resultArr)  
                offsetedFlat.extend(resultArr)  
        _Paths = new_temp
    return offsetedFlat  