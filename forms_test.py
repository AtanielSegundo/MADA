from commons.clipperutils import offsetPaths
from core.geometry import ShowGeometrys,nthgone,center_point,CLOCKWISE
from core.geometry import fold_3d_array_to_2d_using_NaN_separator
from core.geometry import geometrys_from_txt_nan_separeted
import os
import numpy as np 

def OffsetFromTxtFile(file_path:str,iter:int=40,offset=-2,precisao=1e3) :
    geometrys = geometrys_from_txt_nan_separeted(file_path)
    geometrys_offset = offsetPaths(geometrys.copy(),offset,iter=iter,precisao=precisao)
    ShowGeometrys([geometrys,geometrys_offset],spliter=2)
    
if __name__ == "__main__" :
    ITERACOES = 40
    OFFSET    = -2
    PATH      = "assets/formas"
    PRECISAO  = 1E3     #PRECISAO DE 3 DIGITOS 
    for arquivo in os.listdir(PATH) :
        print(f"Lendo {arquivo}")
        OffsetFromTxtFile(os.path.join(PATH,arquivo),ITERACOES,OFFSET,precisao=PRECISAO)