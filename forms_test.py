from commons.clipperutils import offsetPaths
from core.geometry import ShowGeometrys,nthgone
from core.geometry import geometrys_from_txt_nan_separeted
import os
import numpy as np 

def OffsetFromTxtFile(file_path:str,iter:int=40,offset=-2,precisao=1e3) :
    geometrys = geometrys_from_txt_nan_separeted(file_path)
    geometrys_offset = offsetPaths(geometrys.copy(),offset,iter=iter,precisao=precisao)
    ShowGeometrys([geometrys,geometrys_offset],spliter=2)

def create_two_semi_circles(ray:float,separation:float,c_points=1000) :
    semi_circle_1 = nthgone(c_points,ray=ray)[500:]
    semi_circle_1 = np.array([semi_circle_1[idx%len(semi_circle_1)] for idx in range(len(semi_circle_1)+1)])
    semi_circle_2 = nthgone(c_points,ray=ray,center_p=(0,separation))[0:501]
    semi_circle_2 = np.array([semi_circle_2[idx%len(semi_circle_2)] for idx in range(len(semi_circle_2)+1)])
    return [semi_circle_1,semi_circle_2]

if __name__ == "__main__" :
    ITERACOES = 40
    OFFSET    = -2
    PATH      = "assets/formas"
    PRECISAO  = 1E3     #PRECISAO DE 3 DIGITOS 
    for arquivo in os.listdir(PATH) :
        print(f"Lendo {arquivo}")
        OffsetFromTxtFile(os.path.join(PATH,arquivo),ITERACOES,OFFSET,precisao=PRECISAO)
    # two_semi_circles_nthgone = create_two_semi_circles(50,25)
    # ShowGeometrys([two_semi_circles_nthgone,offsetPaths(two_semi_circles_nthgone.copy(),distance=OFFSET,iter=ITERACOES)],spliter=2)