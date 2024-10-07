from core.geometry import ShowGeometrys,nb_offset,INTERNAL,EXTERNAL,nthgone,raw_offset,center_point
from geometrys_pr import circle,square,triangle,hexagon
import numpy as np

def inter(n,geometry,r=None,dir=INTERNAL) :
    if not r :
        x,y = geometry[0] 
        r   = np.sqrt(x*x+y*y)
    return [nb_offset(geometry,i*(r/(n)),dir) for i in range(n)]
    

# square = nthgone(4,1)
# print(square)
N = 3

#   ORIGINAL
# ShowGeometrys([[triangle],
            #    [hexagon],
            #    [square],
            #    [circle]],
            #    fig_title="Geometrias Originais",
            #    titles=["Trinagulo","Hexagono","Quadrado","Circulo"]
            #    ,spliter=2)
DIR = INTERNAL
ShowGeometrys([[triangle]+inter(N,triangle,DIR),
               [hexagon]+inter(N,hexagon,DIR),
               [square]+inter(N,square,DIR),
               [circle]+inter(N,circle,DIR)],
               fig_title="Offset Interno (Ilhas)",
               titles=["Trinagulo","Hexagono","Quadrado","Circulo"]
               ,spliter=2)
DIR = EXTERNAL
ShowGeometrys([[triangle]+inter(N,triangle,DIR),
               [hexagon]+inter(N,hexagon,DIR),
               [circle]+inter(N,circle,DIR),
               [square]+inter(N,square,DIR)],
               fig_title="Offset Externo (Furos)",
               titles=["Trinagulo","Hexagono","Circulo","Quadrado"],
               spliter=2)