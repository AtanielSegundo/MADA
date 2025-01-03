import numpy as np
from core.transform import geometrys_from_txt_nan_separeted
from core.clipper import readPathSVG
from core.slicing import getSliceStl
from core.visualize import SlicesPlotter
from os.path import basename

# format : (Handlers,is_y_flipped)
SUPPORTED = {"svg": (lambda p,**k : [readPathSVG(p,**k)], False),
             "txt": (geometrys_from_txt_nan_separeted, False),
             "stl": (getSliceStl, True)
             }

class Layer:
    #data is an array with shape (N,2)
    def __init__(self,data:np.ndarray,tag:str,is_y_flipped=False,**kwargs):
        for k,val in kwargs.items():
            setattr(self,k,val)
        self.is_y_flipped = is_y_flipped
        self.data = data
        self.tag = tag
    @classmethod
    def From(self,path:str,**kwargs):
        extension = path.split(".")[-1]
        if (_tuple:= SUPPORTED.get(extension,None)):
            handler,is_y_flipped = _tuple
            data = handler(path,**kwargs)
            return Layer(data,basename(path),is_y_flipped,**kwargs)
    
    def show(self):
        plotter = SlicesPlotter([self.data])
        plotter.show()

if __name__ == "__main__":
    l = Layer.From("assets\\svg\\rabbit.svg")
    print(l.data)
