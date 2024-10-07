from matplotlib import pyplot
from core.visualize import showStl 
from core.slicing import sliceStlVector
from os.path import basename 
import numpy as np
import pyslm
from typing import *


if __name__ == '__main__' :
    # showStl("assets/3d/bonnie.stl")
    # sliceStlVector('assets/3d/bonnie.stl',z_step=2,n_slices=200,scaleFactor=1.0)
    
    # showStl("assets/3d/bonnie.stl")
    # sliceStlVector('assets/3d/bonnie.stl',n_slices=200,z_step=2,scaleFactor=1)
    # sliceStlVector('assets/3d/bonnie.stl',n_slices=20,z_step=20,scaleFactor=1)

    # showStl("assets/3d/Flange_inventor.stl")
    # sliceStlVector("assets/3d/Flange_inventor.stl",n_slices=20,z_step=2.5,scaleFactor=1)
    
    # showStl("assets/3d/flange16furos.stl")
    # sliceStlVector("assets/3d/flange16furos.stl",n_slices=20,z_step=2.5,scaleFactor=1)

    showStl("assets/3d/truss_.stl")
    sliceStlVector("assets/3d/truss_.stl",n_slices=100,z_step=1,scaleFactor=1)
    
    showStl("assets/3d/Petro_foice_c.stl")
    sliceStlVector("assets/3d/Petro_foice_c.stl",n_slices=100,z_step=1,scaleFactor=1)

    # showStl("assets/3d/frameGuide.stl")
    # sliceStlVector('assets/3d/frameGuide.stl',n_slices=100,z_step=1,scaleFactor=1)
    # sliceStlVector('assets/3d/frameGuide.stl',n_slices=10,z_step=4,scaleFactor=1)