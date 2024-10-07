import pyslm
import numpy as np
import matplotlib.pyplot as pyplot
from os.path import basename
from typing import List
from ._slice_viewer import SliceViewer,pyglet

def showStl(stl_path:str,scale=0.1) :
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    solidPart.scaleFactor = scale
    solidPart.dropToPlatform() 
    solidPart.geometry.show()

def showSlices3d(slices:List[List[np.ndarray]]):
    """
    Use Pyglet to visualize the 3D slices in a 3D space.

    Parameters:
    - slices: A list of lists of 3D numpy arrays, where each numpy array represents
              a set of points in the 3D space.
    """
    window = SliceViewer(slices)
    pyglet.app.run()


def showSlices3d_matplot(slices: List[List[np.ndarray]], fig_title: str = ""):
    """
    Visualize the 3D slices in a single 3D plot.

    Parameters:
    - slices: List of lists of 3D numpy arrays. Each sublist corresponds to a slice, 
              and each numpy array within the sublist corresponds to a part of that slice.
    - fig_title: Title of the figure (default is '3D Slices Visualization')
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for slice_3d_list in slices:
        for slice_3d in slice_3d_list:
            ax.plot(slice_3d[:, 0], slice_3d[:, 1], slice_3d[:, 2],linestyle='-')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    pyplot.title(fig_title)
    pyplot.show()