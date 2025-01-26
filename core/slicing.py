import numpy as np
import pyslm
import matplotlib.pyplot as pyplot
from typing import List
from os.path import basename,join,dirname
from .visualize import ShowGeometrys

ROTATE_90_X =  np.array([90, 0, 0])
ROTATE_90_Y =  np.array([ 0,90, 0])
ROTATE_90_Z =  np.array([ 0, 0,90])

def showVectors3d_matplot(slices: List[List[np.ndarray]], 
                          fig_title: str = "",
                          thick: int = 1):
    from core.Tour import generateCH
    from core.Points.Grid import generateGridAndClusters
    from core.Layer import Layer
    from core.TSP.strategy import Strategy

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
            slice_2d = slice_3d[:, :2]
            grid, _ = generateGridAndClusters(Layer([slice_2d], "__test__", is_y_flipped=True),
                                               Strategy(), gen_clusters=False)
            path = generateCH(grid.points)
            path_points = []
            z = slice_3d[0,2]
            for idx in path:
                x,y = grid.points[idx]
                path_points.append([x,y,z])
            path_points = np.array(path_points)
            if len(path_points) > 1:
                ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
                        color='r', linewidth=thick)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    pyplot.title(fig_title)
    pyplot.show()

def getSliceStl(stl_path:str,z:float=1,
                origin: List[float] = [0.0, 0.0, 0.0],
                rotation: np.ndarray[float] = np.array([0, 0, 0]),
                scale: float = 1.0,
                dropToPlataform: bool = True) :
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    # Transform the part: Rotate, translate, scale, and drop to platform
    solidPart.scaleFactor = scale
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlataform:
        solidPart.dropToPlatform()
    return solidPart.getVectorSlice(z)

def transformStl(stl_path: str,
                 output_path: str = None,
                 origin: List[float] = [0.0, 0.0, 0.0],
                 rotation: np.ndarray = np.array([0, 0, 0]),
                 scale: float = 1.0,
                 dropToPlataform: bool = True) -> None:
    """
    Transforms an STL file by applying scaling, translation, rotation, 
    and optionally dropping it to the platform, then saves it.
    
    Args:
        stl_path (str): Path to the input STL file.
        output_path (str): Path to save the transformed STL file.
        origin (List[float]): Translation vector.
        rotation (np.ndarray): Rotation angles (in degrees, [x, y, z]).
        scale (float): Scaling factor.
        dropToPlataform (bool): Whether to drop the object to the platform.
    """
    output_path = output_path or dirname(stl_path)
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    solidPart.scaleFactor = scale
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlataform:
        solidPart.dropToPlatform()
    transformed_stl_path = join(output_path, f"{basename(stl_path).split('.')[0]}_transformed.stl")
    solidPart.geometry.export(transformed_stl_path)
    print(f"Transformed STL saved at: {transformed_stl_path}")

def sliceStlVector(stl_path: str, n_slices=6, z_step=14,
                   origin: List[float] = [5.0, 10.0, 0.0],
                   rotation: np.ndarray[float] = np.array([0, 0, 30]),
                   scaleFactor: float = 2.0,
                   dropToPlataform: bool = True,
                   offset_fn=None,
                   d2_mode=False, use_matplot=True):
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    # Transform the part: Rotate, translate, scale, and drop to platform
    solidPart.scaleFactor = scaleFactor
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlataform:
        solidPart.dropToPlatform()
    slices = []
    for i in range(n_slices):
        slice_2d_list = solidPart.getVectorSlice((i + 1) * z_step)
        if offset_fn:
            offset_fn(slice_2d_list)
        if d2_mode:
            slices.append(slice_2d_list)
        else:
            slice_3d_list = []
            for slice_2d in slice_2d_list:
                z_values = np.full(slice_2d.shape[0], (i + 1) * z_step)
                slice_3d = np.column_stack((slice_2d, z_values))
                slice_3d_list.append(slice_3d)
            slices.append(slice_3d_list)
    if d2_mode:
        ShowGeometrys(slices, spliter=1 if n_slices == 1 else 2)
    else:
        showVectors3d_matplot(slices)


def sliceStlRaster(stl_path: str, n_slices=6, z_step=14,
                   origin: List[float] = [5.0, 10.0, 0.0],
                   rotation: np.ndarray = np.array([0, 0, 30]),
                   scaleFactor: float = 2.0,
                   dropToPlatform: bool = True,
                   spliter=3):
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    solidPart.scaleFactor = scaleFactor
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlatform:
        solidPart.dropToPlatform()

    # Unity for rasterization is mm/px and dpi is px/inch
    dpi = 300.0
    resolution = 25.4 / dpi
    rows = n_slices//spliter + n_slices % spliter
    # Create subplots for the number of slices
    _, axs = pyplot.subplots(rows, spliter)

    if n_slices == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one slice

    for idx in range(n_slices):
        _slice = solidPart.getTrimeshSlice((idx + 1) * z_step)
        if _slice:
            sliceImage = _slice.rasterize(
                pitch=resolution, origin=solidPart.boundingBox[:2])
            axs[idx//spliter, idx %
                spliter].imshow(np.array(sliceImage), cmap='gray', origin='lower')
            axs[idx//spliter, idx % spliter].set_title(f"Slice {idx + 1}")
        else:
            axs[idx//spliter, idx %
                spliter].text(0.5, 0.5, "No Slice Data", ha='center', va='center')
            axs[idx//spliter, idx % spliter].axis('off')
    pyplot.show()
