import numpy as np
import pyslm
import matplotlib.pyplot as pyplot
from typing import List
from os.path import basename
from .visualize import ShowGeometrys, showSlices3d_matplot

def getSliceStl(stl_path:str,z:float=1,
                origin: List[float] = [5.0, 10.0, 0.0],
                rotation: np.ndarray[float] = np.array([0, 0, 30]),
                scale: float = 2.0,
                dropToPlataform: bool = True,) :
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    # Transform the part: Rotate, translate, scale, and drop to platform
    solidPart.scaleFactor = scale
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlataform:
        solidPart.dropToPlatform()
    return solidPart.getVectorSlice(z)

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
        showSlices3d_matplot(slices)


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
