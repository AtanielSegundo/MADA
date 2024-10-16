import pyclipr.pyclipr
from core.geometry import nthgone, nb_offset, ShowGeometrys, INTERNAL, EXTERNAL, center_point, fill_geometry_with_points, nb_points
from matplotlib import pyplot as plt
from assets.geometrys import triangle
from time import process_time_ns
import numpy as np
import pyclipr


def offset_test():
    RAY = 3
    NUM = 2
    CENTER = (1, 2)
    geometry = nthgone(4, RAY, center_p=CENTER)
    start_time = process_time_ns()
    internals = [nb_offset(geometry, (RAY/(NUM+1))*(i+1),
                           direction=INTERNAL, center=CENTER) for i in range(NUM)]
    end_time = process_time_ns()
    print((end_time-start_time)/1e6)
    ShowGeometrys([[geometry]+internals])


def show_fill_geometry():
    RAY = 2
    CENTER = (2, 4)
    OFFSET = 0.05
    geometry = nthgone(12, RAY, center_p=CENTER)
    start_time = process_time_ns()
    filled = [fill_geometry_with_points(OFFSET, nb_offset(geometry, OFFSET*i, direction=INTERNAL, center=CENTER))
              for i in range(int(RAY/OFFSET)+1)]
    end_time = process_time_ns()
    print((end_time-start_time)/1e6)
    ShowGeometrys([filled], show_points=True)


# offset = [np.array(offset_curve(Polygon(geometry),ds*(i+1),8,join_style="mitre").coords) for i in range(4)]
    # geometry = fill_geometry_with_points(ds,nthgone(4,3,center_p=(6,5)))
    # offset = nb_offset(geometry,0.2,INTERNAL,center=center_point(geometry))
    # print(offset)
    # print([geometry[1],offset[0][1]])
    # print(np.sqrt(np.sum(np.power(geometry[1]-offset[0][1], 2))))
    # ShowGeometrys([[geometry]+offset],show_points=True)

if __name__ == "__main__":
    # ds = 0.3
    # geometry = nthgone(4,3,center_p=(6,5))
    # po = pyclipr.ClipperOffset()
    # po.scaleFactor = int(1e-3)
    # po.addPaths([geometry],pyclipr.JoinType.Miter,pyclipr.EndType.Polygon)
    # offset = po.execute(2)
    # print(offset)
    # Tuple definition of a path
    path = [(0.0, 0.), (0, 105.1234), (100, 105.1234), (100, 0), (0, 0)]
    path2 = [(1.0, 1.0), (1.0, 50), (100, 50), (100, 1.0), (1.0, 1.0)]

    # Create an offsetting object
    po = pyclipr.ClipperOffset()

    # Set the scale factor to convert to internal integer representation
    po.scaleFactor = int(1000)

    # add the path - ensuring to use Polygon for the endType argument
    # addPaths is required when working with polygon - this is a list of correctly orientated paths for exterior
    # and interior holes
    po.addPaths([np.array(path)], pyclipr.JoinType.Miter,
                pyclipr.EndType.Polygon)

    # Apply the offsetting operation using a delta.
    offsetSquare = np.array(po.execute(-2)[0])
    print(offsetSquare)
    ShowGeometrys([[np.array(path), offsetSquare]], spliter=2)
