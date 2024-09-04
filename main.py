from core.geometry import nthgone,nb_offset,ShowGeometrys
from matplotlib import pyplot as plt
from time import process_time_ns

EXTERNAL = 1 ; INTERNAL=-1 

if __name__ == "__main__" :
    RAY = 0
    NUM = 2
    geometry   = nthgone(1000,RAY)
    start_time = process_time_ns()
    internals  = [nb_offset(geometry,(RAY/NUM)*i,direction=INTERNAL) for i in range(NUM+1)] 
    end_time   = process_time_ns()
    ShowGeometrys([[geometry]+internals])
    print((end_time-start_time)/1e6)
    plt.show()