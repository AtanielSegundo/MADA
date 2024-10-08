from typing import List
import numpy as np
import matplotlib.pyplot as plt

def ShowGeometrys(geometrysList:List[List[np.ndarray]],fig_title=None,titles=None,
                  spliter=2,show_points=False) :
    n = len(geometrysList)
    marker_size = 1 if not show_points else 2
    line_ = "o-" if not show_points else 'o'
    if titles: assert len(titles) == n
    rows = n//spliter + n%spliter 
    _,axs = plt.subplots(rows,spliter)
    for ii,geometrys in enumerate(geometrysList) :
        for geometry in geometrys :
            if (rows) > 1 :
                axs[ii//spliter,ii%spliter].plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
            elif rows == 1 and spliter == 1:
                axs.plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
            else :
                axs[ii%spliter].plot(geometry[:,0],geometry[:,1],line_,markersize=marker_size)
                
        if titles : axs[ii//spliter,ii%spliter].set_title(titles[ii])
    if fig_title: plt.suptitle(fig_title,fontsize=22,weight='bold')  
    plt.show()