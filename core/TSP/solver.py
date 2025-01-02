import numpy as np
from abc import ABC, abstractmethod
from typing import *

class Solver(ABC):
    def __init__(self,temporary_folder:str,**kwargs):
        self.temporary_folder = temporary_folder
    @abstractmethod
    def setDistanceMatrix(self,points:np.ndarray,suffix:str="",start_end:Tuple[int, int]=None,**kwargs):
        pass
    @abstractmethod
    def setInitialTourFile(self,path:np.ndarray,suffix:str="",**kwargs):
        pass
    @abstractmethod
    def solve(self, points:np.ndarray,runs:int=None,seed:int=None,initial_path:np.ndarray=None,start_end:Tuple[int, int]=None,suffix:str="",**kwargs):
        pass
