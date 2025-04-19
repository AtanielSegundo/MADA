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

class NoSolver(Solver):
    def __init__(self, temporary_folder,**kwargs):
        super().__init__(temporary_folder)
        self.distance_matrix_file = None
        self.initial_tour_file = None

    def setDistanceMatrix(self,points:np.ndarray,suffix:str="",start_end:Tuple[int, int]=None,**kwargs):
        pass
    
    def setInitialTourFile(self,path:np.ndarray,suffix:str="",**kwargs):
        pass

    def solve(self,points:np.ndarray,runs:int=None,seed:int=None,initial_path:np.ndarray=None,start_end:Tuple[int, int]=None,suffix:str="",**kwargs):
        from core.Tour import Tour
        assert points.shape[1] == 2, "Points must be a 2D array with shape (len, 2)"
        if initial_path is not None:
            return 1,Tour(initial_path)
        else:
            return 1,Tour(np.array([i for i in range(points.shape[0])]))