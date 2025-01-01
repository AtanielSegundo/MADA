import os
import shutil
import numpy as np
import subprocess
import tempfile
import warnings
from lkh import LKHProblem,NoToursException
from typing import *
from core.TSP.solver import Solver
from core.Points.operations import compute_distance_matrix_numba_parallel
from core.Tour import Tour

def custom_lkh_solve(solver='LKH', problem=None,merge_tour_files=None, **params):
    assert shutil.which(solver) is not None, f'{solver} not found.'
    assert ('problem_file' in params) ^ (problem is not None), 'Specify a problem object *or* a path.'
    if 'problem_file' in params:
        problem = LKHProblem.load(params['problem_file'])
    if not isinstance(problem, LKHProblem):
        warnings.warn('Subclassing LKHProblem is recommended. Proceed at your own risk!')
    if len(problem.depots) > 1:
        warnings.warn('LKH-3 cannot solve multi-depot problems.')
    prob_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    problem.write(prob_file)
    prob_file.write('\n')
    prob_file.close()
    params['problem_file'] = prob_file.name
    if 'tour_file' not in params:
        tour_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        params['tour_file'] = tour_file.name
        tour_file.close()
    par_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    par_file.write('SPECIAL\n')
    for k, v in params.items():
        if v is None:
            pass
        else:
            par_file.write(f'{k.upper()} = {v}\n')
    if merge_tour_files:
        for f in merge_tour_files:
            par_file.write(f'MERGE_TOUR_FILE = {f}\n')
    par_file.close()
    try:
        subprocess.check_output([solver, par_file.name], stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())
    if not os.path.isfile(params['tour_file']) or os.stat(params['tour_file']).st_size == 0:
        raise NoToursException(f"{params['tour_file']} does not appear to contain any tours. LKH probably did not find solution.")
    solution = LKHProblem.load(params['tour_file'])
    lenght = [s for s in str(solution).split("\n") if 'NAME:' in s][0].split(":")[-1].split(".")[1]
    tour = solution.tours[0]
    routes = []
    route = []
    for node in tour:
        if node in problem.depots or node > problem.dimension:
            if len(route) > 0:
                routes.append(route)
            route = []
        else:
            route.append(node)
    routes.append(route)
    os.remove(par_file.name)
    if 'prob_file' in locals():
        os.remove(prob_file.name)
    if 'tour_file' in locals():
        os.remove(tour_file.name)
    return int(lenght),routes


class LKH(Solver):
    def __init__(self, temporary_folder,lkh_exe="core/TSP/LKH/LKH.exe",**kwargs):
        super().__init__(temporary_folder)
        self.lkh_exe = lkh_exe
        self.distance_matrix_file = None
        self.initial_tour_file = None

    def setDistanceMatrix(self,points:np.ndarray,start_end:Tuple[int, int]=None,**kwargs):
        self.distance_matrix_file = os.path.join(self.temporary_folder,"points_distance_matrix_file.txt")
        distance_matrix = compute_distance_matrix_numba_parallel(points,points)
        writeDistanceMatrixProblemFile(distance_matrix,self.distance_matrix_file,start_end)

    def setInitialTourFile(self,path:np.ndarray,**kwargs):
        self.initial_tour_file = os.path.join(self.temporary_folder,"initial_tour_file.txt")
        writeInitialTourFile(path,self.initial_tour_file)
        
    def solve(self,points:np.ndarray,runs:int=None,seed:int=None,initial_path:np.ndarray=None,start_end:Tuple[int, int]=None,**kwargs) -> Tuple[int,Tour]:
        assert points.shape[1] == 2, "Points must be a 2D array with shape (len, 2)"
        self.setDistanceMatrix(points,start_end)
        if initial_path is not None:
            self.setInitialTourFile(initial_path)
            lenght,best_path = custom_lkh_solve(self.lkh_exe, 
                               initial_tour_file=self.initial_tour_file, 
                               problem_file=self.distance_matrix_file, 
                               seed=seed,
                               runs=runs)
        else:
            lenght,best_path = custom_lkh_solve(self.lkh_exe, 
                               problem_file=self.distance_matrix_file, 
                               seed=seed,
                               runs=runs)
        best_path = np.array(best_path[0]) - 1 
        return lenght,Tour(best_path)



def writePointsProblemFile(points:np.ndarray,file_name:str) :
    dimension = points.shape[0]
    with open(file_name,"w") as f:
        f.write("NAME: PointsProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for idx,point in enumerate(points):
            f.write(f"{idx+1} ")
            f.write(" ".join(map(lambda e: f"{e:.3e}",point)))
            f.write("\n")

            
def writeInitialTourFile(tour:np.ndarray,file_name:str) :
    dimension = len(tour)
    with open(file_name,"w") as f:
        f.write("NAME: InitialTourFile\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("TOUR_SECTION\n")
        f.write("\n".join(map(lambda e : str(e+1),tour)))
        f.write("\n-1")

def writeDistanceMatrixProblemFile(distance_matrix: np.ndarray, file_name: str, dummy_edge: Tuple[int, int] = None):
    with open(file_name, "w") as f:
        dimension = distance_matrix.shape[0] + (1 if dummy_edge else 0)
        f.write("NAME: DistanceMatrixProblem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        start_node, end_node = dummy_edge if dummy_edge else (None, None)
        for i, row in enumerate(distance_matrix):
            if dummy_edge:
                formatted_row = [
                    "1e-6" if j == dimension - 1 and (i == start_node or i == end_node) else f"{val:.3f}" 
                    for j, val in enumerate(np.append(row, [1e6] if dummy_edge else []))
                ]
            else:
                formatted_row = [f"{val:.3f}" for val in row]
            f.write(" ".join(formatted_row) + "\n")

        if dummy_edge:
            dummy_row = ["1e-6" if j == start_node or j == end_node else "1e6" for j in range(dimension)]
            f.write(" ".join(dummy_row) + "\n")
