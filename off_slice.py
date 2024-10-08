from core.slicing import sliceStlVector
from core.visualize import showStl
from commons.clipperutils import offsetPaths
from sys import argv
from os.path import basename

PETRO_FOICE_PATH = "assets/3d/Petro_foice.stl"
TRUSS_PATH = "assets/3d/truss_.stl"

stl_paths = [PETRO_FOICE_PATH,TRUSS_PATH]
path_to_n = lambda path : basename(path).lower()

def flag_args_evaluator(argv) :
    args_joined = "".join(argv[1:]) if len(argv) > 1 else None 
    def _eval(s:str,_default:bool=True) :
        if not args_joined    : return _default
        elif s in args_joined : return True
        else : return False 
    return _eval
    
if __name__ == "__main__" :    
    f_eval = flag_args_evaluator(argv)
    if f_eval('show') :
        if f_eval("petro") : showStl(PETRO_FOICE_PATH)
        if f_eval("truss") : showStl(TRUSS_PATH)
    
    n_slices = 2
    iterations = 50
    off_dist = None
    
    for n_sep in range(200,-1,-1) :
        if f_eval(f'i{n_sep}',False) : 
            iterations = n_sep
            break
            
    for m_sep in range(200,-1,-1) :
        if f_eval(f'o{m_sep}',False) : 
            off_dist = m_sep
            break
    
    if off_dist is None : petro_offseter = truss_offseter = None
    else : 
        petro_offseter = lambda gArr : offsetPaths(gArr,-off_dist,iterations)        
        truss_offseter = lambda gArr : offsetPaths(gArr,-off_dist,iterations)
            
    for n_sep in range(200,-1,-1) :
        if f_eval(f's{n_sep}') :
            n_slices = n_sep
            break

    if f_eval("3d") :
        if f_eval("truss") :
            sliceStlVector(TRUSS_PATH,n_slices=n_slices,z_step=1,
                           scaleFactor=1,offset_fn=truss_offseter)
        if f_eval("petro") :
            sliceStlVector(PETRO_FOICE_PATH,n_slices=n_slices,z_step=1,
                           scaleFactor=2,offset_fn=petro_offseter)
    if f_eval("2d") :
        if f_eval("truss") :
            sliceStlVector(TRUSS_PATH,n_slices=n_slices,z_step=1,
                           scaleFactor=1,offset_fn=truss_offseter,d2_mode=True)
        if f_eval("petro") :
            sliceStlVector(PETRO_FOICE_PATH,n_slices=n_slices,z_step=1,
                           scaleFactor=2,offset_fn=petro_offseter,d2_mode=True)