import numpy as np

def fold_3d_array_to_2d_using_NaN_separator(_3d_array: np.ndarray) -> np.ndarray:
    _1d_arrays = []
    for array_2d in _3d_array:
        _1d_arrays.extend(array_2d.flatten())
        _1d_arrays.extend([np.nan, np.nan])
    _1d_arrays = _1d_arrays[:-2]
    result = np.array(_1d_arrays).reshape(-1, 2)
    return result

def geometrys_from_txt_nan_separeted(txt_path:str,sep=" ") -> np.ndarray:
    geometrys = []
    raw_geometrys  = np.fromfile(txt_path,sep=sep).reshape(-1,2)
    nans_idx = list(np.argwhere(np.isnan(raw_geometrys))[:,0])[::2]
    nans_len = len(nans_idx)
    if nans_len == 0 : return [raw_geometrys]
    ii = -1                 # Comeca em -1 para preservar logica do loop
    for jj in range(nans_len):
        geometrys.append(raw_geometrys[ii+1:nans_idx[jj]])
        ii = nans_idx[jj]
        if jj == nans_len - 1 :
            geometrys.append(raw_geometrys[ii+1:])
    return geometrys