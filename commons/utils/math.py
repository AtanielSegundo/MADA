import numpy as np


def online_mean(new_mean: np.ndarray, current_mean: np.ndarray, count):
    return (current_mean+(new_mean-current_mean)/(count+1), (count+1))
