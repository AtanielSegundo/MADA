from commons.utils.clipper import offsetPaths, offsetTXT
from core.geometry import  nthgone
from core.visualize import ShowGeometrys
import os
import numpy as np


def create_two_semi_circles(ray: float, separation: float, c_points=1000):
    semi_circle_1 = nthgone(c_points, ray=ray)[(c_points):]
    semi_circle_2 = nthgone(c_points, ray=ray)[0:(c_points)]
    semi_circle_1 = np.array([[semi_circle_1[idx % len(semi_circle_1)]] for idx in range(len(semi_circle_1)+1)])
    semi_circle_2 = np.array([[semi_circle_2[idx % len(semi_circle_2)]] for idx in range(len(semi_circle_2)+1)])
    return [semi_circle_1, semi_circle_2]


if __name__ == "__main__":
    ITERACOES = 40
    OFFSET = -2
    PATH = "assets/txt/formas"
    PRECISAO = 1E3  # PRECISAO DE 3 DIGITOS
    for arquivo in os.listdir(PATH):
        print(f"Lendo {arquivo}")
        offsetTXT(os.path.join(PATH, arquivo),
                          ITERACOES, OFFSET, precisao=PRECISAO)
    # two_semi_circles_nthgone = create_two_semi_circles(50,25)
    # ShowGeometrys([two_semi_circles_nthgone,offsetPaths(two_semi_circles_nthgone.copy(),distance=OFFSET,iter=ITERACOES)],spliter=2)
