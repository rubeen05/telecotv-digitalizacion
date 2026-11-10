from p6_dct.dct import matriz_dct
import numpy as np

T = matriz_dct(8)
# np.eye(8) crea la identidad 8x8: unos en la diagonal y ceros en el resto.
# Si T es ortogonal, T @ T.T coincide con esa identidad.
print(np.max(np.abs(T @ T.T - np.eye(8))))   # del orden de 1e-15