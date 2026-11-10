import numpy as np
from scipy.fft import dctn, idctn
def matriz_dct(N=8):
    # fila u = patron de frecuencia u; columna v = posicion espacial
    # Reservar una matriz NxN; el bucle calculará cada uno de sus valores.
    T = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            # La primera fila corresponde a DC y tiene el mismo peso en todas las posiciones.
            if u == 0:
                T[u, v] = np.sqrt(1/N)              # peso w(0)
            else:
                # Los índices Python empiezan en 0: (2v-1)(u-1) pasa a (2v+1)u.
                T[u, v] = np.sqrt(2/N) * np.cos(np.pi * ((2*v + 1) * u / (2*N)))
    return T

def mi_dct(x):
    # isinstance comprueba que x sea un array de NumPy.
    # dtype indica el tipo de sus valores; aquí se exige float64.
    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        # raise detiene la llamada y muestra el motivo del error.
        raise TypeError('mi_dct: la entrada debe ser un array float64')

    # shape contiene (filas, columnas): solo se admiten bloques de 8x8.
    if x.shape != (8, 8):
        raise ValueError('mi_dct: la entrada debe ser un bloque 8x8')

    # Construir la matriz de la DCT para este tamaño de bloque.
    T = matriz_dct(8)
    # Completar el producto que transforma los píxeles en coeficientes.
    X = T @ x @ T.T
    return X   # producto matricial de la DCT 2D


def mi_idct(X):
    # X debe contener los coeficientes en un array NumPy de tipo float64.
    # Si el tipo no es correcto, raise detiene la llamada con un mensaje.
    if not isinstance(X, np.ndarray) or X.dtype != np.float64:
        raise TypeError('mi_idct: la entrada debe ser un array float64')

    # Comprobar que hay 8 filas y 8 columnas, como en la DCT directa.
    if X.shape != (8, 8):
        raise ValueError('mi_idct: la entrada debe ser un bloque 8x8')

    T = matriz_dct(8)
    # Completar el producto inverso para recuperar los valores del bloque.
    x_rec=T.T @ X @ T
    return x_rec   # producto matricial de la transformada inversa