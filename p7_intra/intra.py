import numpy as np
from p6_dct.dct import matriz_dct
from p7_intra.parametros import PESOS, ORDEN
from p7_intra.apoyo import extender, iter_bloques

T = matriz_dct(8)

def redondear(x):
    # Los valores a mitad entre dos enteros se redondean alejándose de cero.
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def validar(X, FC, pesos):
    if not isinstance(X, np.ndarray) or X.shape != (8, 8) or X.dtype != np.float64:
        raise ValueError('Se necesita un bloque 8x8 float64')
    if not np.isfinite(X).all():
        raise ValueError('El bloque debe contener valores finitos')
    if not isinstance(FC, (int, np.integer)) or isinstance(FC, (bool, np.bool_)) or not 1 <= FC <= 112:
        raise ValueError('FC debe ser entero entre 1 y 112')
    if pesos.shape != (8, 8) or not np.isfinite(pesos).all() or np.any(pesos <= 0):
        raise ValueError('Los pesos deben ser positivos y finitos, forma 8x8')

def cuantificar(X, FC, pesos=PESOS):
    validar(X, FC, pesos)
    indices = redondear(16 * X / (FC * pesos))   # Aplicar la expresión de cuantificación AC.
    indices[0, 0] = X[0, 0]        # Conservar DC sin cuantificar.
    return indices



def zigzag(indices):
    # ORDEN es la permutación del material original, adaptada a NumPy.
    return indices.ravel()[ORDEN]

def inversa_zigzag(vector):
    if np.asarray(vector).shape != (64,):
        raise ValueError('El vector debe tener 64 elementos')
    matriz = np.empty(64, dtype=np.float64)
    matriz[ORDEN] = vector
    return matriz.reshape(8, 8)

def ceros_finales(vector):
    if np.asarray(vector).shape != (64,):
        raise ValueError('El vector debe tener 64 elementos')
    # Obtener posiciones de los valores distintos de cero, no sus valores.
    no_nulos = np.flatnonzero(vector != 0)
    if no_nulos.size == 0:
        return vector.size    # Devolver el número de elementos: todos son cero.
    ultima_posicion = no_nulos[-1]
    return vector.size-1-ultima_posicion        # Contar las posiciones que quedan hasta el final del vector.



def cuantificacion_inversa(indices, FC, pesos=PESOS):
    validar(indices, FC, pesos)
    Xr = redondear(indices * FC * pesos / 16)    # Deshacer el escalado AC con los mismos pesos y FC.
    Xr[0, 0] = indices[0, 0]         # Restaurar el DC conservado sin redondearlo.
    return Xr



def cod_intra(imagen, FC, pesos=PESOS, desplazamiento=128):
    extendida, forma = extender(imagen)
    indices = np.empty_like(extendida)
    for fila, columna, bloque in iter_bloques(extendida):
        # T es la matriz de P6, calculada una vez al importar este módulo.
        X = T @ (bloque - desplazamiento) @ T.T    # Aplicar la DCT matricial a (bloque - desplazamiento).
        # Cuantificar X con FC y pesos; guardar el resultado en la posición de este bloque.
        indices[fila:fila+8, columna:columna+8] = cuantificar(X, FC, pesos)
    return indices, forma

def decod_intra(indices, forma, FC, pesos=PESOS, desplazamiento=128):
    recuperada = np.empty_like(indices)
    for fila, columna, bloque in iter_bloques(indices):
        Xr = cuantificacion_inversa(bloque, FC, pesos)
        recuperada[fila:fila+8, columna:columna+8] = T.T @ Xr @ T   # Aplicar a Xr la IDCT matricial con T.T y T.
    h, w = forma
    return recuperada[:h, :w] + desplazamiento



