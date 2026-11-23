import numpy as np
from p6_dct.dct import mi_dct, mi_idct
from p5_muestreo.muestreo import rmse
from p7_intra.intra import cuantificar, cuantificacion_inversa
from p7_intra.parametros import FACTORES
from p7_intra.apoyo import cargar_y, seleccion, guardar_tabla, figura_reconstrucciones

bloques = seleccion(cargar_y('p4_color/datos/escena_referencia.png'))
filas = []
for nombre, b in bloques.items():
    X = mi_dct(b - 128)
    reconstrucciones = []
    print(f'\n{nombre}:')
    print(f'{"FC":>3s} {"max |X-Xr|":>14s} {"RMSE espacial":>16s}')
    for FC in FACTORES:
        indices = cuantificar(X, FC)
        Xr = cuantificacion_inversa(indices, FC)
        rec = mi_idct(Xr) + 128            # Aplicar mi_idct a Xr y recuperar los niveles.
        error_max = abs(np.max(b)-np.max(rec))             # Máxima diferencia absoluta de coeficientes.
        error_rmse = rmse(b, rec)            # Llamar a rmse con el bloque original b y su reconstrucción rec.
        assert Xr[0, 0] == X[0, 0]
        reconstrucciones.append(rec)
        filas.append((nombre, FC, error_max, error_rmse))
        print(f'{FC:3d} {error_max:14.4f} {error_rmse:16.4f}')
    figura_reconstrucciones(nombre, b, reconstrucciones)
guardar_tabla('error_bloques', ['bloque', 'FC', 'error_max_DCT', 'RMSE'], filas)
# DC puede ser fraccionario. No convertir la matriz completa a entero.
b = np.full((8, 8), 150.125, dtype=np.float64)
X = mi_dct(b - 128)
for FC in FACTORES:
    assert cuantificacion_inversa(cuantificar(X, FC), FC)[0, 0] == X[0, 0]
print('DC fraccionario preservado')



