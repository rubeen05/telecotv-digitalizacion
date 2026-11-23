import numpy as np
from p6_dct.dct import matriz_dct, mi_dct, mi_idct
from p7_intra.apoyo import cargar_y, dct_imagen, seleccion, figura_dct, figura_bloques

imagen = cargar_y('p4_color/datos/escena_referencia.png')
T = matriz_dct(8)   # Se construye una vez; es la misma matriz verificada en P6.
D_sin = dct_imagen(imagen, T)
D_centrada = dct_imagen(imagen-128, T)   # Centrar los niveles antes de transformar.
print(f'Yq observado: {imagen.min():.0f} a {imagen.max():.0f}')
print(f'Sin centrar: mínimo {D_sin.min():.4f}; máximo {D_sin.max():.4f}')
print(f'Centrada:    mínimo {D_centrada.min():.4f}; máximo {D_centrada.max():.4f}')
figura_dct(imagen, D_sin, D_centrada)



# Extraer los cuatro bloques de 8x8 elegidos en la imagen de referencia.
bloques = seleccion(imagen)
coeficientes = {}    # Guardar la DCT de cada bloque para la figura final.
for nombre, bloque in bloques.items():
    X = mi_dct(bloque-128)    # Centrar los niveles del bloque antes de aplicar la DCT.
    coeficientes[nombre] = X
    rec = mi_idct(X) + 128    # Aplicar la IDCT a X y recuperar los niveles originales.

    # Comparar los coeficientes obtenidos con y sin centrado.
    diferencia = X - mi_dct(bloque)
    ac = diferencia.copy()
    ac[0, 0] = 0    # Excluir DC para medir el cambio solo en las posiciones AC.

    # Mostrar el cambio en DC, el mayor cambio en AC y el error máximo en píxeles.
    print(f'{nombre:18s} cambio_DC={diferencia[0,0]:.4f} '
          f'cambio_max_AC={np.abs(ac).max():.3e} '
          f'error_recon={np.abs(bloque-rec).max():.3e}')
figura_bloques(imagen, bloques, coeficientes)