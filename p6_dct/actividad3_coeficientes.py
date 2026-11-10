import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from p4_color.conversion import rgb_a_ycrcb
from p6_dct.dct import mi_dct, mi_idct
from p5_muestreo.muestreo import rmse

rgb = np.array(Image.open('p4_color/datos/escena_referencia.png').convert('RGB'))
Yq, _, _ = rgb_a_ycrcb(rgb)
img = Yq.astype(np.float64)
x = img[704:712, 432:440]  # Seleccionar las filas y columnas del bloque señalado.
X = mi_dct(x)
# Estos ajustes solo cambian cómo se imprime la matriz; no redondean sus valores almacenados.
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print('BLOQUE CON DETALLE - coeficientes DCT con signo:')
print(X)

# Ordenar por valor absoluto y seleccionar las cuatro posiciones mayores.
idx = np.argsort(np.abs(X), axis=None)[::-1][:4]
# Obtener la fila y la columna de cada posición seleccionada.
filas, columnas = np.unravel_index(idx, X.shape)
# Recorrer juntos los índices de fila y columna.
print('\nBLOQUE CON DETALLE - cuatro coeficientes de mayor valor absoluto:')
for f, c in zip(filas, columnas):
    print(f'X[{f},{c}] = {X[f,c]:9.3f}   posición contando desde 1: ({f+1},{c+1})')


    # Reconstruir el bloque casi uniforme conservando solo DC.
plano = img[0:8, 0:8]
Xp = mi_dct(plano)   # Calcular la DCT de plano con mi_dct.
solo_dc = np.zeros_like(Xp)
solo_dc[0, 0] = Xp[0, 0]   # Copiar el coeficiente DC de Xp.
print('\nBLOQUE CASI UNIFORME - esquina superior izquierda del camino')
print(f'Reconstrucción solo con DC: RMSE={rmse(plano, mi_idct(solo_dc)):.3f} niveles Yq')

# Escribir las seis posiciones de la tabla con índices de Python.
posiciones = [
    (0, 0),  # Primer coeficiente.
    (0, 1),  # Segundo coeficiente.
    (1, 0),  # Tercer coeficiente.
    (3, 0),  # Cuarto coeficiente.
    (6, 5),  # Quinto coeficiente.
    (1, 7),  # Sexto coeficiente.
]

# Repetir con 1, 2, ..., 6 coeficientes de la lista, acumulándolos en ese orden.
print('\nBLOQUE CON DETALLE - bloque señalado en la imagen')
print('Reconstrucciones con las posiciones de la tabla. Errores en niveles Yq:')
for k in range(1, len(posiciones) + 1):
    # Empezar de nuevo en cada ensayo: misma forma y tipo que X, pero todos los valores a cero.
    Xk = np.zeros_like(X)
    # posiciones[:k] selecciona las primeras k posiciones de la lista.
    for f, c in posiciones[:k]:
        Xk[f, c] = X[f, c]   # conservar el coeficiente de X
    # La IDCT reconstruye el bloque usando solo los coeficientes que se han copiado.
    xk = mi_idct(Xk)   # Reconstruir el bloque con mi_idct a partir de Xk.
    print(f'{k} coef.  maxdif={np.max(np.abs(x - xk)):7.3f}  RMSE={rmse(x, xk):7.3f}')


    # Conservar los cuatro coeficientes de mayor valor absoluto.
X4 = np.zeros_like(X)
# filas y columnas contienen los cuatro pares obtenidos con argsort y unravel_index.
X4[filas, columnas] = X[filas, columnas]   # conservar solo los cuatro seleccionados
x4 = mi_idct(X4)   # Reconstruir el bloque con mi_idct a partir de X4.
print('\nBLOQUE CON DETALLE - selección de los cuatro mayores en valor absoluto')
print(f'4 mayores  maxdif={np.max(np.abs(x - x4)):7.3f}  RMSE={rmse(x, x4):7.3f}')

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
# ravel permite recorrer los ocho paneles con un solo índice, de ax[0] a ax[7].
ax = ax.ravel()
ax[0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[0].set_title('Original (64 coef.)')
for i, n in enumerate(range(1, 7)):
    Xk = np.zeros_like(X)
    for f, c in posiciones[:n]:
        Xk[f, c] = X[f, c]
    ax[i+1].imshow(mi_idct(Xk), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i+1].set_title(f'{n} coeficiente(s)')
ax[7].imshow(x4, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[7].set_title('4 mayores')
for a in ax: a.axis('off')
plt.tight_layout()
plt.savefig('p6_dct/resultados/actividad3_coeficientes.png', dpi=100, bbox_inches='tight')
plt.show()