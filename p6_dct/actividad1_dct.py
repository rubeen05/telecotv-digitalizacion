import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn
from p4_color.conversion import rgb_a_ycrcb
from p6_dct.dct import mi_dct

# Abrir las imágenes en RGB y obtener su luminancia con la función de P4.
def cargar_y(ruta):
    rgb = np.array(Image.open(ruta).convert('RGB'))
    # Conservar Yq; los dos guiones bajos reciben Crq y Cbq, que no se usan aquí.
    Yq, _, _ = rgb_a_ycrcb(rgb)
    # Convertir a float64 antes de llamar a mi_dct, sin cambiar los niveles de Yq.
    return Yq.astype(np.float64)

img = cargar_y('p4_color/datos/escena_referencia.png')
cap = cargar_y('p4_color/datos/escena_capturada.png')
h, w = cap.shape
# Cada recorte tiene 8 filas y 8 columnas. La captura usa sus propias coordenadas.
bloques = {
    'uniforme': img[0:8, 0:8],
    'detalle': img[704:712, 432:440],
    'captura': cap[h//2:h//2+8, w//2:w//2+8],
}

# Mostrar los tres recortes con la misma escala de luminancia.
fig_bloques, ejes = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
for eje, (nombre, x) in zip(ejes, bloques.items()):
    eje.imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    eje.set_title(nombre)
    eje.axis('off')
plt.show()


# Calcular ambas DCT para cada bloque y conservarlas para la figura final.
resultados = {}
print('bloque       DC       error_max_DCT')
# items permite recorrer el nombre y el contenido de cada bloque.
for nombre, x in bloques.items():
    X = mi_dct(x)   # Aplicar la función mi_dct al bloque x.
    # Usar la misma normalización que en nuestra matriz T.
    X_ref = dctn(x, norm='ortho')
    error = np.max(np.abs(X-X_ref))   # Mayor diferencia absoluta entre X y X_ref.
    # En comprobar_matriz.py ya se usaron np.abs y np.max.
    # np es el nombre abreviado de NumPy, importado al principio.
    # Restar las matrices, tomar valores absolutos y elegir el mayor.
    print(f'{nombre:9s} {X[0, 0]:9.3f} {error:.3e}')
    # Guardar juntas las dos matrices de coeficientes de este bloque.
    resultados[nombre] = (X, X_ref)


    # Cada fila muestra un bloque, su DCT propia y su DCT de SciPy.
fig, ax = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
# enumerate añade el número de fila para colocar cada bloque en la figura.
for i, (nombre, x) in enumerate(bloques.items()):
    X, X_ref = resultados[nombre]
    ax[i, 0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 0].set_title(nombre + ' - Yq')
    # Representar la magnitud de los coeficientes con la misma escala en ambas DCT.
    for j, (A, titulo) in enumerate([(X, 'mi_dct'), (X_ref, 'SciPy')], 1):
        im = ax[i, j].imshow(np.abs(A), cmap='jet', vmin=0, vmax=np.abs(X_ref).max())
        ax[i, j].set_title('|' + titulo + '|')
        fig.colorbar(im, ax=ax[i, j], shrink=0.7)
for a in ax.flat:
    a.axis('off')
plt.savefig('p6_dct/resultados/actividad1_dct.png', dpi=120)
plt.show()