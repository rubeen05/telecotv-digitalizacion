import numpy as np        # arrays y operaciones matriciales
from PIL import Image     # leer ficheros de imagen (Pillow)
import matplotlib.pyplot as plt  # visualización
from pathlib import Path

Path("p4_color/resultados").mkdir(parents=True, exist_ok=True)
# Cargar la imagen con Pillow
img_barras = Image.open('p4_color/datos/barras_referencia.png').convert('RGB')

# Convertir a array numpy con tipo float64
arr_barras = np.array(img_barras, dtype=np.float64)

# Verificar: debe mostrar (1080, 1920, 3)
print('Forma del array:', arr_barras.shape)
R = arr_barras[:, :, 0]
G = arr_barras[:, :, 1]
B = arr_barras[:, :, 2]

print("Dimensiones de R:", R.shape)
print("Dimensiones de G:", G.shape)
print("Dimensiones de B:", B.shape)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 fila, 3 columnas; tamaño en pulgadas

# Subplot 0 — modelo; G y B siguen exactamente el mismo patrón
# cmap='gray': muestra el valor numérico en escala de grises
# vmin=0, vmax=255: fija el rango dinámico completo — sin esto matplotlib autoescalaría
# cada canal por separado y los valores no serían comparables entre sí
axes[0].imshow(R, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Canal R')

# Subplot 0 — modelo; G y B siguen exactamente el mismo patrón
# cmap='gray': muestra el valor numérico en escala de grises
# vmin=0, vmax=255: fija el rango dinámico completo — sin esto matplotlib autoescalaría
# cada canal por separado y los valores no serían comparables entre sí
axes[1].imshow(G, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Canal G')

axes[2].imshow(B, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Canal B')
# Completar subplots 1 y 2 para los canales G y B
# En Python, cada subplot se gestiona como un índice del array 'axes'

plt.tight_layout()  # ajusta los márgenes para que los títulos no se solapen
plt.savefig('p4_color/resultados/canales_barras.png', dpi=100, bbox_inches='tight')
# ^ dpi=100: resolución de la imagen · bbox_inches='tight': recorta el margen blanco exterior
#plt.show()  # muestra la figura en pantalla y la cierra (llamar siempre al final)

# Paso 04 con el fotograma capturado
img_escena = Image.open('p4_color/datos/escena_capturada.png').convert('RGB')
arr_escena = np.array(img_escena, dtype=np.float64)

R_e = arr_escena[:, :, 0]
G_e = arr_escena[:, :, 1]
B_e = arr_escena[:, :, 2]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(R_e, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Canal R')
axes[1].imshow(G_e, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Canal G')
axes[2].imshow(B_e, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Canal B')

plt.tight_layout()
plt.savefig('p4_color/resultados/canales_escena.png', dpi=100, bbox_inches='tight')

# Comenzar con los canales en su orden original
nueva_imagen = np.stack([B, G, R], axis=2)
nueva_imagen_uint8 = nueva_imagen.astype(np.uint8)  # imshow necesita uint8 para colores

plt.figure()
plt.imshow(nueva_imagen_uint8)
plt.title('Orden de los canales: B, G, R')
plt.savefig('p4_color/resultados/canales_reordenados.png', dpi=100, bbox_inches='tight')
plt.show()
