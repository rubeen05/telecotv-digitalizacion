import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from p4_color.conversion import rgb_a_ycrcb
from p6_dct.dct import mi_dct, mi_idct

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

# Reconstruir sin descartar coeficientes y comprobar la energía.
fig, ax = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
print('bloque    error_max_IDCT   energia_x   energia_X')
for i, (nombre, x) in enumerate(bloques.items()):
    X = mi_dct(x)   # Calcular la DCT del bloque x.
    # Se utilizan todos los coeficientes: aquí no se ha descartado información.
    rec = mi_idct(X)   # Reconstruir el bloque a partir de X (DCT inversa).
    error = np.max(np.abs(x-rec))   # Mayor diferencia absoluta entre x y rec.
    # Usar np.abs y np.max, como en la comparación de las dos DCT.
    # La suma de cuadrados mide la energía; debe coincidir antes y después de la DCT.
    print(f'{nombre:9s} {error:.3e} {np.sum(x*x):.6f} {np.sum(X*X):.6f}')
    ax[i, 0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 0].set_title(nombre + ' - original')
    ax[i, 1].imshow(rec, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 1].set_title('IDCT sin descarte')
    # La barra de color permite distinguir un residuo de redondeo de un error apreciable.
    im = ax[i, 2].imshow(np.abs(x-rec), cmap='jet', vmin=0)
    ax[i, 2].set_title('|diferencia| (niveles)')
    fig.colorbar(im, ax=ax[i, 2], shrink=0.7)
for a in ax.flat:
    a.axis('off')
plt.savefig('p6_dct/resultados/actividad2_idct.png', dpi=120)
plt.show()



# 4. Cada funcion debe rechazar las dos entradas
for funcion in [mi_dct, mi_idct]:
    # La primera entrada tiene un tipo incorrecto; la segunda, un tamaño incorrecto.
    for prueba in [np.zeros((8, 8), dtype=np.uint8), np.zeros((4, 4))]:
        # try permite comprobar si la función rechaza la entrada sin detener todo el script.
        try:
            funcion(prueba)
            print('FALLO:', funcion.__name__, 'no rechazo la entrada')
        # except recoge el error esperado; e contiene el mensaje que lo explica.
        except (TypeError, ValueError) as e:
            print('validacion OK ->', e)