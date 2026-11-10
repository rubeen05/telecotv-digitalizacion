import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn, idctn
from p4_color.conversion import rgb_a_ycrcb
from p5_muestreo.muestreo import rmse

def magnitud_db(X):
    # Obtener la magnitud de cada coeficiente con np.abs.
    Xa = np.abs(X)
    # Obtener la mayor magnitud de Xa con .max().
    maximo = np.max(Xa)
    # Si todos son cero, devolver una matriz de -inf del mismo tamaño.
    if maximo == 0:
        return np.full(X.shape, -np.inf)
    # Conservar log10(0) = -inf sin mostrar el aviso de NumPy.
    with np.errstate(divide='ignore'):
        # Expresar Xa en dB respecto a maximo, usando np.log10.
        XndB = 20*np.log10(Xa/maximo)
    return XndB

# Abrir las imágenes originales, sin modificar sus archivos.
rgb_referencia = np.array(Image.open('p4_color/datos/escena_referencia.png').convert('RGB'))
rgb_captura = np.array(Image.open('p4_color/datos/escena_capturada.png').convert('RGB'))
h, w = rgb_captura.shape[:2]
print(f'Captura original: {h} filas y {w} columnas')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(rgb_captura)
ax.set_title('Captura original: localizar el área de imagen sin márgenes negros')
ax.set_xlabel('Columna (x)')
ax.set_ylabel('Fila (y)')
plt.show()



# Escribir los límites del área útil observados en la figura anterior.
fila_inicial, fila_final = 200, 400
columna_inicial, columna_final = 200, 500
# Comprobar que el recorte está dentro de la captura y contiene píxeles.
if not (0 <= fila_inicial < fila_final <= h and
        0 <= columna_inicial < columna_final <= w):
    raise ValueError('Revisar los límites del área útil de la captura')

# Recortar la captura en memoria, sin redimensionarla.
captura_util = rgb_captura[fila_inicial:fila_final, columna_inicial:columna_final]
print(f'Área útil: filas {fila_inicial}:{fila_final}, columnas {columna_inicial}:{columna_final}')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(captura_util)
ax.set_title('Captura recortada: comprobar que se conserva todo el contenido de la escena')
plt.show()

# Conservar la referencia completa y la captura recortada para todos los ensayos.
imagenes = {}
for nombre, rgb in [('referencia', rgb_referencia), ('captura', captura_util)]:
    Yq, _, _ = rgb_a_ycrcb(rgb)
    imagenes[nombre] = Yq.astype(np.float64)
    img = imagenes[nombre]
    # Calcular la DCT de toda el área que se va a analizar.
    X = dctn(img, norm='ortho')
    rec = idctn(X, norm='ortho')
    print(f'{nombre}: Error máximo de reconstrucción sin descartar coeficientes: {np.max(np.abs(img-rec)):.3e}')



    # Recuperar el fotograma de las varillas utilizado en P5.
rgb_varillas = np.array(Image.open('p5_muestreo/datos/escena_varillas.png').convert('RGB'))
Y_varillas, _, _ = rgb_a_ycrcb(rgb_varillas)
# Reunir las tres imágenes para representarlas con el mismo procedimiento.
comparacion = {
    'camino': imagenes['referencia'],
    'varillas': Y_varillas.astype(np.float64),
    'captura propia': imagenes['captura'],
}

fig, ax = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
for i, (nombre, img) in enumerate(comparacion.items()):
    X = dctn(img, norm='ortho')
    XndB = magnitud_db(X)
    ax[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[i, 0].set_title(nombre + ' - Yq')
    # Mostrar tanto el espectro completo como su esquina de bajas frecuencias.
    for j, A in [(1, XndB), (2, XndB[:64, :64])]:
        # El límite -120 dB afecta solo al dibujo; XndB conserva sus valores originales.
        im = ax[i, j].imshow(np.maximum(A, -120), cmap=plt.get_cmap('jet', 64),
                             vmin=-120, vmax=0, interpolation='nearest', aspect='auto')
        ax[i, j].set_title('DCT completa' if j == 1 else 'DCT: indices [0:64,0:64]')
        ax[i, j].set_xlabel('Frecuencia horizontal (v)')
        ax[i, j].set_ylabel('Frecuencia vertical (u)')
        fig.colorbar(im, ax=ax[i, j], label='dB respecto al máximo')
plt.savefig('p6_dct/resultados/espectros.png', dpi=120)
plt.show()



# Comparar con la referencia completa o con la captura recortada, sin cambiar su tamaño.
for nombre, img in imagenes.items():
    X = dctn(img, norm='ortho')
    XndB = magnitud_db(X)
    h, w = img.shape
    # Mismo recorte relativo en los cuatro umbrales de esta imagen
    r, c = h//2, w//2
    # slice define los intervalos del zoom; min evita que sobrepasen el borde de la imagen.
    zona = (slice(r, min(r+128, h)), slice(c, min(c+128, w)))
    fig, ax = plt.subplots(4, 3, figsize=(12, 12), layout='constrained')
    print(f'\nImagen: {nombre}')
    print(f"{'Umbral (dB)':>12} | {'Coef. no nulos (%)':>20} | {'RMSE (niveles Yq)':>20}")
    for i, umbral in enumerate([-120, -90, -60, -30]):
        # Trabajar sobre una copia: el ensayo siguiente debe partir otra vez de la DCT completa.
        Xu = np.copy(X)   # Crear una copia de X con .copy().
        # La comparación selecciona las posiciones cuya magnitud queda por debajo del umbral.
        Xu[XndB < umbral] = 0   # Sustituir por cero los coeficientes seleccionados.
        rec = idctn(Xu, norm='ortho')
        # Contar valores realmente no nulos y dividir entre el número total de coeficientes.
        pct = 100*(np.count_nonzero(Xu)/Xu.size)   # Calcular el porcentaje con np.count_nonzero y Xu.size.
        error = rmse(img, rec)  # antes de clipping; toda la imagen
        print(f'{umbral:12d} | {pct:20.4f} | {error:20.3f}')
        # Sumar 128 permite mostrar diferencias negativas; solo esta visualización se limita a 0-255.
        dif = np.clip(img - rec + 128, 0, 255)
        for j, (A, titulo) in enumerate([(img, 'Original'), (rec, 'Reconstruida'),
                                         (dif, 'Diferencia +128')]):
            ax[i, j].imshow(np.clip(A, 0, 255), cmap='gray', vmin=0, vmax=255)
            ax[i, j].set_title(f'{umbral} dB - {titulo}')
            ax[i, j].axis('off')
            # Añadir un pequeño panel de ampliación dentro de cada imagen.
            zoom = ax[i, j].inset_axes([0.53, 0.52, 0.46, 0.46])
            zoom.imshow(np.clip(A[zona], 0, 255), cmap='gray', vmin=0, vmax=255,
                        interpolation='nearest')
            zoom.set_xticks([]); zoom.set_yticks([])
            for borde in zoom.spines.values():
                borde.set_color('#f59e0b')
    fig.suptitle(nombre + f' - zoom: filas {r}:{min(r+128,h)}, columnas {c}:{min(c+128,w)}')
    plt.savefig(f'p6_dct/resultados/umbrales_{nombre}.png', dpi=150)
    plt.show()



    # Comprobar el procedimiento con dos entradas de resultado conocido.
controles = {
    'constante': np.full((8, 8), 128.0, dtype=np.float64),
    'todos los píxeles a cero': np.zeros((8, 8), dtype=np.float64),
}
for nombre, img_control in controles.items():
    X_control = dctn(img_control, norm='ortho')
    db_control = magnitud_db(X_control)
    print(f'\nControl: {nombre}')
    # isneginf detecta -inf; all comprueba si aparece en todas las posiciones.
    print('Todas las magnitudes son -inf:', np.all(np.isneginf(db_control)))
    print(f"{'Umbral (dB)':>12} | {'Coef. no nulos (%)':>20} | {'RMSE (niveles Yq)':>20}")
    for umbral in [-120, -90, -60, -30]:
        # Repetir los mismos pasos que en las imágenes de referencia y captura.
        Xu = X_control.copy()
        Xu[db_control < umbral] = 0
        rec_control = idctn(Xu, norm='ortho')
        pct = 100.0 * np.count_nonzero(Xu) / Xu.size
        error = rmse(img_control, rec_control)
        print(f'{umbral:12d} | {pct:20.4f} | {error:20.3e}')


    