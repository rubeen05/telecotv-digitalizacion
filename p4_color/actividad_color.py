from p4_color.conversion import rgb_a_ycrcb, ycrcb_a_rgb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img_barras = Image.open('p4_color/datos/barras_referencia.png').convert('RGB')
arr_barras = np.array(img_barras)   # uint8 por defecto — no añadir dtype=np.float64
Yq, Crq, Cbq = rgb_a_ycrcb(arr_barras)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(arr_barras); axes[0].set_title('RGB original')
axes[1].imshow(Yq,  cmap='gray', vmin=0, vmax=255); axes[1].set_title('Yq')
axes[2].imshow(Crq, cmap='gray', vmin=0, vmax=255); axes[2].set_title('Crq')
axes[3].imshow(Cbq, cmap='gray', vmin=0, vmax=255); axes[3].set_title('Cbq')
plt.tight_layout()
plt.savefig('p4_color/resultados/yq_crq_cbq_barras.png', dpi=100, bbox_inches='tight')


Yq_barra_blanca=int(Yq[500, 100])
Yq_barra_negra=int(Yq[500, 1900])
Crq_barra_blanca=int(Crq[500, 100])
Crq_barra_roja=int(Crq[500, 1400])

print('Yq_barra_blanca=', Yq_barra_blanca)
print('Yq_barra_negra=', Yq_barra_negra)
print('Crq_barra_blanca=', Crq_barra_blanca)
print('Crq_barra_roja=', Crq_barra_roja)

img_escena = Image.open('p4_color/datos/escena_capturada.png').convert('RGB')
arr_escena = np.array(img_escena)   # uint8 por defecto — no añadir dtype=np.float64
Yq, Crq, Cbq = rgb_a_ycrcb(arr_escena)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(arr_escena); axes[0].set_title('RGB original')
axes[1].imshow(Yq,  cmap='gray', vmin=0, vmax=255); axes[1].set_title('Yq')
axes[2].imshow(Crq, cmap='gray', vmin=0, vmax=255); axes[2].set_title('Crq')
axes[3].imshow(Cbq, cmap='gray', vmin=0, vmax=255); axes[3].set_title('Cbq')
plt.tight_layout()
plt.savefig('p4_color/resultados/yq_crq_cbq_escena.png', dpi=100, bbox_inches='tight')

# Conversión directa e inversa
Yq, Crq, Cbq = rgb_a_ycrcb(arr_barras.astype(np.uint8))
img_rec = ycrcb_a_rgb(Yq, Crq, Cbq)

# Error real de cada píxel y canal
diff = arr_barras.astype(np.float64) - img_rec.astype(np.float64)

rmse = np.sqrt(np.mean(
    (arr_barras.astype(np.float64) - img_rec.astype(np.float64)) ** 2
))
print(f'RMSE barras de referencia: {rmse:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# Ampliar la diferencia solo para visualizarla
dif_visible = np.clip(128 + 20 * diff, 0, 255).astype(np.uint8)
for ax, imagen, titulo in zip(axes,
        [arr_barras, img_rec, dif_visible],
        ["Original", "Reconstruida", "Diferencia x20; gris 128 = error cero"]):
    ax.imshow(imagen)
    ax.set_title(titulo, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("p4_color/resultados/inversa_barras.png", dpi=120)

# Conversión directa e inversa
escena = np.array(Image.open("p4_color/datos/escena_capturada.png").convert("RGB"))
Yesc, Cresc, Cbesc = rgb_a_ycrcb(escena)
escena_rec = ycrcb_a_rgb(Yesc, Cresc, Cbesc)
rmse_escena = np.sqrt(np.mean((escena.astype(float) - escena_rec.astype(float)) ** 2))
print(f"RMSE de conversión de la escena: {rmse_escena:.4f}")

# Error real de cada píxel y canal
diffesc = escena.astype(np.float64) - escena_rec.astype(np.float64)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# Ampliar la diferencia solo para visualizarla
dif_visible_esc = np.clip(128 + 20 * diffesc, 0, 255).astype(np.uint8)
for ax, imagen, titulo in zip(
    axes,
    [escena, escena_rec, dif_visible_esc],
    ["Original", "Reconstruida", "Diferencia x20; gris 128 = error cero"]):
    ax.imshow(imagen)
    ax.set_title(titulo, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("p4_color/resultados/inversa_escena.png", dpi=120)

