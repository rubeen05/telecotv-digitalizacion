import numpy as np

def rgb_a_ycrcb(imagen_rgb):
    # Entrada: array numpy uint8, forma (alto, ancho, 3)
    # Salida: tres arrays uint8 — Yq, Crq, Cbq

    img = imagen_rgb.astype(np.float64)   # evitar desbordamientos en uint8

    # Separar los canales RGB como en la tarea anterior
    # Normalizar cada canal al rango [0, 1] dividiendo entre 255.0
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    R=R/255
    G=G/255
    B=B/255

        # Luminancia según BT.601 SDTV — resultado en [0, 1]
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    # Diferencias de color: cuánto se aleja R y B de la luminancia
    R_Y = R-Y
    B_Y = B-Y

    # Escalar para obtener Cr y Cb en el rango [-0.5, +0.5]
    Cr = 0.713 * R_Y
    Cb = 0.564 * B_Y

    Yq  = np.clip(16 + np.round(219*Y), 0, 255).astype(np.uint8)
    Crq = np.clip(128 + np.round(224*Cr), 0, 255).astype(np.uint8)
    Cbq = np.clip(128 + np.round(224*Cb), 0, 255).astype(np.uint8)
    return Yq, Crq, Cbq

def ycrcb_a_rgb(Yq, Crq, Cbq):
    # Entrada: tres arrays uint8 — Yq, Crq, Cbq
    # Salida: array numpy uint8, forma (alto, ancho, 3)

    Y  = (Yq.astype(np.float64)  - 16) / 219
    Cr = (Crq.astype(np.float64) - 128) / 224
    Cb = (Cbq.astype(np.float64) - 128) / 224

        # Deshacer el escalado de crominancia → recuperar diferencias de color
    R_Y = Cr / 0.713
    B_Y = Cb / 0.564

    # Reconstruir R, B y despejar G de la ecuación de luminancia
    R = R_Y+Y
    B = B_Y+Y
    G = (Y-0.299*R-0.114*B)/0.587

    R_u8 = np.clip(np.round(R * 255), 0, 255).astype(np.uint8)
    G_u8 = np.clip(np.round(G * 255), 0, 255).astype(np.uint8)
    B_u8 = np.clip(np.round(B * 255), 0, 255).astype(np.uint8)
    return np.stack([R_u8, G_u8, B_u8], axis=2)
