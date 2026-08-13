import cv2
import matplotlib.pyplot as plt
import numpy as np

imagen = cv2.imread("./imagenes/imagen_prueba.jpg") ## LEER IMAGEN IMREAD

if imagen is None:
    print("Imagen no cargada verifique el codigo proporcionado")
    exit()

alpha = 1.2  # Contraste (mayor a 1 aumenta el contraste)
beta = 10    # Brillo (positivo aumenta el birllo)
ajuste_imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta) ## AJUSTE DE BRILLO Y CONTRASTE

ajuste_imagen = cv2.cvtColor(ajuste_imagen, cv2.COLOR_BGR2RGB) # CONVERTIR A RGB POR COMPATIBILIDAD

#MOSTRAMOS LA IMAGEN CON MATPLOTLIB
# plt.figure(figsize=(8, 6)) # ESTO ES PARA DEFINIR EL TAMAÑO DE LA IMAGEN SIENDO 8 DE ANCHO Y 6 DE ALTO
# plt.imshow(ajuste_imagen) # ESTO ES PARA MOSTRAR LA IMAGEN
# plt.title("Imagen Cargada") # ESTO ES PARA PONERLE UN TITULO A LA IMAGEN
# plt.axis("off") # ESTO ES PARA QUE NO SE VISUALICEN LOS EJES DE LA IMAGEN
# plt.show() # ESTO ES PARA MOSTRAR LA IMAGEN

ajuste_imagen = cv2.cvtColor(ajuste_imagen, cv2.COLOR_BGR2RGB) # CONVERTIR A BGR PARA GUARDAR LA IMAGEN AJUSTADA
cv2.imwrite("./imagenes/imagen_ajustada3.jpg", ajuste_imagen) ## GUARDAR IMAGEN AJUSTADA

hsv = cv2.cvtColor(ajuste_imagen, cv2.COLOR_BGR2HSV) # CONVERTIR A HSV

h, s,v = cv2.split(hsv) # SEPARAR CANALES HSV

saturacion = 5 # AUMENTAR SATURACION

s = cv2.multiply(s, saturacion) # AUMENTAR SATURACION
s = np.clip(s, 0, 255).astype(np.uint8) # LIMITAR VALORES ENTRE 0 Y 255
hsv_ajustado = cv2.merge([h, s, v])
color_corregido = cv2.cvtColor(hsv_ajustado, cv2.COLOR_HSV2BGR) # CONVERTIR A BGR PARA GUARDAR LA IMAGEN AJUSTADA



color_corregido = cv2.cvtColor(color_corregido, cv2.COLOR_BGR2RGB) # CONVERTIR A BGR PARA GUARDAR LA IMAGEN AJUSTADA
cv2.imwrite("./imagenes/imagen_ajustada_color.jpg", color_corregido)