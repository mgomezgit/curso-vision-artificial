import cv2
import matplotlib.pyplot as plt
import numpy as np

imagen = cv2.imread("./imagenes/imagen_prueba.jpg") ## LEER IMAGEN IMREAD

if imagen is None:
    print("Imagen no cargada verifique el codigo proporcionado")
    exit()

# Dibujar una línea
# EL PRIMER PARAMETRO ES DE PUNTO A QUE PUNTO EN X QUIERO QUE SE EXTENDA EL SEGUNDO ES EN Y, EL TERCERO ES EL COLOR Y EL CUARTO ES EL GROSOR DE LA LINEA
cv2.line(imagen, (50, 50), (200, 50), (255, 0, 0), 3)  # Azul en BGR

# Dibujar un rectángulo: esquina superprior izquierda (100, 100), esquina inferior derecha (300, 200), color verde (0, 255, 0), grosor 2
cv2.rectangle(imagen, (100, 100), (300, 200), (0, 255, 0), 2)  # Verde en BGR

cv2.putText(imagen, "Zona de alto tráfico", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Rojo en BGR

imagen_tratada = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # CONVERTIR A RGB POR COMPATIBILIDAD

# Mostrar imagen anotada
plt.figure()
plt.title("Imagen con anotaciones")
plt.imshow(imagen_tratada)
plt.axis('off')
plt.show()