import cv2
import matplotlib.pyplot as plt

def obtenerImagen(ruta):
    return cv2.imread(f"./imagenes/{ruta}") ## LEER IMAGEN IMREAD


imagenCargada = obtenerImagen("imagen_prueba.jpg")

if imagenCargada is None:
    print("Imagen no cargada verifique el codigo proporcionado")
else:
    #PASAMOS DE BGR A RGB YA QUE OPENCV SOLO USA BGR LO CUAL NOS PROVOCA PROBLEMAS PARA MATPLOTLIB QUE TRABAJA EN RGB LO CONVENCIONAL
    imagen_rgb = cv2.cvtColor(imagenCargada, cv2.COLOR_BGR2RGB) ## CONVERTIR A RGB

    #MOSTRAMOS LA IMAGEN CON MATPLOTLIB
    plt.figure(figsize=(8, 6)) # ESTO ES PARA DEFINIR EL TAMAÑO DE LA IMAGEN SIENDO 8 DE ANCHO Y 6 DE ALTO
    plt.imshow(imagen_rgb) # ESTO ES PARA MOSTRAR LA IMAGEN
    plt.title("Imagen Cargada") # ESTO ES PARA PONERLE UN TITULO A LA IMAGEN
    plt.axis("off") # ESTO ES PARA QUE NO SE VISUALICEN LOS EJES DE LA IMAGEN
    plt.show() # ESTO ES PARA MOSTRAR LA IMAGEN