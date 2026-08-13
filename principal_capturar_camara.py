import cv2
import matplotlib.pyplot as plt

# Inicia la captura de video desde la cámara predeterminada
# se coloca 0 para que tome la camara por defecto
# si se pueden tener ams de una camara se expcifica el numero de la camara que se quiere usar 1 2 3 4...
cap = cv2.VideoCapture(0)

# validamos que la cámara se haya abierto correctamente
if  not cap.isOpened():
    print("No se pudo abrir la cámara")
else: 
    while True:
        # COMO NECESITAMOS TOMAR LA CAMARA NOS TOCA HACERLO FRAME POR FRAME POR ESO HACEMOS EL WHILE INFINITO EN CASO DE OBTENER EL FRAME CORRECTAMENTE
        ret, frame = cap.read()  # Lee un frame de la cámara

        if not ret:
            print("No se pudo leer el frame de la cámara")
            break

        cv2.imshow('Captura de Cámara', frame)  # Muestra el frame en una ventana

        #PARA SALIR Y CERRAR TODO APRETAMOS LA TECLA QUE DIGAMOS EN EL SIGUIENTE CODIGO
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    # Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()