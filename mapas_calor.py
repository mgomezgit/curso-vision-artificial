import cv2
import matplotlib.pyplot as plt
import numpy as np

video_ruta = "./videos/video_tienda.mp4"
#video_ruta2 = "./videos/park_deteccion.avi"

cap = cv2.VideoCapture(video_ruta)

# Aquí creamos al "guardia invisible" que va a vigilar el video. 
# Su único trabajo es aprenderse de memoria cómo es el fondo (el piso, las paredes) para ignorarlo 
# y avisarnos solo cuando algo se mueva (como un cliente caminando).
background_extractor = cv2.createBackgroundSubtractorMOG2(
    history=500,          # ¿Cuánta memoria tiene? Mira 500 fotos atrás para saber qué está quieto. Si un objeto no se mueve en ese tiempo, el guardia asume que ya es parte del piso o la pared.
    varThreshold=16,      # ¿Qué tan sospechoso es? Es la sensibilidad. Un 16 ayuda a ignorar cosas insignificantes como el viento moviendo una cortina o cambios suaves de luz.
    detectShadows=True,   # ¿Es inteligente? Sí, detecta las sombras que la gente proyecta en el suelo al caminar y las separa para que no te arruinen el mapa final.
    )

# "Heatmap" significa Mapa de Calor. Aquí creamos la alcancía donde guardaremos el movimiento.
# Empieza vacía (None) porque primero el video tiene que arrancar para saber de qué tamaño (ancho y alto) debe ser nuestro lienzo.
heatmap_acumulado = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ¿Es el primer segundo del video? Creamos nuestro lienzo invisible.
    # Tomamos las medidas del video (ancho y alto) y creamos un lienzo completamente negro lleno de CEROS.
    # Usamos 'float32' porque los números van a sumar tanto fotograma tras fotograma que necesitamos una alcancía gigante para que no se trabe.
    if heatmap_acumulado is None:
        heatmap_acumulado = np.zeros_like(frame.shape[:2], dtype=np.float32)

    # El guardia hace su magia en el frame actual: borra todo lo que está quieto (lo pinta de negro) 
    # y lo que se esté moviendo (una persona) lo resalta dibujando una silueta blanca.
    fgmask = background_extractor.apply(frame)

    # El efecto "Impresora": Agarramos nuestro lienzo negro y le sumamos (encimamos) la silueta blanca que acaba de pasar.
    # Si alguien pasa corriendo por un pasillo, se suma una vez. Si alguien se queda parado mucho tiempo en una zona (ej. una caja registradora),
    # esa zona recibirá sumas y sumas en cada fotograma, acumulando números altísimos que luego se verán como "puntos calientes".
    heatmap_acumulado = cv2.add(heatmap_acumulado, fgmask.astype(np.float32))

# Cuando el video termina, Matplotlib agarra esa "alcancía" llena de números acumulados
# y los transforma en colores usando el estilo "hot" (donde los ceros son negros y los números más altos brillan en amarillo/rojo).
plt.imshow(heatmap_acumulado, cmap="hot")
plt.title("Heatmap acumulado")
plt.axis("off")
plt.show()


# =====================================================================
# ¿POR QUÉ NORMALIZAMOS? 
# Porque los píxeles acumularon números gigantescos (como 5000 o más) 
# debido a que sumamos 255 en cada fotograma donde hubo movimiento. 
# Como las pantallas solo pueden mostrar colores entre 0 y 255, 
# "aplastamos" proporcionalmente todo para que quepa en ese rango.
# =====================================================================
heatmap_norm = cv2.normalize(heatmap_acumulado, None, 0, 255, cv2.NORM_MINMAX)

# ¿POR QUÉ HACEMOS ESTO? 
# Porque los números normalizados aún tienen decimales (ej. 124.5).
# Las imágenes reales necesitan números enteros limpios (0, 1, 2... 255).
# 'uint8' convierte todo a enteros para que la computadora no gaste memoria de más.
heatmap_norm = np.uint8(heatmap_norm)

# =====================================================================
# ¿POR QUÉ APLICAMOS UN COLORMAP?
# Porque el mapa normalizado es solo una foto aburrida en blanco y negro.
# Al aplicar 'VIRIDIS', obligamos a la computadora a que traduzca:
# Los valores cercanos a 0 (poca acción) se verán azul/morado.
# Los valores cercanos a 255 (el tope de movimiento) brillarán en amarillo.
# Así el ojo humano detecta el flujo de gente al instante.
# =====================================================================
colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_VIRIDIS)

# =====================================================================
# ¿POR QUÉ ESTAS LÍNEAS DE MATPLOTLIB?
# =====================================================================
plt.figure(figsize=(10,8)) # Para que la ventana no se vea chiquita ni deforme.

# ¿Por qué cv2.COLOR_BGR2RGB? Porque OpenCV guarda los colores al revés (Azul-Verde-Rojo)
# y Matplotlib los lee normal (Rojo-Verde-Azul). Si no los volteamos, los colores saldrán cambiados.
plt.imshow(cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB))

plt.title("Heatmap Normalizado") # Para ponerle título al gráfico.
plt.axis("off")                  # Para quitar las coordenadas pixel por pixel y ver solo el mapa.
plt.show()                       # Para que finalmente se abra la ventana en tu pantalla.

