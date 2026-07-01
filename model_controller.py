# ============================================================================
# Controlador del coche autonomo en Webots.
#
# El coche se conduce solo combinando cuatro comportamientos:
#
#   1. CONDUCCION (CNN). Una red neuronal tipo NVIDIA mira la imagen de la
#      camara y predice el angulo del volante para seguir el carril. Es el
#      modo normal de manejo.
#
#   2. EVASION. Si hay un autobus parado delante, el coche lo rodea con una
#      maniobra de rebase: sale del carril, lo adelanta y vuelve. Toda esa
#      logica vive en maniobra_evasion.py.
#
#   3. DISTANCIA (ACC). Control de crucero adaptativo: si reconoce un vehiculo
#      delante, ajusta la velocidad para seguirlo manteniendo una separacion
#      de seguridad en vez de chocar.
#
#   4. FRENADO DE EMERGENCIA. Si la camara ve un peaton y el radar lo confirma
#      cerca, frena en seco. Es la prioridad maxima: la seguridad manda.
#
# Ademas pinta tres paneles de telemetria (estado, LiDAR y radar) en los
# displays del tablero.
#
# Antes de usarlo hay que convertir una sola vez los pesos del modelo:
#   python convert_keras_to_pytorch.py   ->  genera nvidia_model.pth
# ============================================================================

from controller import Display
from vehicle import Car, Driver
import numpy as np
import cv2
import math
import os
import torch

from nvidia_pytorch import NvidiaModel
from maniobra_evasion import ManiobraEvasion

# ---- Configuracion ----
MODEL_PATH = "nvidia_model.pth"     # checkpoint del modelo, junto al controlador
MAX_ANGLE = 0.15                     # limite del angulo del volante (rad)
CRUISE_SPEED = 30                   # velocidad de crucero (km/h)
STEER_SMOOTHING = 0.3               # suavizado del volante: 0 = nada, ~1 = muy suave
CNN_STEER_GAIN  = 1.5               # amplifica el angulo de la CNN (>1 cierra mas las curvas)
# Recorte de la imagen. Debe ser EL MISMO que uso el grabador para entrenar:
# solo la mitad inferior del frame. Si cambia, el modelo ve algo distinto.
DATASET_ROI_TOP_FRAC = 0.53
# LiDAR: cono de vision y rango valido (se usa para la evasion y su display).
LIDAR_ANGLE_DEG = 15        # angulo total del cono frontal (grados)
LIDAR_MAX_RANGE = 20.0      # distancia maxima que se tiene en cuenta (m)
LIDAR_MIN_RANGE = 0.1       # distancia minima (descarta ruido muy cercano)
LIDAR_PREAVISO_RANGE = 15.0 # al ver el bus a esta distancia, el coche endereza y espera
# ACC: seguir al vehiculo de delante.
ACC_GAP = 5.0           # separacion que se intenta mantener (m)
ACC_RANGE = 30.0        # solo se sigue al de delante si esta a esta distancia o menos (m)
ACC_KP = 3.0            # ganancia del control de distancia (km/h por metro de error)
ACC_DIST_SMOOTH = 0.6   # suavizado de la distancia frontal (quita ruido del LiDAR)
ACC_SPEED_SMOOTH = 0.3  # suavizado del comando de velocidad (0 = sin suavizar)
ACC_BRAKE = 0.5         # intensidad de freno del ACC para mantener distancia [0,1] (mas suave que el del peaton)
ACC_BRAKE_MARGIN = 2.0  # km/h: solo frena si la velocidad real supera a la objetivo del ACC en mas de esto
# Frenado de emergencia por peaton.
BRAKE_PEATON = 1.0      # intensidad de freno [0, 1]; 1.0 es el maximo de la API
PEATON_BRAKE_DIST = 7.0 # se frena si el peaton (radar) esta a esta distancia o menos (m)
# Palabras (en minuscula) con las que se reconoce un vehiculo en el campo 'model'
# de la camara. Cubren los PROTOs de Webots que usa el trafico de SUMO y nombres
# genericos. Se buscan como subcadena, asi "bmw" encaja con "BmwX5" y "BMW X5".
# El ACC solo regula la velocidad cuando ve uno de estos delante.
VEHICULOS = (
    # genericos / personalizados
    "car", "auto", "carro", "coche", "vehic",
    # PROTOs estandar de Webots
    "bmw",              # BmwX5
    "citroen",          # CitroenCZero
    "lincoln",          # LincolnMKZ
    "rover",            # RangeRoverSportSVR
    "tesla",            # TeslaModel3
    "toyota", "prius",  # ToyotaPrius
    "mercedes", "sprinter",  # MercedesBenzSprinter
)

# Semaforo en rojo
SEMAFORO_BRAKE      = 1.0   # intensidad de freno [0, 1]
SEMAFORO_STOP_DIST  = 12.0  # frena si el semaforo rojo esta a <= esta distancia (m)

SEMAFORO_DETECT_SIZE_PX = 4    # tamaño minimo para DETECTAR (lejos, activa slowdown)
SEMAFORO_LATCH_SIZE_PX  = 6    # tamaño para LATCHEAR y frenar en seco
SEMAFORO_SLOW_SPEED     = 6.0  # velocidad de crucero reducida mientras se acerca (km/h)

# Semaforo - deteccion de verde
GREEN_HSV_LOWER = np.array([40,  80, 80])
GREEN_HSV_UPPER = np.array([85, 255, 255])
GREEN_MIN_PIXELS = 8   # menos pixeles que el rojo porque la luz verde es mas pequeña
SEMAFORO_TIMEOUT = 5.0   # segundos maximos esperando en rojo antes de soltar el latch


# Rango HSV para detectar rojo (dos rangos porque el rojo rodea los 0/180 grados)
RED_HSV_LOWER1 = np.array([0,   120, 100])
RED_HSV_UPPER1 = np.array([8,   255, 255])
RED_HSV_LOWER2 = np.array([172, 120, 100])
RED_HSV_UPPER2 = np.array([180, 255, 255])
RED_MIN_PIXELS = 30   # minimo de pixeles rojos para considerar que hay luz roja


# True si el objeto reconocido es un autobus. Se excluye la parada de autobus
# ("bus stop"), que NO debe disparar la maniobra de evasion.
def es_bus(model_lower):
    return "bus" in model_lower and "stop" not in model_lower


# Lee la imagen de la camara como arreglo (alto, ancho, 4) en formato BGRA.
def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


# Prepara la imagen igual que en el entrenamiento: recorta la mitad inferior
# (el mismo ROI del grabador) y normaliza a [0, 1]. Nada mas: el modelo se
# entreno solo con img/255, sin YUV ni suavizado.
def preprocess(img_rgb):
    roi_top = int(img_rgb.shape[0] * DATASET_ROI_TOP_FRAC)
    img = img_rgb[roi_top:, :, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # La red espera 320x76. Con la camara del mundo el ROI ya sale a ese tamano,
    img = cv2.resize(img, (320, 76))
    img = img / 127.5 - 1.0
    return img


# Distancia al objeto mas cercano dentro del cono frontal del LiDAR (m), o None
# si no hay nada valido. Sirve para disparar la evasion y para el display.
def lidar_distancia_frontal(lidar):
    ranges = lidar.getRangeImage()
    if not ranges:
        return None
    n = len(ranges)
    sensor_fov = lidar.getFov()
    half_view = math.radians(LIDAR_ANGLE_DEG / 2)
    closest = float('inf')
    for i, r in enumerate(ranges):
        angle = -sensor_fov / 2 + (i / max(n - 1, 1)) * sensor_fov
        if abs(angle) > half_view:
            continue
        if r == float('inf') or r < LIDAR_MIN_RANGE or r > LIDAR_MAX_RANGE:
            continue
        if r < closest:
            closest = r
    return closest if closest < float('inf') else None

def detectar_luz_roja(img_rgb, camera):
    """
    Devuelve (hay_rojo, distancia_estimada_px) donde distancia_estimada_px
    es el tamaño vertical del objeto en imagen (proxy de distancia: mas grande = mas cerca).
    Solo dispara si el recognition API confirma un traffic light con rojo en su ROI.
    Sin fallback CV para evitar falsos positivos con edificios/carros rojos.
    """
    objs = camera.getRecognitionObjects()
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    for obj in objs:
        model = obj.getModel().lower()
        if "traffic" not in model and "semaforo" not in model and "light" not in model:
            continue
        pos2d  = obj.getPositionOnImage()
        size2d = obj.getSizeOnImage()
        x0 = max(0, int(pos2d[0] - size2d[0] / 2))
        x1 = min(w, int(pos2d[0] + size2d[0] / 2))
        y0 = max(0, int(pos2d[1] - size2d[1] / 2))
        y1 = min(h, int(pos2d[1] + size2d[1] / 2))
        roi_red = red_mask[y0:y1, x0:x1]
        if roi_red.size > 0 and cv2.countNonZero(roi_red) >= RED_MIN_PIXELS:
            return True, size2d[1]   # confirmado: rojo en un traffic light reconocido

    return False, 0

def detectar_luz_verde(img_rgb, camera):
    """Devuelve True si el recognition ve un traffic light con verde en su ROI."""
    objs = camera.getRecognitionObjects()
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)

    for obj in objs:
        model = obj.getModel().lower()
        if "traffic" not in model and "semaforo" not in model and "light" not in model:
            continue
        pos2d  = obj.getPositionOnImage()
        size2d = obj.getSizeOnImage()
        x0 = max(0, int(pos2d[0] - size2d[0] / 2))
        x1 = min(w, int(pos2d[0] + size2d[0] / 2))
        y0 = max(0, int(pos2d[1] - size2d[1] / 2))
        y1 = min(h, int(pos2d[1] + size2d[1] / 2))
        roi_green = green_mask[y0:y1, x0:x1]
        if roi_green.size > 0 and cv2.countNonZero(roi_green) >= GREEN_MIN_PIXELS:
            return True
    return False

# Distancia al blanco de radar mas cercano (m), o None si no hay ninguno. El
# radar tiene mas alcance y un cono mas amplio que el LiDAR, asi que no pierde
# al peaton de cerca. Se usa para confirmar la distancia en el frenado.
def radar_distancia_frontal(radar):
    targets = radar.getTargets()
    if not targets:
        return None
    return min(t.distance for t in targets)


# Velocidad objetivo del ACC (km/h) para seguir al de delante a ACC_GAP metros:
#   - Sin nadie cerca         -> velocidad de crucero.
#   - El de delante mas lento -> bajamos para igualarlo.
#   - El de delante parado    -> nos detenemos a ACC_GAP metros.
# Se estima su velocidad por como cambia la distancia (anticipacion) y se corrige
# el error de separacion con una ganancia proporcional.
def acc_velocidad_objetivo(distancia, dist_prev, current_speed, dt):
    if distancia is None or distancia > ACC_RANGE:
        return CRUISE_SPEED
    if dist_prev is None or dt <= 0:
        v_lider = current_speed
    else:
        closing = (distancia - dist_prev) / dt      # m/s (+ si se aleja)
        v_lider = current_speed + closing * 3.6      # km/h
    objetivo = v_lider + ACC_KP * (distancia - ACC_GAP)
    return max(0.0, min(CRUISE_SPEED, objetivo))


# Dibuja los circulos de referencia (25/50/75/100% del alcance) y la etiqueta
# en metros. Lo comparten las vistas de LiDAR y de radar.
def dibujar_circulos_alcance(canvas, cx, cy, scale, max_range):
    for frac, color in ((0.25, (250, 0, 0)), (0.5, (255, 165, 0)),
                        (0.75, (255, 165, 0)), (1.0, (0, 250, 0))):
        cv2.circle(canvas, (cx, cy), int(scale * frac), color, 1)
    cv2.putText(canvas, f"{max_range:.0f}m", (cx + scale + 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1)


# Dibuja las dos lineas que marcan los limites del cono de vision (+- half_angle).
def dibujar_lineas_cono(canvas, cx, cy, scale, half_angle):
    for ang in (-half_angle, half_angle):
        ex = int(cx + scale * math.sin(ang))
        ey = int(cy - scale * math.cos(ang))
        cv2.line(canvas, (cx, cy), (ex, ey), (0, 200, 255), 1)


# Panel del LiDAR: vista de pajaro del barrido frontal, mas las lecturas de los
# sensores laterales y del giroscopio.
def render_lidar_display(display, lidar, ds_frontal, ds_trasero, gyro, distancia_frontal):
    w = display.getWidth()
    h = display.getHeight()
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Vista de pajaro del barrido ---
    ranges = lidar.getRangeImage()
    if ranges:
        n = len(ranges)
        sensor_fov = lidar.getFov()
        half_view = math.radians(LIDAR_ANGLE_DEG / 2)
        cx = w // 2 + w // 4
        cy = h * 2 // 3
        scale = min(w - cx, cy) - 6

        dibujar_circulos_alcance(canvas, cx, cy, scale, LIDAR_MAX_RANGE)

        # Un punto por cada rayo dentro del cono y con rango valido
        for i, r in enumerate(ranges):
            angle = -sensor_fov / 2 + (i / max(n - 1, 1)) * sensor_fov
            if abs(angle) > half_view:
                continue
            if r == float('inf') or r < LIDAR_MIN_RANGE or r > LIDAR_MAX_RANGE:
                continue
            px = int(cx + (r / LIDAR_MAX_RANGE) * scale * math.sin(angle))
            py = int(cy - (r / LIDAR_MAX_RANGE) * scale * math.cos(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 1, (0, 220, 0), -1)

        dibujar_lineas_cono(canvas, cx, cy, scale, half_view)

        # Marcador del coche, nombre del sensor y distancia al objeto mas cercano
        cv2.circle(canvas, (cx, cy), 4, (0, 160, 255), -1)
        cv2.putText(canvas, "Sick LMS 291",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        det_text = f"Dist: {distancia_frontal:.2f} m" if distancia_frontal is not None else "Dist: ---"
        cv2.putText(canvas, det_text,
                    (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)

    # --- Sensores laterales ---
    d_front = ds_frontal.getValue()
    d_rear = ds_trasero.getValue()
    cv2.putText(canvas, "Sensores lat.",
                    (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(canvas, f"Frontal: {d_front:.3f}", (4, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(canvas, f"Trasero: {d_rear:.3f}", (4, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # --- Giroscopio (eje z) ---
    gz = gyro.getValues()[2]
    cv2.putText(canvas, "Giroscopio",
                (4, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(canvas, f"Eje Z: {gz:+.4f} rad/s", (4, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    ref = display.imageNew(canvas.tobytes(), Display.RGB, width=w, height=h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)


# Panel del radar: vista de pajaro de los blancos detectados (por distancia y
# azimut) y la telemetria de seguimiento (velocidad real, objetivo del ACC,
# distancia al de delante y objetos reconocidos por la camara).
def render_radar_display(display, radar, objs, acc_speed, speed, distancia_frontal):
    w = display.getWidth()
    h = display.getHeight()
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    max_range = radar.getMaxRange()
    hfov = radar.getHorizontalFov()
    targets = radar.getTargets()

    # --- Vista de pajaro de los blancos ---
    cx = w // 2 + w // 4
    cy = h * 2 // 3
    scale = min(w - cx, cy) - 6

    dibujar_circulos_alcance(canvas, cx, cy, scale, max_range)
    dibujar_lineas_cono(canvas, cx, cy, scale, hfov / 2)

    # Un punto por cada blanco, ubicado por su distancia y azimut
    for t in targets:
        if max_range > 0:
            px = int(cx + (t.distance / max_range) * scale * math.sin(t.azimuth))
            py = int(cy - (t.distance / max_range) * scale * math.cos(t.azimuth))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 3, (255, 80, 80), -1)

    # Marcador del coche, titulo y numero de blancos
    cv2.circle(canvas, (cx, cy), 4, (0, 160, 255), -1)
    cv2.putText(canvas, "Radar", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(canvas, f"Blancos: {len(targets)}", (4, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)

    # --- Telemetria de seguimiento ---
    cv2.putText(canvas, f"Vel real: {speed:.1f} km/h", (4, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 120), 1)
    if acc_speed is not None:
        cv2.putText(canvas, f"ACC Vobj: {acc_speed:.1f} km/h", (4, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 120), 1)
    dist_act = f"{distancia_frontal:.1f}" if distancia_frontal is not None else "---"
    cv2.putText(canvas, f"Dist a/o: {dist_act} / {ACC_GAP:.1f} m", (4, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 120), 1)

    # --- Objetos reconocidos (bus, peatones y autos) ---
    cv2.putText(canvas, "Reconocimiento obj.",
                (4, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    # Solo nos interesa el peaton (persona): se excluye la marca vial "pedestrian
    # crossing" (paso de cebra), que contiene "pedestrian" pero no es una persona.
    filtered = [o for o in objs
                if es_bus(m := o.getModel().lower()) or
                   ("pedestrian" in m and "crossing" not in m) or
                   any(a in m for a in VEHICULOS)]
    if filtered:
        for idx, obj in enumerate(filtered[:2]):    # max 2 para que quepan
            cv2.putText(canvas, f"  {obj.getModel()}", (4, 102 + idx * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
    else:
        cv2.putText(canvas, "  Sin deteccion", (4, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    ref = display.imageNew(canvas.tobytes(), Display.RGB, width=w, height=h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)


# HUD de estado: luces CNN/EVASION, el modo actual y barras de volante,
# velocidad y freno.
def render_status_display(display, final_steering, modo, res, speed, brake):
    w = display.getWidth()
    h = display.getHeight()
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Luces de estado (verde = activo, rojo = inactivo) ---
    cnn_activo = (modo == "CNN")
    ev_activo  = res.maniobra_activa
    cv2.circle(canvas, (12, 13), 7, (0, 200, 0) if cnn_activo else (210, 0, 0), -1)
    cv2.putText(canvas, "CNN", (23, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.circle(canvas, (w // 2 + 4, 13), 7, (0, 200, 0) if ev_activo else (210, 0, 0), -1)
    cv2.putText(canvas, "EVASION", (w // 2 + 15, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Texto del estado (con sub-fase si la hay)
    sub = f"/{res.sub_fase}" if res.sub_fase else ""
    cv2.putText(canvas, f"Estado: {modo}{sub}", (4, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    cv2.line(canvas, (4, 40), (w - 4, 40), (55, 55, 55), 1)

    # --- Barras de telemetria ---
    LBL = 44          # ancho para la etiqueta
    BX0 = LBL + 2    # inicio de la barra
    BX1 = w - 5      # fin de la barra
    BW  = BX1 - BX0  # ancho util

    # Barra con cero al centro (valores +/-, p. ej. el volante)
    def barra_centrada(y, valor, max_val, label):
        cv2.putText(canvas, label, (2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.rectangle(canvas, (BX0, y - 7), (BX1, y + 7), (28, 28, 28), -1)
        cx = BX0 + BW // 2
        cv2.line(canvas, (cx, y - 7), (cx, y + 7), (65, 65, 65), 1)
        norm = max(-1.0, min(1.0, valor / max_val)) if max_val else 0.0
        fx   = int(cx + norm * (BW // 2))
        col  = (0, 180, 70) if norm >= 0 else (200, 100, 0)
        if fx > cx:
            cv2.rectangle(canvas, (cx + 1, y - 6), (fx, y + 6), col, -1)
        elif fx < cx:
            cv2.rectangle(canvas, (fx, y - 6), (cx - 1, y + 6), col, -1)
        fx = max(BX0 + 4, min(BX1 - 4, fx))
        cv2.circle(canvas, (fx, y), 4, (230, 230, 230), -1)
        cv2.putText(canvas, f"{valor:+.3f}", (BX0 + 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Barra que crece de izquierda a derecha (valores positivos, p. ej. velocidad)
    def barra_lineal(y, valor, max_val, color, label):
        cv2.putText(canvas, label, (2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.rectangle(canvas, (BX0, y - 7), (BX1, y + 7), (28, 28, 28), -1)
        frac = max(0.0, min(1.0, valor / max_val)) if max_val > 0 else 0.0
        fx   = int(BX0 + frac * BW)
        if fx > BX0:
            cv2.rectangle(canvas, (BX0, y - 6), (fx, y + 6), color, -1)
        fx = max(BX0 + 4, min(BX1 - 4, fx))
        cv2.circle(canvas, (fx, y), 4, (230, 230, 230), -1)
        cv2.putText(canvas, f"{valor:.2f}", (BX0 + 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    barra_centrada(62,  final_steering, MAX_ANGLE,          "Steer")
    barra_lineal  (92,  speed,          CRUISE_SPEED * 1.2, (0, 170, 220), "Speed")
    barra_lineal  (122, brake,          1.0,                (30,  60, 220), "Brake")

    ref = display.imageNew(canvas.tobytes(), Display.RGB, width=w, height=h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)


def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    display_img = Display("display")

    lidar = robot.getDevice("Sick LMS 291")
    lidar.enable(timestep)

    ds_frontal = robot.getDevice("sensor_distancia_frontal")
    ds_frontal.enable(timestep)
    ds_trasero = robot.getDevice("sensor_distancia_trasero")
    ds_trasero.enable(timestep)

    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    radar = robot.getDevice("radar")
    radar.enable(timestep)

    semaforo_frenando = False
    semaforo_latch = False   # True = estamos detenidos esperando verde; no se suelta hasta ver verde
    semaforo_latch_timer = 0.0

    display_lidar = Display("display_lidar")
    display_radar = Display("display_radar")

    # Carga del modelo. La red es pequena, asi que con CPU basta; si hay GPU
    # (CUDA o MPS) se aprovecha, pero no hace falta. eval() apaga el Dropout
    # para que la prediccion sea estable.
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontro el modelo '{model_path}'. Convierte primero los "
            f"pesos con: python convert_keras_to_pytorch.py"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = NvidiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo cargado: {model_path}  (device: {device})")

    evasion = ManiobraEvasion(dist_evasion=7.0, on_log=print)
    smoothed_steering = 0.0
    dt = timestep / 1000.0
    modo_hold = False           # True: CNN en pausa, coche recto esperando la evasion
    dist_filt = None            # distancia frontal suavizada para el ACC
    dist_prev = None            # la misma del frame anterior (para estimar velocidad)
    acc_speed = CRUISE_SPEED    # comando de velocidad del ACC (km/h), suavizado
    peaton_frenando = False     # True mientras dura el freno de emergencia (para el log)
    acc_frenando = False        # True mientras el ACC frena para mantener distancia (para el log)

    while robot.step() != -1:
        image = get_image(camera)                       # BGRA

        # --- Conduccion CNN (se calcula siempre; se aplica solo si no hay maniobra) ---
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        x = preprocess(rgb).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            steering_cnn = float(model(x).item()) * CNN_STEER_GAIN
        steering_cnn = max(-MAX_ANGLE, min(MAX_ANGLE, steering_cnn))

        # --- Lectura de sensores y reconocimiento ---
        distancia_frontal = lidar_distancia_frontal(lidar)
        gz = gyro.getValues()[2]
        objs = camera.getRecognitionObjects()
        
        # El radar confirma que el peaton esta cerca (mas fiable que el LiDAR aqui).
        distancia_radar = radar_distancia_frontal(radar)
                                
        luz_roja, semaforo_size_px = detectar_luz_roja(rgb, camera)

        luz_verde = detectar_luz_verde(rgb, camera)

        # --- Maquina de estados del semaforo ---
        # DETECTADO (lejos): ve rojo pequeno -> reduce velocidad y espera confirmar
        # LATCHEADO  (cerca): rojo grande confirmado -> frena en seco y espera verde
        if luz_roja and semaforo_size_px >= SEMAFORO_DETECT_SIZE_PX:
            if not semaforo_latch:
                semaforo_latch = True
                semaforo_latch_timer = 0.0
                print(f"[SEMAFORO] Rojo detectado (size={semaforo_size_px}px): reduciendo velocidad")

        if semaforo_latch:
            semaforo_latch_timer += dt

        if semaforo_latch:
            if luz_verde:
                semaforo_latch = False
                semaforo_latch_timer = 0.0
                print("[SEMAFORO] Verde detectado: reanudando")
            elif semaforo_latch_timer >= SEMAFORO_TIMEOUT:
                semaforo_latch_timer = 0.0
                if luz_roja:
                    print(f"[SEMAFORO] Timeout: sigue en rojo, esperando {SEMAFORO_TIMEOUT}s mas")
                else:
                    semaforo_latch = False
                    print("[SEMAFORO] Timeout sin rojo visible: reanudando")

        # Dos niveles: slowdown (ve rojo lejos) vs freno total (latch activo y rojo grande)
        semaforo_cerca    = semaforo_latch and (not luz_roja or semaforo_size_px >= SEMAFORO_LATCH_SIZE_PX)

        bus_enfrente = any(es_bus(o.getModel().lower()) for o in objs)
        # Peaton: solo por la camara, excluyendo el paso de cebra ("pedestrian
        # crossing"), que tambien contiene la palabra "pedestrian".
        peaton_enfrente = any(
            "pedestrian" in (m := o.getModel().lower()) and "crossing" not in m
            for o in objs
        )

        peaton_cerca = (peaton_enfrente and distancia_radar is not None
                        and distancia_radar <= PEATON_BRAKE_DIST)
        # Vehiculo delante (auto de SUMO o bus). El ACC solo regula la velocidad
        # si hay uno; asi no frena por muros, postes ni peatones.
        vehiculo_enfrente = bus_enfrente or any(
            any(v in o.getModel().lower() for v in VEHICULOS)
            for o in objs
        )

        # --- Entrar en HOLD: al ver el bus de lejos, el coche endereza y espera ---
        if not modo_hold and not evasion.maniobra_activa:
            if bus_enfrente and distancia_frontal is not None \
               and distancia_frontal <= LIDAR_PREAVISO_RANGE:
                modo_hold = True
                smoothed_steering = 0.0
                print(f"[HOLD] Bus a {distancia_frontal:.2f} m: CNN en pausa, avance recto")

        # --- Salir de HOLD si el bus ya no esta delante (y no hay maniobra). Evita
        #     quedarse recto para siempre si el bus desaparece sin disparar la evasion.
        if modo_hold and not evasion.maniobra_activa and not bus_enfrente:
            modo_hold = False
            print("[HOLD] Bus ya no detectado: se reanuda la CNN")

        # --- Maquina de estados de la evasion (se actualiza siempre) ---
        res = evasion.update(
            dt=dt,
            gz=gz,
            obstaculo_enfrente=bus_enfrente,
            distancia_frontal=distancia_frontal,
            lat_delantero=ds_frontal.getValue(),
            lat_trasero=ds_trasero.getValue(),
        )

        # --- Quien manda en el volante: EVASION > HOLD (recto) > CNN ---
        if res.maniobra_activa and res.angulo is not None:
            final_steering = res.angulo
        elif modo_hold:
            final_steering = 0.0
        else:
            if res.pid_reset:                    # maniobra recien terminada
                smoothed_steering = 0.0
            smoothed_steering = (STEER_SMOOTHING * smoothed_steering +
                                 (1 - STEER_SMOOTHING) * steering_cnn)
            final_steering = smoothed_steering

        if res.pid_reset:                        # al acabar la maniobra, volver a CNN
            modo_hold = False

        driver.setSteeringAngle(final_steering)

        # --- Velocidad: ACC para seguir al de delante ---
        # Solo se regula si la camara ve un vehiculo (no por muros ni peatones).
        # Durante la evasion se mantiene la velocidad de crucero para adelantar.
        # --- Velocidad: prioridad semaforo > ACC ---
        if res.maniobra_activa or not vehiculo_enfrente:
            dist_filt = None
            target_speed = CRUISE_SPEED
        else:
            if distancia_frontal is None:
                dist_filt = None
            elif dist_filt is None:
                dist_filt = distancia_frontal
            else:
                dist_filt = (ACC_DIST_SMOOTH * dist_filt +
                            (1 - ACC_DIST_SMOOTH) * distancia_frontal)
            target_speed = acc_velocidad_objetivo(
                dist_filt, dist_prev, driver.getCurrentSpeed(), dt)
        dist_prev = dist_filt

        # Semaforo limita la velocidad objetivo ANTES de suavizar
        if semaforo_latch and not semaforo_cerca:    # slowdown activo
            target_speed = min(target_speed, SEMAFORO_SLOW_SPEED)

        acc_speed = (ACC_SPEED_SMOOTH * acc_speed +
                    (1 - ACC_SPEED_SMOOTH) * target_speed)

        # --- Frenado de emergencia por peaton ---
        # Prioridad maxima: pisa al ACC y a la evasion. Control directo: corta el
        # acelerador (setThrottle 0) y aplica el freno al maximo (setBrakeIntensity).
        if peaton_cerca:
            brake = BRAKE_PEATON
            acc_speed = 0.0
            driver.setThrottle(0.0)              # aceleracion (par del motor) a 0
            driver.setBrakeIntensity(brake)      # freno al maximo
            if not peaton_frenando:              # se entra a frenar: avisa una vez
                print(f"[FRENO] Peaton a {distancia_radar:.2f} m: FRENO DE EMERGENCIA")
                peaton_frenando = True
        
        elif semaforo_cerca:                        
            brake = SEMAFORO_BRAKE
            acc_speed = 0.0
            driver.setThrottle(0.0)
            driver.setBrakeIntensity(brake)
            if not semaforo_frenando:
                print(f"[SEMAFORO] Luz roja detectada (size={semaforo_size_px}px): deteniendo")
                semaforo_frenando = True

        else:
            if peaton_frenando:                  # veniamos de un frenado por peaton
                print("[FRENO] Peaton despejado: freno liberado")
                peaton_frenando = False

            if semaforo_frenando and not semaforo_cerca:
                print("[SEMAFORO] Luz roja despejada: reanudando")
                semaforo_frenando = False

            # --- Frenado del ACC para mantener la distancia ---
            # setCruisingSpeed solo controla el acelerador: al bajar la velocidad
            # objetivo deja de acelerar, pero no frena. Si seguimos a un vehiculo y
            # la velocidad real supera a la objetivo, aplicamos un freno suave (mas
            # flojo que el del peaton) para cerrar la distancia con decision.
            siguiendo = vehiculo_enfrente and not res.maniobra_activa
            if siguiendo and driver.getCurrentSpeed() - acc_speed > ACC_BRAKE_MARGIN:
                brake = ACC_BRAKE
                if not acc_frenando:             # se entra a frenar: avisa una vez
                    print(f"[ACC] Acercandose demasiado: freno al {ACC_BRAKE * 100:.0f} %")
                    acc_frenando = True
            else:
                brake = 0.0
                if acc_frenando:                 # se deja de frenar: avisa una vez
                    print("[ACC] Distancia estabilizada: freno liberado")
                    acc_frenando = False
            crucero = max(0.0, acc_speed)
            driver.setBrakeIntensity(brake)
            driver.setCruisingSpeed(crucero)

        # ---- Telemetria en los displays ----
        if res.maniobra_activa:
            modo = res.estado
        elif modo_hold:
            modo = "HOLD"
        elif semaforo_cerca:
            modo = "SEMAFORO"
        else:
            modo = "CNN"
        speed = driver.getCurrentSpeed()
        render_status_display(display_img, final_steering, modo, res, speed, brake)
        render_lidar_display(display_lidar, lidar, ds_frontal, ds_trasero, gyro, distancia_frontal)
        render_radar_display(display_radar, radar, objs, acc_speed, speed, distancia_frontal)


if __name__ == "__main__":
    main()
