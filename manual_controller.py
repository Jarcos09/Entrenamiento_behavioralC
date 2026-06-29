from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv

# ---- Constantes de control ----
MAX_ANGLE = 0.3              # angulo de direccion maximo (rad)
SPEED_INCR = 5               # cuanto se baja la velocidad en huecos cortos
SPEED_INTERSECTION = 30      # velocidad al cruzar una interseccion
SPEED_STRAIGHT = 50          # velocidad normal siguiendo el carril

# ---- Ajuste del seguimiento de carril ----
# Fraccion de la altura donde empieza el ROI (0 = arriba, 1 = abajo).
ROI_TOP_FRAC = 0.55
# Numero de franjas horizontales que se escanean buscando el dash.
NUM_STRIPS = 10
# Ancho maximo (fraccion del ancho de imagen) para aceptar un cluster como
# dash de carril. Mas ancho = cebra, linea de alto o marca de interseccion.
MAX_CLUSTER_WIDTH_FRAC = 0.18
# Salto maximo (px) permitido entre la eleccion de una franja y search_x.
# Rechaza carteles al borde de la via, que quedan lejos del dash.
MAX_STRIP_JUMP = 35
# Limite absoluto (px) alrededor del objetivo del carril. Evita engancharse al
# carril opuesto o a marcas de cruce cuando nuestro dash desaparece un momento.
MAX_TARGET_DEVIATION = 50
# Si casi todas las franjas tienen mucha cobertura blanca a la vez, es una
# cebra llenando el cuadro y no una linea delgada.
CROSSWALK_COVERAGE_FRAC = 0.35   # fraccion del ancho contada como "blanca"
CROSSWALK_STRIP_FRACTION = 0.6   # fraccion de franjas que deben superar lo anterior
# Suavizado del error (mayor = mas reactivo, menos suave).
ERROR_EMA_ALPHA = 0.35
# Frames consecutivos sin deteccion antes de tratarlo como hueco real
# (gap del dash / interseccion) en vez de ruido momentaneo.
COAST_FRAMES_CURVE = 5
# Posicion objetivo de la linea blanca en la imagen (fraccion del ancho desde
# el borde izquierdo). Define la posicion lateral del auto en el carril:
# menor = el auto se aleja mas de la linea (mas a la derecha); mayor = mas pegado.
WHITE_TARGET_FRAC = 0.20

# ---- Grabacion del dataset ----
# Carpeta (relativa al controlador) donde se guardan imagenes + etiquetas.
DATASET_DIR = "dataset_curvas"
# Etiqueta para distinguir datasets de distintas personas/corridas y evitar
# colisiones de nombres al juntarlos.
DATASET_TAG = "JC"
# |steering| a partir del cual el frame se considera "curva".
CURVE_STEER_THRESHOLD = 0.04
# Cadencia de guardado por tiempo
SAMPLE_PERIOD_STRAIGHT = 0.2   # segundos entre muestras en recta (2/seg)
SAMPLE_PERIOD_CURVE = 0.1      # segundos entre muestras en curva (10/seg)
# ROI del dataset: la camara esta por encima del auto, asi que la mitad
# superior es cielo/edificios. Guardamos solo de esta fraccion hacia abajo
# (0.5 = mitad inferior). Con la camara en 320x240 -> imagenes de 320x120.
DATASET_ROI_TOP_FRAC = 0.53
# Recorte de la parte derecha (carril opuesto/edificios) con borde inclinado:
# fraccion del ancho que se quita por la derecha, abajo y arriba del ROI.
DATASET_ROI_RIGHT_BOTTOM = 0.0
DATASET_ROI_RIGHT_TOP = 0.0

# ---- Conduccion manual (CIL) ----
# Segundo modo: el operador conduce con el teclado y da comandos de navegacion
# en las intersecciones (seguir / izquierda / recto / derecha) mientras se
# guarda el angulo. El operador controla el acelerador, el freno y la reversa
# con las flechas Arriba/Abajo (la velocidad NO forma parte del entrenamiento).
# Acelerador / freno / reversa: la velocidad es un
# valor con signo en km/h. > 0 avanza, < 0 reversa. setCruisingSpeed con una
# velocidad negativa engancha la reversa automaticamente, asi que al frenar
# desde adelante la velocidad cruza 0 y pasa sola a reversa.
MANUAL_MAX_SPEED = 40       # km/h maxima hacia adelante
MANUAL_MAX_REVERSE = -30    # km/h maxima en reversa (valor negativo)
MANUAL_SPEED_INCR = 5       # km/h que sube/baja por pulsacion de flecha
SPEED_DEBOUNCE = 0.1        # s entre escalones de velocidad (antirrebote)
# Limite del angulo en manual. Mayor que el del PID porque los giros del centro
# de los mundos son muy abruptos y necesitan mas angulo para tomarse bien.
MANUAL_MAX_ANGLE = 0.1
ANGLE_INCR = 0.04           # cuanto cambia el angulo por frame con las flechas
ANGLE_DECAY = 0.85          # auto-centrado del volante al soltar las flechas
# Vuelta cerrada: dos teclas (A=izquierda, D=derecha) que permiten un angulo
# mucho mayor que las flechas, para tomar una esquina / dar vuelta en una calle.
MANUAL_SHARP_ANGLE = 0.5    # tope de angulo en vuelta cerrada (rad)
SHARP_ANGLE_INCR = 0.10     # incremento por frame en vuelta cerrada (mas agresivo)

CMD_FOLLOW = 2              # seguir carril (por defecto; tambien el modo LANE)
CMD_LEFT = 3               # girar a la izquierda
CMD_RIGHT = 4              # girar a la derecha
CMD_STRAIGHT = 5           # seguir derecho en la interseccion
CMD_NAMES = {CMD_FOLLOW: "SEGUIR", CMD_LEFT: "IZQUIERDA",
             CMD_RIGHT: "DERECHA", CMD_STRAIGHT: "CRUCE"}


# Obtiene la imagen de la camara como arreglo (alto, ancho, 4) en BGRA.
def get_image(camera):
    raw_image = camera.getImage()
    return np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

def display_image_bgr(display, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(), Display.RGB,
        width=image_rgb.shape[1], height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    # Liberamos la referencia cada frame para no acumularlas en corridas largas.
    display.imageDelete(image_ref)


# Detecta la linea blanca (poca saturacion, mucho valor) en espacio HSV.
def detect_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    return cv2.inRange(hsv, lower_white, upper_white)


# Recorta la mitad inferior de la imagen (region de interes) con fillPoly.
def apply_roi(image, top_frac=ROI_TOP_FRAC):
    height, width = image.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (0, height * top_frac),
        (width, height * top_frac),
        (width, height)
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(image, mask)


# ROI del dataset: recorta la mitad superior (cielo) y enmascara en negro la
# parte derecha con un borde inclinado (mas recorte arriba que abajo), para
# quitar el carril opuesto / los edificios de la derecha.
def apply_dataset_roi(frame_bgr):
    roi_top = int(frame_bgr.shape[0] * DATASET_ROI_TOP_FRAC)
    roi = frame_bgr[roi_top:, :, :]
    h, w = roi.shape[:2]
    x_top = int(w * (1 - DATASET_ROI_RIGHT_TOP))        # borde derecho arriba
    x_bottom = int(w * (1 - DATASET_ROI_RIGHT_BOTTOM))  # borde derecho abajo
    poly = np.array([[(0, 0), (x_top, 0), (x_bottom, h), (0, h)]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(roi, roi, mask=mask)


# ============================================================
# RASTREADOR DEL DASH BLANCO (ventana deslizante)
# Escanea franjas horizontales de la mascara blanca, en cada una toma el
# cluster mas cercano a la posicion esperada del carril y descarta los que son
# demasiado anchos para ser un dash (cebras, lineas de alto, marcas de cruce).
# Devuelve: (x_ponderada_o_None, puntos_debug, bandera_cebra)
# ============================================================
def track_white_dash(white_roi, prev_x, img_width, img_height, lane_target,
                     roi_top_frac=ROI_TOP_FRAC, num_strips=NUM_STRIPS):
    strip_top = int(img_height * roi_top_frac)
    strip_height = max(1, (img_height - strip_top) // num_strips)
    max_cluster_width = img_width * MAX_CLUSTER_WIDTH_FRAC

    weighted_sum = 0.0
    weight_total = 0.0
    debug_points = []           # (x, y) de los centros aceptados, para el overlay
    wide_strip_count = 0
    valid_strip_count = 0
    search_x = prev_x           # posicion esperada del dash; se ajusta por franja

    # Componentes conectados sobre TODO el ROI. Descartamos los blobs mas
    # anchos que max_cluster_width: un dash de carril es angosto, mientras que
    # cebras, lineas de alto y marcas de interseccion son anchas.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (white_roi > 0).astype(np.uint8), connectivity=8
    )
    good_labels = set()
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_WIDTH] <= max_cluster_width:
            good_labels.add(lbl)
    good_mask = np.isin(labels, list(good_labels)) if good_labels else None

    # Recorremos de abajo (cerca del auto, mas confiable) hacia arriba.
    for i in range(num_strips):
        y_bottom = img_height - i * strip_height
        y_top = max(strip_top, y_bottom - strip_height)
        if y_top >= y_bottom:
            continue

        strip = white_roi[y_top:y_bottom, :]
        white_cols = np.where(np.sum(strip > 0, axis=0) > 0)[0]
        if len(white_cols) == 0:
            continue
        valid_strip_count += 1

        # Deteccion de cebra: usa la mascara cruda (antes de filtrar) para que
        # una franja de cebra cuente como "ancha" aunque luego se descarte.
        if len(white_cols) / float(img_width) > CROSSWALK_COVERAGE_FRAC:
            wide_strip_count += 1

        if good_mask is None:
            continue

        good_cols = np.where(np.any(good_mask[y_top:y_bottom, :], axis=0))[0]
        if len(good_cols) == 0:
            continue

        # Agrupamos good_cols en clusters contiguos (separacion > 5 px).
        clusters = []
        cluster_start = good_cols[0]
        prev_col = good_cols[0]
        for c in good_cols[1:]:
            if c - prev_col > 5:
                clusters.append((cluster_start, prev_col))
                cluster_start = c
            prev_col = c
        clusters.append((cluster_start, prev_col))
        candidate_centers = [(a + b) / 2.0 for (a, b) in clusters]

        # Sin estimacion previa, asumimos el dash en la zona izquierda.
        if search_x is None:
            search_x = img_width * 0.35

        best_cx = min(candidate_centers, key=lambda cx: abs(cx - search_x))

        # Rechazo por salto relativo (carteles al borde de la via)...
        if abs(best_cx - search_x) > MAX_STRIP_JUMP:
            continue
        # ...y por desviacion absoluta del carril (linea del carril opuesto o
        # marcas de cruce en intersecciones).
        if abs(best_cx - lane_target) > MAX_TARGET_DEVIATION:
            continue

        cy = (y_top + y_bottom) / 2.0
        # Las franjas de abajo (i chico) pesan mas: estan mas cerca del auto.
        weight = num_strips - i
        weighted_sum += best_cx * weight
        weight_total += weight
        debug_points.append((int(best_cx), int(cy)))
        search_x = best_cx       # ancla para la siguiente franja de arriba

    crosswalk_detected = (
        valid_strip_count > 0 and
        (wide_strip_count / float(valid_strip_count)) >= CROSSWALK_STRIP_FRACTION
    )
    if weight_total == 0 or crosswalk_detected:
        return None, debug_points, crosswalk_detected
    return weighted_sum / weight_total, debug_points, crosswalk_detected


# ============================================================
# CONTROLADOR PID
# ============================================================
class PIDController:
    def __init__(self, Kp=0.01, Ki=0.000001, Kd=0.03, max_output=0.5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, error):
        if error is None:
            return 0.0
        P = self.Kp * error
        self.integral += error
        # Limita el integral para evitar windup en huecos/intersecciones largos.
        self.integral = max(-50.0, min(50.0, self.integral))
        I = self.Ki * self.integral
        D = self.Kd * (error - self.previous_error)
        self.previous_error = error
        output = P + I + D
        return max(-self.max_output, min(self.max_output, output))

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0


# ============================================================
# GRABACION DEL DATASET
# Guarda frames de la camara + el angulo aplicado en cada uno, para entrenar
# un modelo end-to-end (clonacion de comportamiento). Imagenes en
# dataset/images/, etiquetas en dataset/labels.csv.
# ============================================================
def init_dataset(base_dir=DATASET_DIR):
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "labels.csv")
    # Modo append: varias corridas se acumulan en el mismo dataset. El
    # encabezado solo se escribe cuando el archivo es nuevo.
    new_file = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["timestamp", "sim_time", "image", "steering",
                         "speed", "command"])
        csv_file.flush()
    return images_dir, csv_file, writer


def save_sample(images_dir, writer, csv_file, image, steering, speed,
                command, sim_time):
    # El nombre del archivo es el timestamp (con microsegundos), asi coincide
    # con la columna 'timestamp'. Es unico en la practica porque cada frame se
    # guarda en un instante de reloj distinto. DATASET_TAG (si no esta vacio) se
    # antepone para no colisionar al juntar datos de varias personas.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if DATASET_TAG:
        stamp = f"{DATASET_TAG}_{stamp}"
    filename = f"{stamp}.png"
    # La imagen viene en BGRA; quitamos el alfa para guardarla como BGR normal.
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # ROI del dataset: mitad inferior + recorte derecho inclinado.
    roi_img = apply_dataset_roi(frame_bgr)
    cv2.imwrite(os.path.join(images_dir, filename), roi_img)

    writer.writerow([stamp, f"{sim_time:.3f}", filename,
                     f"{steering:.5f}", speed, command])
    # flush por fila: Webots puede cortar el controlador de golpe y no queremos
    # perder muestras que sigan en el buffer.
    csv_file.flush()


def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # GPS para mostrar la posicion del vehiculo en el display. El nodo GPS del
    # mundo no tiene campo 'name', asi que su nombre por defecto es "gps".
    gps = robot.getDevice("gps")
    gps.enable(timestep)

    # El Display del mundo no tiene 'name', su nombre por defecto es "display".
    display_img = Display("display")
    # Display a la misma resolucion que la camara; escalamos el ROI a este tamano.
    disp_w = display_img.getWidth()
    disp_h = display_img.getHeight()

    # Kd bajo: como el error ya viene suavizado (EMA), no hace falta mucha
    # accion derivativa, que ademas provocaba oscilacion.
    pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.015, max_output=MAX_ANGLE)

    cam_width = camera.getWidth()
    cam_height = camera.getHeight()

    # Objetivo: posicion en la imagen donde se mantiene la linea blanca
    # (borde izquierdo del carril). Se controla con WHITE_TARGET_FRAC.
    white_target = cam_width * WHITE_TARGET_FRAC

    # Ultima x conocida del dash: ancla de busqueda y respaldo en huecos cortos.
    last_dash_x = white_target
    last_error = 0.0
    smoothed_error = 0.0
    frames_without_lines = 0
    current_speed = 0

    print(f"Camara: {cam_width}x{cam_height}, white_target: {white_target:.1f}")

    # Grabacion del dataset (imagen + angulo + comando). Apagada hasta 'g'.
    images_dir, csv_file, csv_writer = init_dataset()
    last_save_time = -1.0       # tiempo (sim) de la ultima muestra guardada
    recording = False
    g_was_down = False          # flanco de la tecla 'g'
    q_was_down = False          # flanco de la tecla 'q'

    # Modo de conduccion: "LANE" = seguidor de linea (PID); "MANUAL" = operador
    # con teclado + comandos de navegacion (estilo Codevilla / CIL).
    drive_mode = "LANE"
    manual_angle = 0.0          # angulo de direccion en modo manual
    manual_speed = 0            # velocidad con signo en modo manual (km/h; <0 = reversa)
    last_speed_step = -1.0      # tiempo (sim) del ultimo escalon de velocidad (antirrebote)
    nav_command = CMD_FOLLOW    # comando de navegacion actual

    print(f"Carpeta del dataset: {os.path.abspath(DATASET_DIR)}")
    print("Teclas: 'r' graba/detiene | 'q' cambia de modo (LANE/MANUAL)")
    print("Modo MANUAL: Izq/Der=giro suave | 'a'/'d'=vuelta cerrada izq/der | "
          "Arriba=acelerar, Abajo=frenar/reversa, Espacio=freno | 'c'=cruce recto")

    while robot.step() != -1:
        # ---- Lectura de teclado: juntamos todas las teclas de este frame ----
        keys = []
        k = keyboard.getKey()
        while k != -1:
            keys.append(k)
            k = keyboard.getKey()

        # Toggle de grabacion con 'r' (por flanco: un toque = un cambio).
        g_down = ord('R') in keys
        if g_down and not g_was_down:
            recording = not recording
            if recording:
                last_save_time = -1.0   # captura ya la primera muestra
                print(">>> Grabacion INICIADA")
            else:
                print(">>> Grabacion DETENIDA")
        g_was_down = g_down

        # Toggle de modo de conduccion con 'q' (por flanco).
        q_down = ord('Q') in keys
        if q_down and not q_was_down:
            drive_mode = "MANUAL" if drive_mode == "LANE" else "LANE"
            manual_angle = 0.0           # arrancamos centrados al cambiar
            manual_speed = 0             # y detenidos (evita un arranque brusco)
            print(">>> Modo: " +
                  ("MANUAL" if drive_mode == "MANUAL" else "AUTONOMO"))
        q_was_down = q_down

        # Direccion manual: flechas Izq/Der = giro suave (tope MANUAL_MAX_ANGLE);
        # A / D = vuelta cerrada (tope MANUAL_SHARP_ANGLE, mas angulo y mas rapido)
        # para tomar una esquina. Las teclas cerradas tienen prioridad sobre las
        # flechas. Auto-centrado al soltar todo.
        sharp_left = ord('A') in keys
        sharp_right = ord('D') in keys
        if sharp_left:
            manual_angle -= SHARP_ANGLE_INCR
        elif sharp_right:
            manual_angle += SHARP_ANGLE_INCR
        elif Keyboard.LEFT in keys:
            manual_angle -= ANGLE_INCR
        elif Keyboard.RIGHT in keys:
            manual_angle += ANGLE_INCR
        else:
            manual_angle *= ANGLE_DECAY
        # Tope amplio salvo cuando se pide giro suave con las flechas; asi un
        # angulo cerrado decae suave al soltar (sin saltos bruscos al centro).
        soft = ((Keyboard.LEFT in keys or Keyboard.RIGHT in keys)
                and not (sharp_left or sharp_right))
        limit = MANUAL_MAX_ANGLE if soft else MANUAL_SHARP_ANGLE
        manual_angle = max(-limit, min(limit, manual_angle))

        # Comando de navegacion (se guarda en el CSV):
        #   'c'         = cruzando la interseccion en recto (STRAIGHT). Antes era
        #                 la flecha ARRIBA, que ahora es el acelerador.
        #   Izq / Der   = giro en la interseccion (LEFT / RIGHT)
        #   sin tecla   = siguiendo el carril (FOLLOW)
        # (En modo LANE se fuerza a SEGUIR mas abajo.)
        if ord('C') in keys:
            nav_command = CMD_STRAIGHT
        elif Keyboard.LEFT in keys or ord('A') in keys:
            nav_command = CMD_LEFT
        elif Keyboard.RIGHT in keys or ord('D') in keys:
            nav_command = CMD_RIGHT
        else:
            nav_command = CMD_FOLLOW

        image = get_image(camera)
        white_mask = detect_white(image)
        white_roi = apply_roi(white_mask)

        if drive_mode == "LANE":
            # ---- Seguidor de linea (PID) ----
            dash_x, _, _ = track_white_dash(
                white_roi, last_dash_x, cam_width, cam_height, white_target
            )

            raw_error = None
            if dash_x is not None:
                raw_error = dash_x - white_target
                last_dash_x = dash_x

            if raw_error is not None:
                if frames_without_lines >= COAST_FRAMES_CURVE:
                    pid.reset()
                    smoothed_error = raw_error   # reenganche tras hueco largo
                else:
                    # Media movil exponencial: suaviza el ruido entre frames.
                    smoothed_error = (ERROR_EMA_ALPHA * raw_error +
                                      (1 - ERROR_EMA_ALPHA) * smoothed_error)
                last_error = smoothed_error
                frames_without_lines = 0
                current_speed = SPEED_STRAIGHT
            else:
                frames_without_lines += 1
                if frames_without_lines < COAST_FRAMES_CURVE:
                    # Hueco corto (gap del dash o ruido): mantenemos el error.
                    smoothed_error = last_error
                    current_speed = SPEED_STRAIGHT - SPEED_INCR
                else:
                    # Hueco largo (probable interseccion): recto y mas lento.
                    smoothed_error = 0.0
                    last_error = 0.0
                    last_dash_x = white_target   # reinicia el ancla
                    current_speed = SPEED_INTERSECTION

            steering = pid.compute(smoothed_error / cam_width)
            nav_command = CMD_FOLLOW         # el seguidor siempre "sigue carril"
        else:
            # ---- Conduccion manual (recoleccion estilo Codevilla / CIL) ----
            steering = manual_angle

            # Acelerador (ARRIBA) y freno/reversa (ABAJO). La velocidad lleva
            # signo: al frenar desde adelante baja hasta 0 y, si se sigue
            # pulsando ABAJO, cruza a negativo = reversa. setCruisingSpeed con
            # velocidad negativa engancha la reversa de forma automatica.
            # Antirrebote por tiempo: cada SPEED_DEBOUNCE segundos un escalon,
            # asi mantener la flecha sube/baja la velocidad de forma gradual.
            now = robot.getTime()
            if now - last_speed_step >= SPEED_DEBOUNCE:
                if Keyboard.UP in keys and manual_speed < MANUAL_MAX_SPEED:
                    manual_speed += MANUAL_SPEED_INCR
                    last_speed_step = now
                elif Keyboard.DOWN in keys and manual_speed > MANUAL_MAX_REVERSE:
                    manual_speed -= MANUAL_SPEED_INCR
                    last_speed_step = now
            # Freno de emergencia: la barra espaciadora detiene el auto al instante.
            if ord(' ') in keys:
                manual_speed = 0
            current_speed = manual_speed     # con signo (no se entrena)

        # Actuacion
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(current_speed)

        # ---- Captura del dataset (mas muestras en curva) ----
        # Solo si la grabacion esta activa. Muestreo por tiempo: 5/seg en recta
        # y 10/seg en curva, para no llenar el dataset de frames casi rectos.
        if recording:
            now = robot.getTime()
            is_curve = abs(steering) >= CURVE_STEER_THRESHOLD
            sample_period = SAMPLE_PERIOD_CURVE if is_curve else SAMPLE_PERIOD_STRAIGHT
            if now - last_save_time >= sample_period:
                save_sample(images_dir, csv_writer, csv_file, image,
                            steering, current_speed, nav_command, now)
                last_save_time = now

        # ---- Display: imagen recortada por el ROI (lo que se guarda) ----
        # Recortamos el mismo ROI del dataset (mitad inferior) del frame de la
        # camara y lo escalamos al tamano del display.
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        display_output = cv2.resize(apply_dataset_roi(frame_bgr), (disp_w, disp_h))

        # Modo activo (arriba a la izquierda).
        modo_txt = "MANUAL" if drive_mode == "MANUAL" else "AUTONOMO"
        cv2.putText(display_output, f"MODO: {modo_txt}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if drive_mode == "MANUAL":
            cv2.putText(display_output, f"CMD: {CMD_NAMES[nav_command]}", (6, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Velocidad con indicador [R] de reversa cuando es negativa.
            if current_speed < 0:
                vel_txt, vel_col = f"VEL: {current_speed} km/h  [R]", (0, 80, 255)
            else:
                vel_txt, vel_col = f"VEL: {current_speed} km/h", (0, 255, 255)
            cv2.putText(display_output, vel_txt, (6, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_col, 2)

        # Posicion GPS (x, y, z en metros) abajo a la izquierda, en ambos modos.
        px, py, pz = gps.getValues()
        if np.isnan(px):                 # primer paso, antes de la primera lectura
            gps_txt = "GPS  ---"
        else:
            gps_txt = f"GPS  x:{px:.1f}  y:{py:.1f}  z:{pz:.1f}"
        cv2.putText(display_output, gps_txt, (6, disp_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # Indicador REC: circulo rojo + texto arriba a la derecha al grabar.
        if recording:
            cv2.circle(display_output, (disp_w - 22, 20), 9, (0, 0, 255), -1)
            cv2.putText(display_output, "REC", (disp_w - 70, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        display_image_bgr(display_img, display_output)


if __name__ == "__main__":
    main()
