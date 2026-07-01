<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white" alt="Python 3.14"/>
  <img src="https://img.shields.io/badge/Modelo-Nvidia%20Model-orange?style=flat"/>
  <img src="https://img.shields.io/badge/GPU-RTX%205080-76b900?style=flat&logo=nvidia&logoColor=white" alt="Nvidia RTX"/>
</p>

---

<h2 align="center">Proyecto Final — Equipo 01</h2>

<table align="center">
  <thead>
    <tr>
      <th>Nombre</th>
      <th>Matrícula</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Luis Gabriel Lobato Barajas</td><td><code>A01797069</code></td></tr>
    <tr><td>Ángel Eduardo Pérez Cruz</td><td><code>A01797661</code></td></tr>
    <tr><td>Juan Carlos Pérez Nava</td><td><code>A01795941</code></td></tr>
    <tr><td>Israel Sánchez Arenas</td><td><code>A01797385</code></td></tr>
  </tbody>
</table>

---

# Entrenamiento Behavioral Cloning — NVIDIA Model

Proyecto de **behavioral cloning** para conducción autónoma en simulador Webots. El dataset se genera conduciendo manualmente y el modelo aprende a replicar el comportamiento a partir de imágenes de cámara.

## Estructura del proyecto

```
Entrenamiento_behavioralC/
├── dataset/
│   ├── images/              # Fotogramas capturados desde Webots (~27 000 imágenes)
│   └── labels.csv           # Etiquetas: timestamp, sim_time, imagen, steering, speed, command
├── notebooks/
│   └── nvidia_model.ipynb   # Pipeline completo de entrenamiento (exploración)
├── scripts/
│   ├── setup.sh             # Instala dependencias en el entorno conda
│   └── run_train.sh         # Verifica el dataset y lanza el entrenamiento
├── src/
│   ├── model.py             # Arquitectura NVIDIA
│   ├── train.py             # Script principal de entrenamiento
│   └── utils.py             # Carga, augmentación y preprocesamiento
├── mundos/
│   ├── city_01/             # Escenario sin tráfico — usado para grabación del dataset
│   │   └── city_traffic_2025_01.wbt
│   └── city_02/             # Escenario con tráfico SUMO (vehículos en movimiento)
│       ├── city_traffic_2025_02.wbt
│       ├── city_traffic_2025_02_net/  # Red de tráfico SUMO (.xml, .sumocfg)
│       └── protos/          # Modelo BMWX5
├── manual_controller.py     # Controlador Webots: modos AUTÓNOMO (PID) y MANUAL + grabación
├── model_controller.py      # Controlador autónomo: CNN + evasión + ACC + semáforo + peatón
└── requirements.txt
```

## Dataset

El dataset es generado por el controlador de Webots utilizando el escenario "city_traffic_2025_01”, el cual carece de elementos de tráfico, peatones y vehículos detenidos en las carreteras.

<table>
  <thead>
    <tr>
      <th>Campo</th>
      <th>Tipo</th>
      <th>Descripción</th>
      <th>Usado en entrenamiento</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>timestamp</code></td>
      <td>string</td>
      <td>Identificador único del fotograma</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td><code>sim_time</code></td>
      <td>float</td>
      <td>Tiempo de simulación en segundos</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td><code>image</code></td>
      <td>string</td>
      <td>Nombre del archivo PNG en <code>dataset/images/</code></td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td><code>steering</code></td>
      <td>float</td>
      <td>Ángulo de dirección — variable objetivo (regresión)</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td><code>speed</code></td>
      <td>int</td>
      <td>Velocidad del vehículo en la simulación</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td><code>command</code></td>
      <td>int</td>
      <td>Comando de conducción activo</td>
      <td align="center">—</td>
    </tr>
  </tbody>
</table>

## Controlador Manual (`manual_controller.py`)

El controlador corre dentro de Webots y es el encargado de generar el dataset.

### Teclas

<!--| `q` | Cambiar entre modo AUTÓNOMO y MANUAL |-->

| Tecla | Acción |
|-------|--------|
| `r` | Iniciar / detener grabación del dataset |
| `←` / `→` | Giro suave (tope ±0.1 rad) |
| `a` / `d` | Vuelta cerrada (tope ±0.5 rad) |
| `↑` / `↓` | Acelerar / frenar–reversa |
| `Espacio` | Aplicar freno |

### Grabación del dataset

La grabación se activa con **`r`** y guarda cada muestra en `dataset/images/` + una fila en `dataset/labels.csv`. Para no saturar el dataset con frames casi rectos, la cadencia varía según el ángulo aplicado:

| Situación | Cadencia |
|-----------|----------|
| Recta (`\|steering\|` < 0.04 rad) | 1 muestra cada 0.2 s (~5/seg) |
| Curva (`\|steering\|` ≥ 0.04 rad) | 1 muestra cada 0.1 s (~10/seg) |

Cada imagen se recorta con un ROI que elimina el cielo y los edificios (mitad superior de la cámara), dejando solo la zona relevante de la vía. Las corridas de distintos operadores se acumulan en el mismo CSV sin colisiones de nombre gracias a la etiqueta `DATASET_TAG`.

## Controlador Autónomo (`model_controller.py`)

Este controlador corre dentro de Webots y conduce el vehículo de forma completamente autónoma combinando cuatro comportamientos con una jerarquía de prioridad clara:

| Prioridad | Comportamiento | Descripción |
|:---------:|----------------|-------------|
| 1 | **Freno de emergencia** | Si la cámara detecta un peatón y el radar lo confirma a ≤ 7 m, frena al máximo |
| 2 | **Semáforo** | Detecta luz roja vía API de reconocimiento + HSV; reduce velocidad al acercarse y frena en seco hasta ver verde |
| 3 | **Evasión de bus** | Al detectar un autobús parado, ejecuta una maniobra de rebase (sale del carril, adelanta y regresa) |
| 4 | **CNN** | La red NVIDIA predice el ángulo de dirección a partir de la imagen de cámara — modo de conducción normal |
| — | **ACC** | Control de crucero adaptativo: sigue al vehículo de delante manteniendo una separación de seguridad de 5 m |

### Modos de operación

| Modo | Cuándo se activa |
|------|-----------------|
| `CNN` | Conducción normal — la red neuronal controla el volante |
| `HOLD` | El bus se detecta a ≤ 15 m — el coche se endereza y espera antes de evadir |
| `EVASION` | Maniobra activa de rebase (con subfases) |
| `SEMAFORO` | Luz roja confirmada — frena hasta ver verde o hasta timeout de 5 s |
| Freno peatón | Máxima prioridad — corta acelerador y aplica freno directo |

### Sensores utilizados

| Sensor | Uso |
|--------|-----|
| Cámara + Recognition API | Imagen para la CNN, detección de bus / peatón / semáforo / vehículos |
| LiDAR Sick LMS 291 | Distancia frontal para disparar la evasión y el ACC |
| Radar | Confirmación de distancia al peatón (más fiable a corta distancia) |
| Sensores de distancia lateral | Control de separación durante la maniobra de rebase |
| Giroscopio | Velocidad angular (eje Z) para guiar la evasión |

### Telemetría en pantalla

El controlador pinta tres paneles de telemetría en los displays del tablero de Webots:

- **Display estado** — modo activo (CNN / HOLD / EVASION / SEMAFORO), barras de volante, velocidad y freno
- **Display LiDAR** — vista de pájaro del barrido frontal + lecturas de sensores laterales y giroscopio
- **Display radar** — blancos detectados, velocidad real vs objetivo del ACC y objetos reconocidos

## Clonar Repositorio

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jarcos09/Entrenamiento_behavioralC.git
cd Entrenamiento_behavioralC

# 2. Instalar dependencias en el entorno conda (nav)
bash scripts/setup.sh
```

## Uso

```bash
# Lanzar entrenamiento
bash scripts/run_train.sh
```
El modelo entrenado se guarda como `nvidia_model.keras` en la raíz del proyecto.

## Pipeline de entrenamiento

1. **Carga y verificación** del dataset generado en Webots
2. **Balanceo** — submuestreo por bins para evitar sesgo hacia `steering = 0`
3. **Data augmentation** — zoom, pan, brillo aleatorio y flip horizontal
4. **Preprocesamiento** — blur gaussiano, conversión RGB → YUV, normalización a `[-1, 1]`
5. **Entrenamiento** — Adam (lr=1e-4), MSE, con EarlyStopping y ReduceLROnPlateau
6. **Exportación** — modelo guardado como `nvidia_model.keras`

### Arquitectura NVIDIA

```
Input (76 × 320 × 3)
  → Conv2D 24 (5×5, s=2) ELU
  → Conv2D 36 (5×5, s=2) ELU
  → Conv2D 48 (5×5, s=2) ELU
  → Conv2D 64 (3×3)      ELU
  → Conv2D 64 (3×3)      ELU
  → Flatten
  → Dense 100 ELU + Dropout 0.5
  → Dense 50  ELU + Dropout 0.5
  → Dense 10  ELU
  → Dense 1   (steering)
```

Total de parámetros: **559 419** (~2.1 MB)

## Dependencias

```
torch
keras (backend PyTorch)
opencv-python
pandas
scikit-learn
imgaug
matplotlib
numpy
```

## Requisitos de hardware

Entrenado con **NVIDIA GeForce RTX 5080**. El notebook y el script funcionan en CPU pero el entrenamiento será considerablemente más lento.

---

## Declaración de uso de Inteligencia Artificial

**Herramienta utilizada:**

> Anthropic. (2026). *Claude* (claude-sonnet-4-6) \[Modelo de lenguaje grande\]. Utilizado para refinamiento de código Python.
> https://claude.ai
