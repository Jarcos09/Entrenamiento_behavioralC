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
