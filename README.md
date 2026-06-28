<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white" alt="Python 3.14"/>
  <img src="https://img.shields.io/badge/Modelo-Nvidia%20Model-orange?style=flat"/>
  <img src="https://img.shields.io/badge/GPU-RTX%205080-76b900?style=flat&logo=nvidia&logoColor=white" alt="Nvidia RTX"/>
</p>

---

# Entrenamiento Behavioral Cloning — NVIDIA Model

Proyecto de **behavioral cloning** para conducción autónoma en simulador Webots. El dataset se genera conduciendo manualmente y el modelo aprende a replicar el comportamiento a partir de imágenes de cámara.

## Estructura del proyecto

```
Entrenamiento_behavioralC/
├── dataset/
│   ├── images/          # Fotogramas capturados desde Webots (~27 000 imágenes)
│   └── labels.csv       # Etiquetas: timestamp, sim_time, imagen, steering, speed, command
├── notebooks/
│   └── nvidia_model.ipynb  # Pipeline completo de entrenamiento
└── src/                 # Scripts auxiliares (en desarrollo)
```

## Dataset

| Campo | Descripción |
|-------|-------------|
| `image` | Nombre del archivo PNG |
| `steering` | Ángulo de dirección (regresión continua) |
| `speed` | Velocidad de la simulación |
| `command` | Comando de conducción activo |

## Notebook

`notebooks/nvidia_model.ipynb` implementa el pipeline completo:

1. **Carga y verificación** del dataset generado en Webots
2. **Balanceo** — submuestreo por bins para evitar sesgo hacia `steering = 0`
3. **Data augmentation** — zoom, pan, brillo aleatorio y flip horizontal
4. **Preprocesamiento** — blur gaussiano, conversión RGB → YUV, normalización a `[-1, 1]`
5. **Modelo NVIDIA** — 5 capas convolucionales + 3 capas densas con Dropout
6. **Entrenamiento** — Adam (lr=1e-4), MSE, con EarlyStopping y ReduceLROnPlateau
7. **Exportación** — modelo guardado como `nvidia_model.keras`

### Arquitectura

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

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jarcos09/Entrenamiento_behavioralC.git
cd Entrenamiento_behavioralC

# 2. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** el dataset (`dataset/images/` y `dataset/labels.csv`) no está incluido en el repositorio. Debes grabarlo desde Webots con el controlador (tecla **`g`**) antes de ejecutar el notebook.

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

Entrenado con **NVIDIA GeForce RTX 5080** . El notebook funciona en CPU pero el entrenamiento será considerablemente más lento.
