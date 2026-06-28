import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_img_steering, batch_generator
from model import nvidia_model

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_PATH   = os.path.join(PROJECT_ROOT, 'nvidia_model.keras')

# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
NUM_BINS        = 25
SAMPLES_PER_BIN = 400
BATCH_SIZE      = 100
STEPS_PER_EPOCH = 300
EPOCHS          = 60

# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    print(f"GPU detectada : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ADVERTENCIA: No se detectó GPU — el entrenamiento será muy lento en CPU.")

print(f"Backend Keras : {keras.backend.backend()}")
print(f"PyTorch       : {torch.__version__}")

# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------
csv_path   = os.path.join(OUTPUT_DIR, 'labels.csv')
images_dir = os.path.join(OUTPUT_DIR, 'images')

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"No se encontró '{csv_path}'. Graba primero el dataset con el "
        f"controlador de Webots (tecla 'g')."
    )

full_dataset = pd.read_csv(csv_path)
full_dataset['image_path'] = full_dataset['image'].apply(
    lambda f: os.path.join(images_dir, f)
)
print(f"{len(full_dataset):,} registros cargados.")

data = full_dataset[['image_path', 'steering']].copy()
mask_ok   = data['image_path'].apply(os.path.exists)
n_missing = (~mask_ok).sum()
if n_missing:
    print(f"Advertencia: {n_missing} imágenes faltantes eliminadas del dataset.")
data = data[mask_ok].reset_index(drop=True)
print(f"Registros válidos: {len(data):,}")

# ---------------------------------------------------------------------------
# Histograma previo al balanceo
# ---------------------------------------------------------------------------
hist, bins = np.histogram(data['steering'], NUM_BINS)
center = (bins[:-1] + bins[1:]) * 0.5
width  = bins[1] - bins[0]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')
colors = ['#f38ba8' if h > SAMPLES_PER_BIN else '#89b4fa' for h in hist]
ax.bar(center, hist, width=width * 0.9, color=colors, edgecolor='#313244', linewidth=0.5)
ax.axhline(SAMPLES_PER_BIN, color='#a6e3a1', linewidth=2, linestyle='--')
ax.set_title('Distribución de Steering — antes del balanceo', color='white', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Ángulo de dirección', color='#cdd6f4', fontsize=11)
ax.set_ylabel('Número de muestras', color='#cdd6f4', fontsize=11)
ax.tick_params(colors='#cdd6f4')
for spine in ax.spines.values():
    spine.set_edgecolor('#313244')
ax.grid(axis='y', color='#313244', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(handles=[
    Patch(facecolor='#89b4fa', label='Dentro del límite'),
    Patch(facecolor='#f38ba8', label='Sobre-representado'),
    plt.Line2D([0], [0], color='#a6e3a1', linewidth=2, linestyle='--', label=f'Límite ({SAMPLES_PER_BIN}/bin)'),
], facecolor='#313244', labelcolor='#cdd6f4', fontsize=10)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Balanceo — submuestreo por bins
# ---------------------------------------------------------------------------
print(f"Total antes del balanceo : {len(data):,}")
remove_list = []
for j in range(NUM_BINS):
    mask        = (data['steering'] >= bins[j]) & (data['steering'] <= bins[j + 1])
    bin_indices = shuffle(data.index[mask].tolist())
    remove_list.extend(bin_indices[SAMPLES_PER_BIN:])

data.drop(index=remove_list, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"Registros eliminados     : {len(remove_list):,}")
print(f"Registros restantes      : {len(data):,}")

# ---------------------------------------------------------------------------
# Split train / validación
# ---------------------------------------------------------------------------
image_paths, steerings = load_img_steering(data)
X_train, X_valid, y_train, y_valid = train_test_split(
    image_paths, steerings, test_size=0.2, random_state=6
)
print(f"Training : {len(X_train):,}  |  Validación : {len(X_valid):,}")

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
model = nvidia_model()
model.summary()

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

val_steps = math.ceil(len(X_valid) / BATCH_SIZE)

history = model.fit(
    batch_generator(X_train, y_train, BATCH_SIZE, istraining=True),
    steps_per_epoch  = STEPS_PER_EPOCH,
    epochs           = EPOCHS,
    validation_data  = batch_generator(X_valid, y_valid, BATCH_SIZE, istraining=False),
    validation_steps = val_steps,
    callbacks        = callbacks,
    verbose          = 1,
)

# ---------------------------------------------------------------------------
# Gráfica de pérdida
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')

epoch_range = range(1, len(history.history['loss']) + 1)
ax.plot(epoch_range, history.history['loss'],     color='#89b4fa', linewidth=2, marker='o', markersize=4, label='Entrenamiento')
ax.plot(epoch_range, history.history['val_loss'], color='#f38ba8', linewidth=2, marker='o', markersize=4, label='Validación')

best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best_val   = min(history.history['val_loss'])
ax.axvline(best_epoch, color='#a6e3a1', linewidth=1.5, linestyle='--',
           label=f'Mejor epoch ({best_epoch})  val_loss={best_val:.4f}')

ax.set_title('Pérdida durante el Entrenamiento (MSE)', color='white', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Época', color='#cdd6f4', fontsize=11)
ax.set_ylabel('Loss (MSE)', color='#cdd6f4', fontsize=11)
ax.tick_params(colors='#cdd6f4')
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
for spine in ax.spines.values():
    spine.set_edgecolor('#313244')
ax.grid(color='#313244', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=10)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Guardar modelo final
# ---------------------------------------------------------------------------
model.save(MODEL_PATH)
print(f"Modelo guardado en '{MODEL_PATH}'")
print(f"Tamaño : {os.path.getsize(MODEL_PATH) / 1e6:.2f} MB")
