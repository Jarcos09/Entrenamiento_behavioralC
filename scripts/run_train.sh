#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="nav"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_CSV="$PROJECT_ROOT/dataset/labels.csv"
DATASET_IMG="$PROJECT_ROOT/dataset/images"

echo "=== Entrenamiento Behavioral Cloning — NVIDIA Model ==="
echo "Proyecto : $PROJECT_ROOT"
echo "Entorno  : $CONDA_ENV"
echo ""

# Verificar dataset antes de lanzar
if [[ ! -f "$DATASET_CSV" ]]; then
    echo "ERROR: No se encontró '$DATASET_CSV'."
    echo "  Graba primero el dataset desde Webots con la tecla 'g'."
    exit 1
fi

N_IMGS=$(ls "$DATASET_IMG" 2>/dev/null | wc -l)
echo "Dataset encontrado: $N_IMGS imágenes"
echo ""

# Activar conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python : $(python --version)"
echo ""

# Lanzar entrenamiento
cd "$PROJECT_ROOT"
python src/train.py
