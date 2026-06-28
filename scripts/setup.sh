#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="nav"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Setup — Entrenamiento Behavioral Cloning ==="
echo "Proyecto : $PROJECT_ROOT"
echo "Entorno  : $CONDA_ENV"
echo ""

# Activar conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python : $(python --version)"
echo ""

# Instalar dependencias
pip install -r "$PROJECT_ROOT/requirements.txt"

echo ""
echo "Setup completado."
