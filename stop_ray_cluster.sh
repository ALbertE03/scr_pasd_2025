#!/bin/bash

echo "=== Desactivando Cluster Ray con Docker Compose ==="


if ! docker version &> /dev/null; then
    echo "Error: Docker no está ejecutándose o no está instalado."
    exit 1
fi

echo "Desactivando imágenes Docker..."
docker-compose down

if [ $? -ne 0 ]; then
    echo "Error al desactivar las imágenes Docker."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "[OK] Cluster Ray desactivado exitosamente!"
    echo ""
fi
