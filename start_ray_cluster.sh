#!/bin/bash

echo "=== Iniciando Cluster Ray con Docker Compose ==="


if ! docker version &> /dev/null; then
    echo "Error: Docker no está ejecutándose o no está instalado."
    exit 1
fi

echo "Construyendo imágenes Docker..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "Error al construir las imágenes Docker."
    exit 1
fi

echo "Iniciando cluster Ray..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "[OK] Cluster Ray iniciado exitosamente!"
    echo ""

    echo "Esperando a que los servicios se inicialicen..."
    sleep 15
    
    echo "Estado de los contenedores:"
    docker-compose ps
    
    echo ""
    echo "=== Información del Cluster ==="
    echo "🎯 Ray Dashboard: http://localhost:8265"
    echo "🎨 Streamlit App: http://localhost:8501"
    echo "🔌 Ray Head: localhost:10001"
    echo "📡 GCS Port: localhost:6379"
    echo "🌐 Aplicación web disponible en http://localhost:8501"
    echo ""
    echo "Para añadir workers externos, use: ./add_external_worker.sh"
    echo "Para ver logs: docker-compose logs -f"
    echo "Para detener: ./stop_ray_cluster.sh"
else
    echo "[ERROR] Error al iniciar el cluster."
    exit 1
fi
