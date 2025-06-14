# Script para ejecutar entrenamiento distribuido standalone
Write-Host "Ejecutando entrenamiento distribuido standalone..." -ForegroundColor Green

# Verificar si Ray está instalado
try {
    python -c "import ray; print('Ray version:', ray.__version__)"
} catch {
    Write-Host "Ray no está instalado. Instalando dependencias..." -ForegroundColor Red
    pip install -r requirements.txt
}

# Ejecutar entrenamiento
Write-Host "Iniciando entrenamiento..." -ForegroundColor Yellow
python train.py

Write-Host "Entrenamiento completado!" -ForegroundColor Green
Read-Host "Presiona Enter para continuar..."
