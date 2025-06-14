# Script para iniciar el cluster Ray con Docker Compose
# Usage: .\start_ray_cluster.ps1

Write-Host "=== Iniciando Cluster Ray con Docker Compose ===" -ForegroundColor Green

# Verificar si Docker está ejecutándose
try {
    docker version | Out-Null
} catch {
    Write-Host "Error: Docker no está ejecutándose o no está instalado." -ForegroundColor Red
    exit 1
}

# Construir y iniciar el cluster
Write-Host "Desactivando imágenes Docker..." -ForegroundColor Yellow
docker-compose down

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al desactivar las imágenes Docker." -ForegroundColor Red
    exit 1
}



if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Cluster Ray desactivado exitosamente!" -ForegroundColor Green
    Write-Host ""
}