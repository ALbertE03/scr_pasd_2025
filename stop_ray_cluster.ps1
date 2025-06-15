Write-Host "=== Desactivando Cluster Ray con Docker Compose ===" -ForegroundColor Green

try {
    docker version | Out-Null
} catch {
    Write-Host "Error: Docker no está ejecutándose o no está instalado." -ForegroundColor Red
    exit 1
}
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