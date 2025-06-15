Write-Host "=== Iniciando Cluster Ray con Docker Compose ===" -ForegroundColor Green

try {
    docker version | Out-Null
} catch {
    Write-Host "Error: Docker no está ejecutándose o no está instalado." -ForegroundColor Red
    exit 1
}
Write-Host "Construyendo imágenes Docker..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al construir las imágenes Docker." -ForegroundColor Red
    exit 1
}

Write-Host "Iniciando cluster Ray..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -eq 0) {    Write-Host "[OK] Cluster Ray iniciado exitosamente!" -ForegroundColor Green
    Write-Host ""
    
    # Esperar un momento para que los servicios se inicialicen
    Write-Host "Esperando a que los servicios se inicialicen..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    # Mostrar estado del cluster
    Write-Host "Estado de los contenedores:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "=== Información del Cluster ===" -ForegroundColor Green
    Write-Host "🎯 Ray Dashboard: http://localhost:8265" -ForegroundColor Cyan
    Write-Host "🎨 Streamlit App: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "🔌 Ray Head: localhost:10001" -ForegroundColor Cyan
    Write-Host "📡 GCS Port: localhost:6379" -ForegroundColor Cyan    
    Write-Host "🌐 Aplicación web disponible en http://localhost:8501" -ForegroundColor Yellow 
    Write-Host ""
    Write-Host "Para añadir workers externos, use: .\add_external_worker.ps1" -ForegroundColor Yellow
    Write-Host "Para ver logs: docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "Para detener: .\stop_ray_cluster.ps1" -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] Error al iniciar el cluster." -ForegroundColor Red
    exit 1
}
