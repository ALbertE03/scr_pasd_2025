Write-Host "=== Desactivando Cluster Ray con Docker Compose ===" -ForegroundColor Green

try {
    docker version | Out-Null
} catch {
    Write-Host "Error: Docker no está ejecutándose o no está instalado." -ForegroundColor Red
    exit 1
}
Write-Host "Verificando contenedores del cluster Ray..." -ForegroundColor Yellow

Write-Host "Desactivando contenedores del docker-compose principal..." -ForegroundColor Yellow
docker-compose down

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al desactivar las imágenes Docker del docker-compose principal." -ForegroundColor Red
    Write-Host "Continuando con la limpieza de otros contenedores..." -ForegroundColor Yellow
}

Write-Host "Buscando y deteniendo workers externos añadidos manualmente..." -ForegroundColor Yellow
$externalWorkers = docker ps -a --filter "name=ray_worker" --format "{{.Names}}"

if ($externalWorkers) {
    Write-Host "Deteniendo los siguientes workers externos:" -ForegroundColor Yellow
    foreach ($worker in $externalWorkers) {
        Write-Host "  - $worker" -ForegroundColor White
        docker stop $worker
        docker rm $worker
    }
    Write-Host "Workers externos detenidos y eliminados exitosamente." -ForegroundColor Green
} else {
    Write-Host "No se encontraron workers externos adicionales." -ForegroundColor Cyan
}

if (Test-Path -Path "docker-compose.external.yml") {
    Write-Host "Desactivando contenedores de docker-compose externos..." -ForegroundColor Yellow
    docker-compose -f docker-compose.external.yml down
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error al desactivar las imágenes Docker de docker-compose externo." -ForegroundColor Red
    } else {
        Write-Host "Contenedores externos desactivados exitosamente." -ForegroundColor Green
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Cluster Ray desactivado exitosamente!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=== Resumen de la operación ===" -ForegroundColor Cyan
$remainingContainers = docker ps -a --filter "name=ray" --format "{{.Names}}"

if ($remainingContainers) {
    Write-Host "⚠️ Advertencia: Aún quedan los siguientes contenedores relacionados con Ray:" -ForegroundColor Yellow
    foreach ($container in $remainingContainers) {
        Write-Host "  - $container" -ForegroundColor White
    }
    Write-Host "Puede detenerlos manualmente con 'docker stop <nombre>' y 'docker rm <nombre>'." -ForegroundColor Yellow
} else {
    Write-Host "✅ Todos los contenedores relacionados con Ray han sido detenidos y eliminados." -ForegroundColor Green
}