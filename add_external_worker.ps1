Write-Host "=== Añadiendo Worker Externo al Cluster Ray ===" -ForegroundColor Green

# Verificar si Docker está ejecutándose
try {
    docker version | Out-Null
} catch {
    Write-Host "Error: Docker no está ejecutándose o no está instalado." -ForegroundColor Red
    exit 1
}

# Solicitar nombre del worker si no se proporciona como parámetro
param(
    [string]$WorkerName
)

if (-not $WorkerName) {
    $WorkerName = Read-Host "Nombre para el worker externo (dejar en blanco para usar 'ray-external-worker')"
    if (-not $WorkerName) {
        $WorkerName = "ray-external-worker"
    }
}

Write-Host "Añadiendo worker externo: $WorkerName" -ForegroundColor Yellow

# Usar la variable de entorno para el docker-compose
$env:WORKER_NAME = $WorkerName

# Asegurarse de utilizar el executable completo de docker-compose
$dockerComposeCommand = "docker-compose"

# Verificar que docker-compose existe usando Get-Command
if (Get-Command $dockerComposeCommand -ErrorAction SilentlyContinue) {
    # Intento de ejecutar docker-compose con el archivo específico
    & $dockerComposeCommand -f docker-compose.external.yml up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Worker externo '$WorkerName' añadido exitosamente" -ForegroundColor Green
        
        # Mostrar estado actualizado
        Write-Host ""
        Write-Host "Estado de los contenedores:" -ForegroundColor Cyan
        & $dockerComposeCommand ps
        
        Write-Host ""
        Write-Host "Para ver logs del worker: docker logs $WorkerName" -ForegroundColor Yellow
    } else {
        Write-Host "[ERROR] Error al añadir worker externo." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[ERROR] No se encontró el comando docker-compose en el sistema." -ForegroundColor Red
    Write-Host "Asegúrate de que Docker Desktop está instalado y que docker-compose está disponible en el PATH." -ForegroundColor Yellow
    exit 1
}
