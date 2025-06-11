# Script de PowerShell para manejar el proyecto ML distribuido
# Uso: .\manage.ps1 [start|stop|status|logs|cleanup|build|test-fault-tolerance]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "logs", "cleanup", "build", "test-fault-tolerance")]
    [string]$Action
)

function Show-Usage {
    Write-Host "Uso: .\manage.ps1 [start|stop|status|logs|cleanup|build|test-fault-tolerance]"
    Write-Host ""
    Write-Host "Comandos disponibles:"
    Write-Host "  start                - Iniciar todos los servicios"
    Write-Host "  stop                 - Detener todos los servicios"
    Write-Host "  status               - Mostrar estado de los servicios"
    Write-Host "  logs                 - Mostrar logs de los servicios"
    Write-Host "  build                - Construir las imágenes Docker"
    Write-Host "  cleanup              - Limpiar recursos Docker no utilizados"
    Write-Host "  test-fault-tolerance - Ejecutar pruebas de tolerancia a fallos"
}

function Start-Services {
    Write-Host "🚀 Iniciando servicios ML distribuidos..." -ForegroundColor Green
    Write-Host "📊 Recursos optimizados para laptops con pocos recursos" -ForegroundColor Yellow
    
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Servicios iniciados correctamente!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Accede a los servicios en:" -ForegroundColor Cyan
        Write-Host "  • Ray Dashboard: http://localhost:8265" -ForegroundColor White
        Write-Host "  • Model Server: http://localhost:8000" -ForegroundColor White
        Write-Host "  • Streamlit App: http://localhost:8501" -ForegroundColor White
        Write-Host ""
        Write-Host "📝 Para ver logs: .\manage.ps1 logs" -ForegroundColor Gray
    } else {
        Write-Host "❌ Error al iniciar servicios" -ForegroundColor Red
    }
}

function Stop-Services {
    Write-Host "🛑 Deteniendo servicios..." -ForegroundColor Yellow
    docker-compose down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Servicios detenidos correctamente!" -ForegroundColor Green
    } else {
        Write-Host "❌ Error al detener servicios" -ForegroundColor Red
    }
}

function Show-Status {
    Write-Host "📊 Estado de los servicios:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host "`n💾 Uso de recursos:" -ForegroundColor Cyan
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
}

function Show-Logs {
    Write-Host "📜 Mostrando logs de los servicios..." -ForegroundColor Cyan
    docker-compose logs --tail=50 -f
}

function Build-Images {
    Write-Host "🔨 Construyendo imágenes Docker..." -ForegroundColor Yellow
    docker-compose build --no-cache
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Imágenes construidas correctamente!" -ForegroundColor Green
    } else {
        Write-Host "❌ Error al construir imágenes" -ForegroundColor Red
    }
}

function Cleanup-Docker {
    Write-Host "🧹 Limpiando recursos Docker no utilizados..." -ForegroundColor Yellow
    
    Write-Host "Eliminando contenedores detenidos..." -ForegroundColor Gray
    docker container prune -f
    
    Write-Host "Eliminando imágenes sin usar..." -ForegroundColor Gray
    docker image prune -f
    
    Write-Host "Eliminando volúmenes sin usar..." -ForegroundColor Gray
    docker volume prune -f
    
    Write-Host "Eliminando redes sin usar..." -ForegroundColor Gray
    docker network prune -f
    
    Write-Host "✅ Limpieza completada!" -ForegroundColor Green
}

function Test-FaultTolerance {
    Write-Host "🧪 Ejecutando pruebas de tolerancia a fallos..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opciones de prueba disponibles:" -ForegroundColor Cyan
    Write-Host "1. Prueba básica (eliminar un worker)" -ForegroundColor White
    Write-Host "2. Prueba de fallos escalonados" -ForegroundColor White
    Write-Host "3. Prueba de estrés con fallos aleatorios" -ForegroundColor White
    Write-Host "4. Todas las pruebas" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "Selecciona una opción (1-4)"
    
    switch ($choice) {
        "1" { 
            Write-Host "Ejecutando prueba básica..." -ForegroundColor Green
            .\test-fault-tolerance.ps1 -TestType basic
        }
        "2" { 
            Write-Host "Ejecutando prueba de fallos escalonados..." -ForegroundColor Green
            .\test-fault-tolerance.ps1 -TestType worker-failure
        }
        "3" { 
            Write-Host "Ejecutando prueba de estrés..." -ForegroundColor Green
            $duration = Read-Host "Duración de la prueba en segundos (default: 300)"
            if ([string]::IsNullOrEmpty($duration)) { $duration = 300 }
            .\test-fault-tolerance.ps1 -TestType stress-test -Duration $duration
        }
        "4" { 
            Write-Host "Ejecutando todas las pruebas..." -ForegroundColor Green
            .\test-fault-tolerance.ps1 -TestType all
        }
        default { 
            Write-Host "Opción inválida. Ejecutando prueba básica..." -ForegroundColor Yellow
            .\test-fault-tolerance.ps1 -TestType basic
        }
    }
    
    Write-Host ""
    Write-Host "💡 También puedes ejecutar el script de Python para pruebas más avanzadas:" -ForegroundColor Cyan
    Write-Host "docker exec -it `$(docker ps -q -f `"name=ray-head`") python fault_tolerance_test.py --duration 10 --trainers 6" -ForegroundColor White
}

# Verificar que Docker esté disponible
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker no está instalado o no está en el PATH" -ForegroundColor Red
    exit 1
}

# Ejecutar acción
switch ($Action) {
    "start" { Start-Services }
    "stop" { Stop-Services }
    "status" { Show-Status }
    "logs" { Show-Logs }
    "build" { Build-Images }
    "cleanup" { Cleanup-Docker }
    "test-fault-tolerance" { Test-FaultTolerance }
    default { Show-Usage }
}
