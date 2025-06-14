# Script de configuración completa del entorno
Write-Host "Configuración completa del entorno de ML distribuido" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green

# Función para verificar si un comando existe
function Test-Command($command) {
    try { Get-Command $command -ErrorAction Stop; return $true }
    catch { return $false }
}

# Verificar requisitos
Write-Host "Verificando requisitos..." -ForegroundColor Yellow

# Verificar Python
if (Test-Command python) {
    $pythonVersion = python --version
    Write-Host "$pythonVersion" -ForegroundColor Green
} else {
    Write-Host "Python no está instalado" -ForegroundColor Red
    exit 1
}

# Verificar Docker
if (Test-Command docker) {
    try {
        docker version | Out-Null
        Write-Host "Docker está disponible" -ForegroundColor Green
    } catch {
        Write-Host "Docker no está ejecutándose" -ForegroundColor Red
        Write-Host "Por favor inicia Docker Desktop" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Docker no está instalado" -ForegroundColor Red
    exit 1
}

# Verificar docker-compose
if (Test-Command docker-compose) {
    Write-Host "Docker Compose está disponible" -ForegroundColor Green
} else {
    Write-Host "Docker Compose no está instalado" -ForegroundColor Red
    exit 1
}

# Instalar dependencias Python
Write-Host ""
Write-Host "Instalando dependencias de Python..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "Dependencias instaladas correctamente" -ForegroundColor Green
} else {
    Write-Host "Error instalando dependencias" -ForegroundColor Red
    exit 1
}

# Crear directorios necesarios
Write-Host ""
Write-Host "Creando directorios necesarios..." -ForegroundColor Yellow
$directories = @("models", "results", "logs")

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir
        Write-Host "Directorio creado: $dir" -ForegroundColor Green
    } else {
        Write-Host "Directorio existe: $dir" -ForegroundColor Green
    }
}

# Mostrar opciones disponibles
Write-Host ""
Write-Host "Configuración completada! Opciones disponibles:" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green
Write-Host ""
Write-Host "CLUSTER DISTRIBUIDO (Docker):" -ForegroundColor Cyan
Write-Host "  .\start_ray_cluster.ps1    - Iniciar cluster completo" -ForegroundColor White
Write-Host "  .\stop_ray_cluster.ps1     - Detener cluster" -ForegroundColor White
Write-Host ""
Write-Host "EJECUCIÓN LOCAL:" -ForegroundColor Cyan
Write-Host "  .\run_training_standalone.ps1  - Entrenamiento local" -ForegroundColor White
Write-Host "  .\run_streamlit.ps1            - Interfaz Streamlit local" -ForegroundColor White
Write-Host ""
Write-Host "URLS IMPORTANTES:" -ForegroundColor Cyan
Write-Host "  Ray Dashboard:    http://localhost:8265" -ForegroundColor White
Write-Host "  Streamlit App:    http://localhost:8501" -ForegroundColor White
Write-Host ""

$choice = Read-Host "¿Qué deseas hacer? [C]luster / [L]ocal / [S]treamlit / [N]ada"

switch ($choice.ToUpper()) {
    "C" {
        Write-Host "Iniciando cluster distribuido..." -ForegroundColor Green
        .\start_ray_cluster.ps1
    }
    "L" {
        Write-Host "Ejecutando entrenamiento local..." -ForegroundColor Green
        .\run_training_standalone.ps1
    }
    "S" {
        Write-Host "Iniciando Streamlit..." -ForegroundColor Green
        .\run_streamlit.ps1
    }
    default {
        Write-Host "Configuración completada. Ejecuta los scripts cuando estés listo." -ForegroundColor Green
    }
}
