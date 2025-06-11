# Script de PowerShell para probar tolerancia a fallos del cluster Ray
# Uso: .\test-fault-tolerance.ps1

param(
    [Parameter()]
    [ValidateSet("basic", "worker-failure", "multiple-workers", "stress-test", "all")]
    [string]$TestType = "basic",
    
    [Parameter()]
    [int]$Duration = 300  # Duración de la prueba en segundos (5 minutos por defecto)
)

# Colores para output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Cyan = "Cyan"
$White = "White"

function Write-Header {
    param([string]$Message)
    Write-Host "`n" + "="*60 -ForegroundColor $Cyan
    Write-Host $Message -ForegroundColor $White
    Write-Host "="*60 -ForegroundColor $Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n🔍 $Message" -ForegroundColor $Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Red
}

function Get-RayClusterStatus {
    Write-Step "Verificando estado del cluster Ray..."
    
    try {
        $rayStatus = docker exec $(docker ps -q -f "name=ray-head") ray status
        Write-Host $rayStatus
        return $true
    } catch {
        Write-Error "No se pudo obtener el estado del cluster Ray"
        return $false
    }
}

function Get-RunningServices {
    Write-Step "Servicios en ejecución:"
    docker-compose ps
}

function Start-TrainingJob {
    Write-Step "Iniciando trabajo de entrenamiento en segundo plano..."
    
    # Crear script de entrenamiento continuo
    $trainingScript = @"
import ray
import time
import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote
class TrainingWorker:
    def __init__(self):
        self.iteration = 0
    
    def train_model(self, dataset_size=1000):
        self.iteration += 1
        logger.info(f"Iteración {self.iteration} - Generando dataset de {dataset_size} muestras")
        
        # Generar dataset sintético
        X, y = make_classification(
            n_samples=dataset_size,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=random.randint(1, 1000)
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluar
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        result = {
            'iteration': self.iteration,
            'accuracy': accuracy,
            'training_time': training_time,
            'dataset_size': dataset_size,
            'worker_id': ray.get_runtime_context().get_worker_id()
        }
        
        logger.info(f"Iteración {self.iteration} completada - Accuracy: {accuracy:.4f}, Tiempo: {training_time:.2f}s")
        return result

def continuous_training(duration_seconds=300):
    logger.info(f"Iniciando entrenamiento continuo por {duration_seconds} segundos")
    
    try:
        ray.init(address="ray://localhost:10001", ignore_reinit_error=True)
        logger.info("Conectado al cluster Ray")
    except Exception as e:
        logger.error(f"Error conectando a Ray: {e}")
        return
    
    # Crear workers
    num_workers = min(4, len(ray.nodes()))
    workers = [TrainingWorker.remote() for _ in range(num_workers)]
    logger.info(f"Creados {num_workers} workers")
    
    start_time = time.time()
    iteration = 0
    results = []
    
    while time.time() - start_time < duration_seconds:
        iteration += 1
        logger.info(f"--- Ronda de entrenamiento {iteration} ---")
        
        try:
            # Ejecutar trabajos en paralelo
            futures = [worker.train_model.remote(random.randint(500, 2000)) for worker in workers]
            
            # Esperar resultados con timeout
            ready_futures, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=60)
            
            if ready_futures:
                batch_results = ray.get(ready_futures)
                results.extend(batch_results)
                
                avg_accuracy = np.mean([r['accuracy'] for r in batch_results])
                logger.info(f"Ronda {iteration} - Accuracy promedio: {avg_accuracy:.4f}")
            else:
                logger.warning("Timeout esperando resultados")
            
            # Manejar trabajos que no completaron
            for future in remaining_futures:
                ray.cancel(future)
            
        except Exception as e:
            logger.error(f"Error en iteración {iteration}: {e}")
        
        time.sleep(5)  # Pausa entre iteraciones
    
    logger.info(f"Entrenamiento completado. Total de resultados: {len(results)}")
    
    if results:
        final_accuracy = np.mean([r['accuracy'] for r in results])
        total_time = sum([r['training_time'] for r in results])
        logger.info(f"Accuracy promedio final: {final_accuracy:.4f}")
        logger.info(f"Tiempo total de entrenamiento: {total_time:.2f}s")
    
    ray.shutdown()

if __name__ == "__main__":
    continuous_training($Duration)
"@

    $trainingScript | Out-File -FilePath ".\fault_tolerance_training.py" -Encoding UTF8
    
    # Ejecutar en segundo plano usando Docker
    $jobId = Start-Job -ScriptBlock {
        param($Duration)
        docker exec -i $(docker ps -q -f "name=ray-head") python -c @"
$using:trainingScript
"@
    } -ArgumentList $Duration
    
    return $jobId
}

function Test-BasicFaultTolerance {
    Write-Header "PRUEBA BÁSICA DE TOLERANCIA A FALLOS"
    
    Write-Step "1. Verificando servicios iniciales"
    Get-RunningServices
    
    if (-not (Get-RayClusterStatus)) {
        Write-Error "El cluster Ray no está funcionando correctamente"
        return $false
    }
    
    Write-Step "2. Iniciando trabajo de entrenamiento"
    $job = Start-TrainingJob
    Start-Sleep -Seconds 30  # Dar tiempo para que inicie
    
    Write-Step "3. Eliminando un worker Ray"
    $workers = docker ps -q -f "name=ray-worker"
    if ($workers) {
        $workerToKill = ($workers -split "`n")[0]
        Write-Host "Eliminando worker: $workerToKill" -ForegroundColor $Yellow
        docker stop $workerToKill
        docker rm $workerToKill
    }
    
    Write-Step "4. Verificando estado del cluster después de eliminar worker"
    Start-Sleep -Seconds 10
    Get-RayClusterStatus
    
    Write-Step "5. Esperando que el trabajo continue..."
    Start-Sleep -Seconds 60
    
    Write-Step "6. Recreando worker eliminado"
    docker-compose up -d ray-worker
    Start-Sleep -Seconds 20
    
    Write-Step "7. Estado final del cluster"
    Get-RayClusterStatus
    
    # Limpiar
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
    
    Write-Success "Prueba básica de tolerancia a fallos completada"
    return $true
}

function Test-WorkerFailure {
    Write-Header "PRUEBA DE FALLO DE WORKER DURANTE ENTRENAMIENTO"
    
    Write-Step "Escalando a múltiples workers para la prueba"
    docker-compose up -d --scale ray-worker=3
    Start-Sleep -Seconds 30
    
    Get-RayClusterStatus
    
    Write-Step "Iniciando trabajo de entrenamiento intensivo"
    $job = Start-TrainingJob
    Start-Sleep -Seconds 45
    
    Write-Step "Eliminando workers de forma escalonada"
    $workers = docker ps -q -f "name=ray-worker"
    
    foreach ($worker in $workers) {
        Write-Host "Eliminando worker: $worker" -ForegroundColor $Yellow
        docker stop $worker
        docker rm $worker
        
        Start-Sleep -Seconds 30
        Write-Step "Estado del cluster después de eliminar worker"
        Get-RayClusterStatus
    }
    
    Write-Step "Esperando que el entrenamiento continúe solo con el head node"
    Start-Sleep -Seconds 60
    
    Write-Step "Recreando workers"
    docker-compose up -d --scale ray-worker=2
    Start-Sleep -Seconds 30
    
    Get-RayClusterStatus
    
    # Limpiar
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
    
    Write-Success "Prueba de fallo de worker completada"
}

function Test-StressTest {
    Write-Header "PRUEBA DE ESTRÉS CON FALLOS ALEATORIOS"
    
    Write-Step "Configurando cluster para prueba de estrés"
    docker-compose up -d --scale ray-worker=4
    Start-Sleep -Seconds 45
    
    Write-Step "Iniciando entrenamiento de larga duración"
    $job = Start-TrainingJob
    Start-Sleep -Seconds 60
    
    $endTime = (Get-Date).AddSeconds($Duration)
    $iteration = 1
    
    while ((Get-Date) -lt $endTime) {
        Write-Step "Iteración de fallo $iteration"
        
        # Obtener workers actuales
        $workers = docker ps -q -f "name=ray-worker"
        if ($workers.Count -gt 0) {
            # Eliminar worker aleatorio
            $randomWorker = Get-Random -InputObject $workers
            Write-Host "Eliminando worker aleatorio: $randomWorker" -ForegroundColor $Yellow
            docker stop $randomWorker
            docker rm $randomWorker
        }
        
        Start-Sleep -Seconds (Get-Random -Minimum 45 -Maximum 90)
        
        # Recrear workers si hay muy pocos
        $currentWorkers = docker ps -q -f "name=ray-worker"
        if ($currentWorkers.Count -lt 2) {
            Write-Step "Recreando workers"
            docker-compose up -d --scale ray-worker=3
            Start-Sleep -Seconds 30
        }
        
        Get-RayClusterStatus
        $iteration++
    }
    
    # Limpiar
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
    
    Write-Success "Prueba de estrés completada"
}

function Show-Usage {
    Write-Host "Uso: .\test-fault-tolerance.ps1 [-TestType <tipo>] [-Duration <segundos>]"
    Write-Host ""
    Write-Host "Tipos de prueba disponibles:"
    Write-Host "  basic          - Prueba básica eliminando un worker"
    Write-Host "  worker-failure - Elimina workers de forma escalonada"
    Write-Host "  stress-test    - Fallos aleatorios durante tiempo prolongado"
    Write-Host "  all            - Ejecuta todas las pruebas"
    Write-Host ""
    Write-Host "Parámetros:"
    Write-Host "  -Duration      - Duración de las pruebas en segundos (default: 300)"
}

# Verificar que Docker Compose esté disponible
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Error "Docker Compose no está instalado o no está en el PATH"
    exit 1
}

# Verificar que los servicios estén corriendo
Write-Step "Verificando que los servicios estén ejecutándose"
$runningServices = docker-compose ps -q
if (-not $runningServices) {
    Write-Error "Los servicios no están ejecutándose. Ejecuta '.\manage.ps1 start' primero"
    exit 1
}

# Ejecutar pruebas según el tipo
switch ($TestType) {
    "basic" { 
        Test-BasicFaultTolerance 
    }
    "worker-failure" { 
        Test-WorkerFailure 
    }
    "stress-test" { 
        Test-StressTest 
    }
    "all" {
        Test-BasicFaultTolerance
        Start-Sleep -Seconds 30
        Test-WorkerFailure
        Start-Sleep -Seconds 30
        Test-StressTest
    }
    default { 
        Show-Usage 
    }
}

Write-Header "PRUEBAS DE TOLERANCIA A FALLOS COMPLETADAS"
Write-Host "Para limpiar y restaurar el estado normal:" -ForegroundColor $Cyan
Write-Host "  .\manage.ps1 stop" -ForegroundColor $White
Write-Host "  .\manage.ps1 start" -ForegroundColor $White
