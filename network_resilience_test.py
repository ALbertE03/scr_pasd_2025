"""
Simulador de fallos de red para probar la tolerancia a fallos del cluster Ray
Este script simula diferentes tipos de problemas de red y conectividad
"""

import ray
import time
import random
import logging
import threading
import subprocess
import psutil
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
class NetworkResilienceTest:
    """
    Pruebas de resistencia de red para el cluster Ray
    """
    def __init__(self, test_id):
        self.test_id = test_id
        self.start_time = time.time()
        self.heartbeat_count = 0
        logger.info(f"NetworkResilienceTest {test_id} inicializado")
    
    def heartbeat(self):
        """Latido del corazón para verificar conectividad"""
        self.heartbeat_count += 1
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            'test_id': self.test_id,
            'heartbeat_count': self.heartbeat_count,
            'timestamp': datetime.now().isoformat(),
            'uptime': uptime,
            'worker_id': str(ray.get_runtime_context().get_worker_id()),
            'node_id': str(ray.get_runtime_context().get_node_id())
        }
    
    def compute_intensive_task(self, matrix_size=1000):
        """Tarea intensiva de computación para probar bajo carga"""
        import numpy as np
        
        logger.info(f"Test {self.test_id} - Iniciando tarea intensiva (matriz {matrix_size}x{matrix_size})")
        
        start_time = time.time()
        
        # Crear matriz aleatoria
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        
        # Multiplicación de matrices
        result = np.dot(matrix_a, matrix_b)
        
        # Calcular eigenvalores (computacionalmente intensivo)
        eigenvalues = np.linalg.eigvals(result[:100, :100])  # Solo una submatriz para velocidad
        
        computation_time = time.time() - start_time
        
        result_info = {
            'test_id': self.test_id,
            'task_type': 'compute_intensive',
            'matrix_size': matrix_size,
            'computation_time': computation_time,
            'eigenvalues_count': len(eigenvalues),
            'timestamp': datetime.now().isoformat(),
            'worker_id': str(ray.get_runtime_context().get_worker_id())
        }
        
        logger.info(f"Test {self.test_id} - Tarea intensiva completada en {computation_time:.2f}s")
        return result_info
    
    def memory_stress_test(self, memory_mb=100):
        """Prueba de estrés de memoria"""
        import numpy as np
        
        logger.info(f"Test {self.test_id} - Iniciando prueba de memoria ({memory_mb}MB)")
        
        start_time = time.time()
        
        # Crear arrays para consumir memoria
        arrays = []
        try:
            # Crear arrays de aproximadamente 1MB cada uno
            for i in range(memory_mb):
                arr = np.random.rand(128, 1024)  # ~1MB
                arrays.append(arr)
                
                # Pequeña pausa para no saturar inmediatamente
                if i % 10 == 0:
                    time.sleep(0.01)
            
            # Realizar algunas operaciones en los arrays
            total_sum = sum(np.sum(arr) for arr in arrays[:10])  # Solo los primeros 10
            
            memory_time = time.time() - start_time
            
            result_info = {
                'test_id': self.test_id,
                'task_type': 'memory_stress',
                'memory_allocated_mb': memory_mb,
                'arrays_created': len(arrays),
                'total_sum': float(total_sum),
                'memory_time': memory_time,
                'timestamp': datetime.now().isoformat(),
                'worker_id': str(ray.get_runtime_context().get_worker_id())
            }
            
            logger.info(f"Test {self.test_id} - Prueba de memoria completada: {memory_mb}MB en {memory_time:.2f}s")
            return result_info
            
        except Exception as e:
            logger.error(f"Test {self.test_id} - Error en prueba de memoria: {e}")
            raise
        finally:
            # Limpiar memoria
            del arrays

def run_network_resilience_test(duration_minutes=10, num_tests=6):
    """
    Ejecuta pruebas de resistencia de red
    """
    logger.info(f"Iniciando pruebas de resistencia de red por {duration_minutes} minutos")
    
    try:
        # Conectar a Ray
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
        logger.info("Conectado al cluster Ray para pruebas de red")
        
        # Crear instancias de prueba
        test_instances = [NetworkResilienceTest.remote(f"net_test_{i}") for i in range(num_tests)]
        logger.info(f"Creadas {num_tests} instancias de prueba")
        
        end_time = time.time() + (duration_minutes * 60)
        iteration = 0
        all_results = []
        
        while time.time() < end_time:
            iteration += 1
            logger.info(f"=== Iteración {iteration} ===")
            
            try:
                # Prueba de latidos (heartbeat)
                logger.info("Ejecutando pruebas de latido...")
                heartbeat_futures = [test.heartbeat.remote() for test in test_instances]
                heartbeats = ray.get(heartbeat_futures)
                
                logger.info(f"Latidos recibidos: {len(heartbeats)}/{len(test_instances)}")
                
                # Pruebas computacionales
                logger.info("Ejecutando pruebas computacionales...")
                compute_futures = [
                    test.compute_intensive_task.remote(random.randint(500, 1500)) 
                    for test in test_instances
                ]
                
                # Esperar resultados con timeout
                ready_futures, remaining_futures = ray.wait(
                    compute_futures, 
                    num_returns=len(compute_futures), 
                    timeout=180
                )
                
                if ready_futures:
                    compute_results = ray.get(ready_futures)
                    all_results.extend(compute_results)
                    logger.info(f"Tareas computacionales completadas: {len(compute_results)}")
                else:
                    logger.warning("Timeout en tareas computacionales")
                
                # Cancelar tareas que no completaron
                for future in remaining_futures:
                    ray.cancel(future)
                
                # Pruebas de memoria (alternadas)
                if iteration % 3 == 0:
                    logger.info("Ejecutando pruebas de memoria...")
                    memory_futures = [
                        test.memory_stress_test.remote(random.randint(50, 200))
                        for test in test_instances[:3]  # Solo algunos para no saturar
                    ]
                    
                    ready_memory, remaining_memory = ray.wait(
                        memory_futures,
                        num_returns=len(memory_futures),
                        timeout=120
                    )
                    
                    if ready_memory:
                        memory_results = ray.get(ready_memory)
                        all_results.extend(memory_results)
                        logger.info(f"Pruebas de memoria completadas: {len(memory_results)}")
                    
                    for future in remaining_memory:
                        ray.cancel(future)
                
                # Información del cluster
                try:
                    cluster_resources = ray.cluster_resources()
                    nodes = ray.nodes()
                    logger.info(f"Recursos del cluster: {cluster_resources}")
                    logger.info(f"Nodos activos: {len([n for n in nodes if n['Alive']])}")
                except Exception as e:
                    logger.warning(f"Error obteniendo info del cluster: {e}")
                
                # Pausa entre iteraciones
                time.sleep(random.uniform(15, 30))
                
            except Exception as e:
                logger.error(f"Error en iteración {iteration}: {e}")
                time.sleep(60)  # Pausa más larga en caso de error
        
        # Guardar resultados
        results_summary = {
            'test_duration_minutes': duration_minutes,
            'total_iterations': iteration,
            'total_results': len(all_results),
            'test_end_time': datetime.now().isoformat(),
            'results': all_results
        }
        
        with open('/app/network_resilience_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Pruebas de resistencia de red completadas")
        logger.info(f"Total de resultados: {len(all_results)}")
        logger.info(f"Resultados guardados en network_resilience_results.json")
        
    except Exception as e:
        logger.error(f"Error en pruebas de resistencia: {e}")
    finally:
        try:
            ray.shutdown()
        except:
            pass

def simulate_network_partitions():
    """
    Simula particiones de red usando iptables (requiere privilegios)
    NOTA: Esta función es solo para demostración, requiere privilegios de administrador
    """
    logger.info("ADVERTENCIA: Esta función requiere privilegios de administrador")
    logger.info("Para usar en producción, ejecutar en un entorno controlado")
    
    # Comandos de ejemplo para simular problemas de red
    network_commands = [
        # Simular latencia alta
        "tc qdisc add dev eth0 root netem delay 500ms",
        
        # Simular pérdida de paquetes
        "tc qdisc add dev eth0 root netem loss 10%",
        
        # Simular ancho de banda limitado
        "tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms",
        
        # Restaurar configuración normal
        "tc qdisc del dev eth0 root"
    ]
    
    logger.info("Comandos para simular problemas de red:")
    for i, cmd in enumerate(network_commands, 1):
        logger.info(f"{i}. {cmd}")
    
    logger.info("Ejecutar estos comandos manualmente con privilegios de administrador")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pruebas de resistencia de red para Ray")
    parser.add_argument("--duration", type=int, default=10, help="Duración en minutos (default: 10)")
    parser.add_argument("--tests", type=int, default=6, help="Número de instancias de prueba (default: 6)")
    parser.add_argument("--simulate-network", action="store_true", help="Mostrar comandos para simular problemas de red")
    
    args = parser.parse_args()
    
    if args.simulate_network:
        simulate_network_partitions()
    else:
        run_network_resilience_test(duration_minutes=args.duration, num_tests=args.tests)
