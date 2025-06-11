"""
Script de prueba de tolerancia a fallos para el cluster Ray
Este script se ejecuta dentro del contenedor y simula cargas de trabajo
mientras se monitorea la capacidad del sistema para manejar fallos de nodos.
"""

import ray
import time
import random
import numpy as np
import logging
import json
import threading
from datetime import datetime, timedelta
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import sys
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/fault_tolerance_test.log')
    ]
)
logger = logging.getLogger(__name__)

@ray.remote
class ResilientTrainer:
    """
    Entrenador que puede recuperarse de fallos y continuar trabajando
    """
    def __init__(self, trainer_id):
        self.trainer_id = trainer_id
        self.completed_tasks = 0
        self.start_time = time.time()
        logger.info(f"Trainer {trainer_id} inicializado")
    
    def train_classification_model(self, n_samples=1000, n_features=20):
        """Entrena un modelo de clasificación"""
        try:
            logger.info(f"Trainer {self.trainer_id} - Iniciando tarea de clasificación")
            
            # Generar datos sintéticos
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_classes=random.randint(2, 4),
                random_state=random.randint(1, 10000)
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            model = RandomForestClassifier(
                n_estimators=random.randint(50, 200),
                max_depth=random.randint(5, 15),
                random_state=42
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluar
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            self.completed_tasks += 1
            
            result = {
                'trainer_id': self.trainer_id,
                'task_type': 'classification',
                'completed_tasks': self.completed_tasks,
                'accuracy': accuracy,
                'training_time': training_time,
                'n_samples': n_samples,
                'n_features': n_features,
                'timestamp': datetime.now().isoformat(),
                'worker_id': str(ray.get_runtime_context().get_worker_id()),
                'node_id': str(ray.get_runtime_context().get_node_id())
            }
            
            logger.info(f"Trainer {self.trainer_id} - Tarea {self.completed_tasks} completada: Accuracy={accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Trainer {self.trainer_id} - Error en entrenamiento: {e}")
            raise
    
    def train_regression_model(self, n_samples=1000, n_features=20):
        """Entrena un modelo de regresión"""
        try:
            logger.info(f"Trainer {self.trainer_id} - Iniciando tarea de regresión")
            
            # Generar datos sintéticos
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.8),
                noise=random.uniform(0.1, 1.0),
                random_state=random.randint(1, 10000)
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            model = RandomForestRegressor(
                n_estimators=random.randint(50, 150),
                max_depth=random.randint(5, 12),
                random_state=42
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluar
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            
            self.completed_tasks += 1
            
            result = {
                'trainer_id': self.trainer_id,
                'task_type': 'regression',
                'completed_tasks': self.completed_tasks,
                'mse': mse,
                'training_time': training_time,
                'n_samples': n_samples,
                'n_features': n_features,
                'timestamp': datetime.now().isoformat(),
                'worker_id': str(ray.get_runtime_context().get_worker_id()),
                'node_id': str(ray.get_runtime_context().get_node_id())
            }
            
            logger.info(f"Trainer {self.trainer_id} - Tarea {self.completed_tasks} completada: MSE={mse:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Trainer {self.trainer_id} - Error en entrenamiento: {e}")
            raise
    
    def get_status(self):
        """Retorna el estado actual del trainer"""
        return {
            'trainer_id': self.trainer_id,
            'completed_tasks': self.completed_tasks,
            'uptime': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        }

@ray.remote
class FaultToleranceMonitor:
    """
    Monitor que rastrea el estado del cluster y las tareas
    """
    def __init__(self):
        self.start_time = time.time()
        self.task_results = []
        self.cluster_events = []
        logger.info("Monitor de tolerancia a fallos inicializado")
    
    def log_task_result(self, result):
        """Registra el resultado de una tarea"""
        self.task_results.append(result)
        logger.info(f"Resultado registrado: Trainer {result['trainer_id']}, Tarea {result['completed_tasks']}")
    
    def log_cluster_event(self, event_type, description):
        """Registra un evento del cluster"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'uptime': time.time() - self.start_time
        }
        self.cluster_events.append(event)
        logger.info(f"Evento del cluster: {event_type} - {description}")
    
    def get_statistics(self):
        """Retorna estadísticas del monitoreo"""
        if not self.task_results:
            return {'message': 'No hay resultados disponibles aún'}
        
        # Estadísticas por tipo de tarea
        classification_results = [r for r in self.task_results if r['task_type'] == 'classification']
        regression_results = [r for r in self.task_results if r['task_type'] == 'regression']
        
        stats = {
            'total_tasks': len(self.task_results),
            'classification_tasks': len(classification_results),
            'regression_tasks': len(regression_results),
            'total_uptime': time.time() - self.start_time,
            'cluster_events': len(self.cluster_events)
        }
        
        if classification_results:
            stats['avg_classification_accuracy'] = np.mean([r['accuracy'] for r in classification_results])
            stats['avg_classification_time'] = np.mean([r['training_time'] for r in classification_results])
        
        if regression_results:
            stats['avg_regression_mse'] = np.mean([r['mse'] for r in regression_results])
            stats['avg_regression_time'] = np.mean([r['training_time'] for r in regression_results])
        
        # Estadísticas por trainer
        trainer_stats = {}
        for result in self.task_results:
            trainer_id = result['trainer_id']
            if trainer_id not in trainer_stats:
                trainer_stats[trainer_id] = 0
            trainer_stats[trainer_id] += 1
        
        stats['trainer_task_distribution'] = trainer_stats
        
        return stats
    
    def save_results(self, filename='/app/fault_tolerance_results.json'):
        """Guarda los resultados en un archivo"""
        data = {
            'test_metadata': {
                'start_time': self.start_time,
                'total_duration': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            },
            'task_results': self.task_results,
            'cluster_events': self.cluster_events,
            'statistics': self.get_statistics()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Resultados guardados en {filename}")
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")

def run_fault_tolerance_test(duration_minutes=5, num_trainers=4):
    """
    Ejecuta la prueba de tolerancia a fallos
    """
    logger.info(f"Iniciando prueba de tolerancia a fallos por {duration_minutes} minutos")
    logger.info(f"Número de trainers: {num_trainers}")
    
    try:
        # Conectar a Ray
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
        logger.info("Conectado al cluster Ray")
        
        # Crear monitor
        monitor = FaultToleranceMonitor.remote()
        
        # Crear trainers
        trainers = [ResilientTrainer.remote(f"trainer_{i}") for i in range(num_trainers)]
        logger.info(f"Creados {num_trainers} trainers")
        
        # Registrar evento inicial
        monitor.log_cluster_event.remote("test_start", f"Prueba iniciada con {num_trainers} trainers")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        task_counter = 0
        
        while datetime.now() < end_time:
            try:
                # Crear lote de tareas
                tasks = []
                for trainer in trainers:
                    # Alternar entre clasificación y regresión
                    if random.choice([True, False]):
                        task = trainer.train_classification_model.remote(
                            n_samples=random.randint(800, 2000),
                            n_features=random.randint(15, 30)
                        )
                    else:
                        task = trainer.train_regression_model.remote(
                            n_samples=random.randint(800, 2000),
                            n_features=random.randint(15, 30)
                        )
                    tasks.append(task)
                
                # Esperar resultados con timeout
                ready_tasks, remaining_tasks = ray.wait(tasks, num_returns=len(tasks), timeout=120)
                
                if ready_tasks:
                    try:
                        results = ray.get(ready_tasks)
                        for result in results:
                            monitor.log_task_result.remote(result)
                            task_counter += 1
                        
                        logger.info(f"Lote completado: {len(results)} tareas, Total: {task_counter}")
                    except Exception as e:
                        logger.error(f"Error obteniendo resultados: {e}")
                        monitor.log_cluster_event.remote("task_error", f"Error en lote de tareas: {str(e)}")
                
                # Cancelar tareas que no completaron
                for task in remaining_tasks:
                    ray.cancel(task)
                
                # Verificar estado del cluster
                try:
                    cluster_resources = ray.cluster_resources()
                    available_cpus = cluster_resources.get('CPU', 0)
                    logger.info(f"CPUs disponibles en el cluster: {available_cpus}")
                    
                    if available_cpus < 1:
                        monitor.log_cluster_event.remote("low_resources", "Recursos muy bajos en el cluster")
                    
                except Exception as e:
                    logger.warning(f"Error verificando recursos del cluster: {e}")
                
                # Pausa entre lotes
                time.sleep(random.uniform(10, 20))
                
            except Exception as e:
                logger.error(f"Error en iteración: {e}")
                monitor.log_cluster_event.remote("iteration_error", f"Error en iteración: {str(e)}")
                time.sleep(30)  # Pausa más larga en caso de error
        
        # Finalizar prueba
        logger.info("Prueba de tolerancia a fallos completada")
        monitor.log_cluster_event.remote("test_end", "Prueba completada exitosamente")
        
        # Obtener estadísticas finales
        final_stats = ray.get(monitor.get_statistics.remote())
        logger.info("Estadísticas finales:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Guardar resultados
        ray.get(monitor.save_results.remote())
        
    except Exception as e:
        logger.error(f"Error fatal en la prueba: {e}")
    finally:
        try:
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prueba de tolerancia a fallos para Ray")
    parser.add_argument("--duration", type=int, default=5, help="Duración en minutos (default: 5)")
    parser.add_argument("--trainers", type=int, default=4, help="Número de trainers (default: 4)")
    
    args = parser.parse_args()
    
    run_fault_tolerance_test(duration_minutes=args.duration, num_trainers=args.trainers)
