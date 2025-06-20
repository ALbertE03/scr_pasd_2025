import ray
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, 
    SGDClassifier, 
    RidgeClassifier,
    PassiveAggressiveClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import threading
import logging
from typing import Dict, List, Any, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(max_retries=3, retry_exceptions=True)
def train_model_remote(model, model_name, X_train, y_train, X_test, y_test, node_id=None):
    """Entrena un modelo de forma remota con tolerancia a fallos"""
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando entrenamiento de {model_name} en nodo {node_id}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        training_time = time.time() - start_time
        
        logger.info(f"Entrenamiento de {model_name} completado exitosamente en {training_time:.2f}s")
        
        return {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': y_pred.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"Error entrenando {model_name} en nodo {node_id}: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e),
            'node_id': node_id,
            'timestamp': datetime.now().isoformat()
        }


class DistributedMLTrainer:
    def __init__(self, head_address=None, enable_fault_tolerance=True):
        """Inicializa el entrenador distribuido de ML con tolerancia a fallos"""
        self.enable_fault_tolerance = enable_fault_tolerance
        self.results = {}
        self.trained_models = {}
        self.failed_tasks = []
        self.cluster_nodes = []
        
        if not ray.is_initialized():
            ray_config = {
                "num_cpus": None,  
                "ignore_reinit_error": True,
                "_enable_object_reconstruction": True, 
                "_reconstruction_timeout": 30
            }
            
            if head_address:
                ray_config["address"] = head_address
                logger.info(f"Conectando a cluster Ray en: {head_address}")
            else:
                logger.info("Iniciando Ray en modo local con autodescubrimiento")
            
            ray.init(**ray_config)
        

        self._update_cluster_info()
        
    def _update_cluster_info(self):
        """Actualiza información del cluster para autodescubrimiento"""
        try:
            self.cluster_nodes = ray.nodes()
            alive_nodes = [node for node in self.cluster_nodes if node.get('Alive', False)]
            logger.info(f"Cluster autodescubierto: {len(alive_nodes)} nodos vivos de {len(self.cluster_nodes)} totales")
        except Exception as e:
            logger.warning(f"Error actualizando información del cluster: {e}") 

    def get_available_datasets(self):       
        """Retorna los datasets disponibles para clasificación"""
        return {
            'iris': load_iris(),
            'wine': load_wine(),
            'breast_cancer': load_breast_cancer(),
            'digits': load_digits()
            }

    def get_available_models(self):
        """Retorna los modelos de clasificación supervisada disponibles"""
        models = {
            # Modelos basados en árboles            
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
            'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=0.1, algorithm='SAMME'),
            'ExtraTrees': ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=10),
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'XGBoost': HistGradientBoostingClassifier(random_state=42, max_iter=100, learning_rate=0.1, max_depth=6),
            
            # Modelos lineales (Solo clasificadores)
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver='liblinear'),
            'SGD': SGDClassifier(random_state=42, max_iter=1000, loss='log_loss', alpha=0.0001),
            'PassiveAggressive': PassiveAggressiveClassifier(random_state=42, C=1.0, max_iter=1000),
              # Modelos basados en vecinos (Clasificadores)
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
            
            # Support Vector Machines (Clasificadores)
            'SVM': SVC(random_state=42, probability=True, C=1.0, kernel='rbf'),  
            'LinearSVM': LinearSVC(random_state=42, C=1.0, max_iter=1000, dual=False),
            
            # Modelos de Naive Bayes (Todos son clasificadores)
            'GaussianNB': GaussianNB(var_smoothing=1e-9),
            'BernoulliNB': BernoulliNB(alpha=1.0),
            'MultinomialNB': MultinomialNB(alpha=1.0),
            'ComplementNB': ComplementNB(alpha=1.0),
            
            # Discriminant Analysis (Clasificadores)
            'LDA': LinearDiscriminantAnalysis(solver='svd'),
            'QDA': QuadraticDiscriminantAnalysis(),
            
            # Neural Networks (Clasificador)
            'MLP': MLPClassifier(random_state=42, hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam'),
              # Ensemble Methods (Clasificadores)
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42, n_estimators=10),
            'Voting': VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
                ('svm', SVC(random_state=42, probability=True)),
                ('lr', LogisticRegression(random_state=42))
            ], voting='soft')
        }
            
        return models 
           
    def train_models_distributed(self, dataset_name='iris', selected_models=None, test_size=0.3):
        """Entrena múltiples modelos de forma distribuida con tolerancia a fallos"""
        logger.info(f"Iniciando entrenamiento distribuido con dataset: {dataset_name}")
        

        self._update_cluster_info()

        datasets = self.get_available_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} no disponible. Opciones: {list(datasets.keys())}")
            
        dataset = datasets[dataset_name]
        X, y = dataset.data, dataset.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        
        available_models = self.get_available_models()
        if selected_models is None:
            selected_models = list(available_models.keys())
        

        remote_tasks = []
        task_info = {}
        
        for i, model_name in enumerate(selected_models):
            if model_name in available_models:
                model = available_models[model_name]
                node_id = f"node_{i % len(self.cluster_nodes)}" if self.cluster_nodes else f"node_{i}"
                
                task = train_model_remote.remote(
                    model, model_name, X_train, y_train, X_test, y_test, node_id
                )
                remote_tasks.append(task)
                task_info[task] = {'model_name': model_name, 'node_id': node_id}
        
        logger.info(f"Ejecutando {len(remote_tasks)} entrenamientos en paralelo con tolerancia a fallos...")
        
        results = []
        failed_results = []
        
        try:

            completed_results = ray.get(remote_tasks, timeout=300)  
            
            for result in completed_results:
                if result.get('status') == 'success':
                    results.append(result)
                else:
                    failed_results.append(result)
                    
        except ray.exceptions.GetTimeoutError:
            logger.warning("Timeout en algunas tareas, recuperando resultados parciales...")
            ready_tasks, remaining_tasks = ray.wait(remote_tasks, num_returns=len(remote_tasks), timeout=0)
            
            for task in ready_tasks:
                try:
                    result = ray.get(task)
                    if result.get('status') == 'success':
                        results.append(result)
                    else:
                        failed_results.append(result)
                except Exception as e:
                    failed_results.append({
                        'model_name': task_info[task]['model_name'],
                        'status': 'failed',
                        'error': str(e),
                        'node_id': task_info[task]['node_id']
                    })
            
            for task in remaining_tasks:
                ray.cancel(task)
        
        except Exception as e:
            logger.error(f"Error durante ejecución distribuida: {e}")
            return {}
        
        for result in results:
            self.results[result['model_name']] = result
            self.trained_models[result['model_name']] = result['model']

        self.failed_tasks.extend(failed_results)

        total_tasks = len(selected_models)
        successful_tasks = len(results)
        failed_tasks = len(failed_results)
        
        logger.info(f"Tolerancia a fallos - Exitosos: {successful_tasks}/{total_tasks}, Fallos: {failed_tasks}")
        
        if failed_results:
            logger.warning("Tareas fallidas:")
            for failed in failed_results:
                logger.warning(f"  - {failed['model_name']}: {failed.get('error', 'Error desconocido')}")
        
        if results:
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            logger.info("\nResultados del entrenamiento distribuido:")
            logger.info("=" * 60)
            for model_name, result in sorted_results:
                logger.info(f"{model_name:20} | Accuracy: {result['accuracy']:.4f} | "
                          f"CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f} | "
                          f"Tiempo: {result['training_time']:.2f}s | Nodo: {result.get('node_id', 'N/A')}")
        
        return self.results
    
    def save_results(self, filename="training_results.json"):
        """Guarda los resultados en un archivo JSON"""

        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_result = result.copy()
            serializable_result.pop('model', None)
            serializable_results[model_name] = serializable_result        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Resultados guardados en: {filename}")
    
    def save_models(self, directory="models"):
        """Guarda TODOS los modelos entrenados exitosamente"""
        os.makedirs(directory, exist_ok=True)
        
        saved_count = 0
        for model_name, model in self.trained_models.items():
            try:
                filename = os.path.join(directory, f"{model_name}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                saved_count += 1
                logger.info(f"Modelo {model_name} guardado en: {filename}")
            except Exception as e:
                logger.error(f"Error guardando modelo {model_name}: {e}")
        
        print(f"Modelos guardados en directorio: {directory} ({saved_count} modelos)")
        
        # También crear estructura para API
        api_directory = os.path.join("models", directory.split("_")[-1] if "_" in directory else "default")
        os.makedirs(api_directory, exist_ok=True)
        
        # Copiar modelos para la API
        import shutil
        for model_name, model in self.trained_models.items():
            try:
                source_file = os.path.join(directory, f"{model_name}.pkl")
                dest_file = os.path.join(api_directory, f"{model_name}.pkl")
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"Modelo {model_name} copiado para API: {dest_file}")
            except Exception as e:
                logger.error(f"Error copiando modelo {model_name} para API: {e}")
    
    def get_cluster_info(self):
        """Obtiene información del cluster Ray"""
        return ray.cluster_resources()
    
    def train_multiple_datasets_sequential(self, datasets_list=None, selected_models=None, test_size=0.3):
        """Entrena múltiples datasets secuencialmente en una misma ejecución"""
        if datasets_list is None:
            datasets_list = ['iris', 'wine', 'breast_cancer']
        
        if selected_models is None:
            selected_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
        
        all_results = {}
        execution_summary = {
            'total_datasets': len(datasets_list),
            'successful_datasets': 0,
            'failed_datasets': 0,
            'total_models_trained': 0,
            'total_execution_time': 0,
            'start_time': datetime.now().isoformat()
        }
        
        start_time = time.time()
        logger.info(f"Iniciando entrenamiento secuencial de {len(datasets_list)} datasets")
        logger.info(f"Datasets: {datasets_list}")
        logger.info(f"Modelos por dataset: {selected_models}")
        
        for dataset_name in datasets_list:
            dataset_start_time = time.time()
            logger.info(f"\n{'='*20} PROCESANDO DATASET: {dataset_name.upper()} {'='*20}")
            
            try:
                self.results = {}
                self.trained_models = {}

                dataset_results = self.train_models_distributed(
                    dataset_name=dataset_name,
                    selected_models=selected_models,
                    test_size=test_size
                )
                
                if dataset_results:
                    all_results[dataset_name] = dataset_results
                    execution_summary['successful_datasets'] += 1
                    execution_summary['total_models_trained'] += len(dataset_results)

                    self.save_results(f"results_{dataset_name}.json")
                    self.save_models(f"models_{dataset_name}")
                    
                    dataset_time = time.time() - dataset_start_time
                    logger.info(f"Dataset {dataset_name} completado en {dataset_time:.2f}s")

                    best_model = max(dataset_results.items(), key=lambda x: x[1]['accuracy'])
                    logger.info(f"Mejor modelo para {dataset_name}: {best_model[0]} "
                              f"(Accuracy: {best_model[1]['accuracy']:.4f})")
                
                else:
                    execution_summary['failed_datasets'] += 1
                    logger.error(f"Falló el entrenamiento para dataset {dataset_name}")
                
            except Exception as e:
                execution_summary['failed_datasets'] += 1
                logger.error(f"Error procesando dataset {dataset_name}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        execution_summary['total_execution_time'] = total_time
        execution_summary['end_time'] = datetime.now().isoformat()

        with open("execution_summary.json", 'w') as f:
            json.dump(execution_summary, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info("RESUMEN DE EJECUCIÓN SECUENCIAL")
        logger.info(f"{'='*50}")
        logger.info(f"Datasets procesados exitosamente: {execution_summary['successful_datasets']}/{execution_summary['total_datasets']}")
        logger.info(f"Datasets fallidos: {execution_summary['failed_datasets']}")
        logger.info(f"Total de modelos entrenados: {execution_summary['total_models_trained']}")
        logger.info(f"Tiempo total de ejecución: {total_time:.2f}s")
        
        if self.failed_tasks:
            logger.info(f"Total de tareas individuales fallidas: {len(self.failed_tasks)}")
        
        return all_results, execution_summary
    
    def get_fault_tolerance_stats(self):
        """Obtiene estadísticas de tolerancia a fallos"""
        return {
            'failed_tasks': len(self.failed_tasks),
            'failed_task_details': self.failed_tasks,
            'cluster_nodes': len(self.cluster_nodes),
            'alive_nodes': len([node for node in self.cluster_nodes if node.get('Alive', False)])
        }


def main():
    """Función principal para ejecutar el entrenamiento con tolerancia a fallos"""
    logger.info("Iniciando entrenador distribuido de Machine Learning con tolerancia a fallos")

    trainer = DistributedMLTrainer(enable_fault_tolerance=True)

    cluster_info = trainer.get_cluster_info()
    logger.info(f"Recursos del cluster autodescubierto: {cluster_info}")

    logger.info("\n🚀 MODO: Entrenamiento Secuencial de Múltiples Datasets")
    
    datasets_to_train = ['iris', 'wine', 'breast_cancer']
    models_to_use = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN']
    
    all_results, summary = trainer.train_multiple_datasets_sequential(
        datasets_list=datasets_to_train,
        selected_models=models_to_use
    )
    
    fault_stats = trainer.get_fault_tolerance_stats()
    logger.info(f"\n📊 ESTADÍSTICAS DE TOLERANCIA A FALLOS:")
    logger.info(f"Nodos en cluster: {fault_stats['cluster_nodes']}")
    logger.info(f"Nodos vivos: {fault_stats['alive_nodes']}")
    logger.info(f"Tareas fallidas: {fault_stats['failed_tasks']}")
    
    if fault_stats['failed_tasks'] > 0:
        logger.info("Detalles de tareas fallidas:")
        for failed_task in fault_stats['failed_task_details']:
            logger.info(f"  - {failed_task['model_name']}: {failed_task.get('error', 'Error desconocido')}")

    if all_results:
        logger.info(f"\n📈 COMPARACIÓN ENTRE DATASETS:")
        logger.info("-" * 80)
        logger.info(f"{'Dataset':<15} {'Mejor Modelo':<20} {'Accuracy':<10} {'Tiempo Avg':<12}")
        logger.info("-" * 80)
        
        for dataset_name, results in all_results.items():
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                avg_time = sum(r['training_time'] for r in results.values()) / len(results)
                logger.info(f"{dataset_name:<15} {best_model[0]:<20} {best_model[1]['accuracy']:<10.4f} {avg_time:<12.2f}")
    
    logger.info("\n✅ Entrenamiento distribuido completado con tolerancia a fallos!")
    logger.info(f"📁 Resultados guardados en archivos results_*.json")
    logger.info(f"🤖 Modelos guardados en directorios models_*")
    logger.info(f"📊 Resumen de ejecución guardado en execution_summary.json")


if __name__ == "__main__":
    #main()
    pass

