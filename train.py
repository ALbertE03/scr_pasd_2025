import ray
import time
import json
from datetime import datetime
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, 
    SGDClassifier, 
    PassiveAggressiveClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import (
    GaussianNB, 
    BernoulliNB, 
    MultinomialNB, 
    ComplementNB
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

import pickle
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True,scheduling_strategy="SPREAD",runtime_env={"fail_fast": False },catch_exceptions=True,retry_delay_s=2.0)
def iris():
    return load_iris()
@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True,scheduling_strategy="SPREAD",runtime_env={"fail_fast": False },catch_exceptions=True,retry_delay_s=2.0)
def wine():
    return load_wine()
@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True,scheduling_strategy="SPREAD",runtime_env={"fail_fast": False },catch_exceptions=True,retry_delay_s=2.0)
def breast_cancer():
    return load_breast_cancer()
@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True,scheduling_strategy="SPREAD",runtime_env={"fail_fast": False },catch_exceptions=True,retry_delay_s=2.0)
def digits():
    return load_digits()

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
            'iris': iris.remote(),
            'wine': wine.remote(),
            'breast_cancer': breast_cancer.remote(),
            'digits': digits.remote()
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
    
    def get_fault_tolerance_stats(self):
        """Obtiene estadísticas de tolerancia a fallos"""
        return {
            'failed_tasks': len(self.failed_tasks),
            'failed_task_details': self.failed_tasks,
            'cluster_nodes': len(self.cluster_nodes),
            'alive_nodes': len([node for node in self.cluster_nodes if node.get('Alive', False)])
        }


if __name__ == "__main__":
    
    pass

