import ray
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


@ray.remote
def train_model_remote(model, model_name, X_train, y_train, X_test, y_test):
    """Entrena un modelo de forma remota"""
    start_time = time.time()
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calcular cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    training_time = time.time() - start_time
    
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
        'timestamp': datetime.now().isoformat()
    }


class DistributedMLTrainer:
    def __init__(self, head_address=None):
        """Inicializa el entrenador distribuido de ML"""
        if not ray.is_initialized():
            if head_address:
                ray.init(address=head_address)
            else:
                ray.init()
        
        self.results = {}
        self.trained_models = {}
        
    def get_available_datasets(self):
        """Retorna los datasets disponibles"""
        return {
            'iris': load_iris(),
            'wine': load_wine(),
            'breast_cancer': load_breast_cancer()
        }
    
    def get_available_models(self):
        """Retorna los modelos disponibles"""
        return {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SGD': SGDClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'NaiveBayes': GaussianNB()        }

    def train_models_distributed(self, dataset_name='iris', selected_models=None, test_size=0.3):
        """Entrena múltiples modelos de forma distribuida"""
        print(f"Iniciando entrenamiento distribuido con dataset: {dataset_name}")
        
        # Cargar dataset
        datasets = self.get_available_datasets()
        dataset = datasets[dataset_name]
        X, y = dataset.data, dataset.target
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        
        # Obtener modelos
        available_models = self.get_available_models()
        if selected_models is None:
            selected_models = list(available_models.keys())
          # Crear tareas remotas
        remote_tasks = []
        for model_name in selected_models:
            if model_name in available_models:
                model = available_models[model_name]
                task = train_model_remote.remote(
                    model, model_name, X_train, y_train, X_test, y_test
                )
                remote_tasks.append(task)
        
        print(f"Ejecutando {len(remote_tasks)} entrenamientos en paralelo...")
        
        # Ejecutar tareas y obtener resultados
        results = ray.get(remote_tasks)
        
        # Procesar resultados
        for result in results:
            self.results[result['model_name']] = result
            self.trained_models[result['model_name']] = result['model']
        
        # Ordenar por accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print("\n Resultados del entrenamiento distribuido:")
        print("=" * 60)
        for model_name, result in sorted_results:
            print(f"{model_name:20} | Accuracy: {result['accuracy']:.4f} | "
                  f"CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f} | "
                  f"Tiempo: {result['training_time']:.2f}s")
        
        return self.results
    
    def save_results(self, filename="training_results.json"):
        """Guarda los resultados en un archivo JSON"""
        # Convertir modelos a una representación serializable
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_result = result.copy()
            # Remover el objeto modelo para serialización
            serializable_result.pop('model', None)
            serializable_results[model_name] = serializable_result
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Resultados guardados en: {filename}")
    
    def save_models(self, directory="models"):
        """Guarda los modelos entrenados"""
        os.makedirs(directory, exist_ok=True)
        for model_name, model in self.trained_models.items():
            filename = os.path.join(directory, f"{model_name}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        print(f"Modelos guardados en directorio: {directory}")
    
    def get_cluster_info(self):
        """Obtiene información del cluster Ray"""
        return ray.cluster_resources()


def main():
    """Función principal para ejecutar el entrenamiento"""
    print("Iniciando entrenador distribuido de Machine Learning")
    
    # Inicializar entrenador
    trainer = DistributedMLTrainer()
    
    # Mostrar información del cluster
    cluster_info = trainer.get_cluster_info()
    print(f"Recursos del cluster: {cluster_info}")
    
    # Entrenar modelos con diferentes datasets
    datasets = ['iris', 'wine', 'breast_cancer']
    
    for dataset in datasets:
        print(f"\n{'='*20} DATASET: {dataset.upper()} {'='*20}")
        results = trainer.train_models_distributed(
            dataset_name=dataset,
            selected_models=['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
        )
        
        # Guardar resultados
        trainer.save_results(f"results_{dataset}.json")
        trainer.save_models(f"models_{dataset}")
    
    print("\nEntrenamiento distribuido completado!")


if __name__ == "__main__":
    main()

