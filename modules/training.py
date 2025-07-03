import ray
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.linear_model import (LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, 
                                  LinearRegression, Ridge, Lasso)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote
class ModelManager:
    """Actor para gestionar el mapeo de modelos en el Ray object store"""
    
    def __init__(self):
        self.model_registry = {} 
        self.scores={}
        
    def create_model_id(self, model_name, dataset_name):
        """Crear ID único para el modelo basado en nombre del modelo y dataset"""
        return f"{dataset_name}_{model_name}"
    
    def store_model(self, model_name, dataset_name, model_pipeline):
        """Guardar modelo en object store y registrar la referencia"""
        model_id = self.create_model_id(model_name, dataset_name)
        model_ref = ray.put(model_pipeline)
        self.model_registry[model_id] = model_ref
        logger.info(f"Modelo {model_id} almacenado en object store")
        return model_id, model_ref
    
    def get_model(self, model_name, dataset_name):
        """Obtener referencia del modelo del object store"""
        model_id = self.create_model_id(model_name, dataset_name)
        if model_id in self.model_registry:
            return self.model_registry[model_id]
        else:
            logger.warning(f"Modelo {model_id} no encontrado en el registry")
            return None
    
    def list_all_model_ids(self):
        """Retornar todos los IDs de modelos disponibles"""
        return list(self.model_registry.keys())
    
    def search_models(self, model_name=None, dataset_name=None):
        """Buscar modelos por nombre de modelo o dataset"""
        matching_ids = []
        for model_id in self.model_registry.keys():
            parts = model_id.split('_', 1)
            if len(parts) == 2:
                dataset, model = parts[0], parts[1]
                if (model_name is None or model_name in model) and \
                   (dataset_name is None or dataset_name in dataset):
                    matching_ids.append(model_id)
        return matching_ids
    
    def remove_model(self, model_name, dataset_name):
        """Remover modelo del registry"""
        model_id = self.create_model_id(model_name, dataset_name)
        if model_id in self.model_registry:
            del self.model_registry[model_id]
            logger.info(f"Modelo {model_id} removido del registry")
            return True
        return False
    def get_scores(self, id):
        """Obtener scores del modelo desde el object store"""
        if id in self.scores:
            score_ref = self.scores[id]
            return ray.get(score_ref)  # Obtener los datos reales del object store
        return None
    
    def save_scores(self, model, dataset, scores):
        """Guardar scores en el object store"""
        id = self.create_model_id(model, dataset)
        score_ref = ray.put(scores)
        self.scores[id] = score_ref
        logger.info(f"Scores guardados para modelo {id}")
    
    def delete_model(self, model_id):
        """Eliminar modelo y sus scores del registry"""
        deleted = False
        
        # Eliminar del registry de modelos
        if model_id in self.model_registry:
            del self.model_registry[model_id]
            deleted = True
            logger.info(f"Modelo {model_id} eliminado del registry")
        
        # Eliminar scores asociados
        if model_id in self.scores:
            del self.scores[model_id]
            logger.info(f"Scores eliminados para modelo {model_id}")
        
        return deleted
    def get_registry_stats(self):
        """Obtener estadísticas del registry"""
        return {
            "total_models": len(self.model_registry),
            "model_ids": list(self.model_registry.keys())
        }


@ray.remote(num_cpus=1, max_retries=3)
def train_and_evaluate_model(task):
    """Entrena un solo modelo usando todos los datos"""
    model_name = task["model_name"]
    preprocessor = task["preprocessor"]
    model = task["model"]
    X_df_ref = task["X_ref"]
    y_series_ref = task["y_ref"]
    metrics_to_calc = task["metrics_to_calc"]
    problem_type = task["problem_type"]
    test_size = task["test_size"]
    random_state = task["random_state"]
    dataset_name = task["dataset_name"]
    model_manager = task["model_manager"]
    
    try:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        X_df = ray.get(X_df_ref)
        y_series = ray.get(y_series_ref)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=test_size, random_state=random_state,
            stratify=y_series if problem_type == "Clasificación" else None
        )

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        class_labels = task.get('class_labels')
        
        scores = {}
        for metric in metrics_to_calc:
            try:
                if problem_type == "Clasificación":
                    if metric == "matriz de confusion":
                        if class_labels:
                            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
                            scores["Confusion Matrix"] = {
                                "matrix": cm.tolist(),
                                "labels": class_labels
                            }
                        else:
                            scores["Confusion Matrix"] = "N/A (labels missing)"
                    elif metric == "Accuracy": 
                        scores[metric] = accuracy_score(y_test, y_pred)
                    elif "F1" in metric: 
                        scores[metric] = f1_score(y_test, y_pred, average='weighted')
                    elif "Precision" in metric: 
                        scores[metric] = precision_score(y_test, y_pred, average='weighted')
                    elif "Recall" in metric: 
                        scores[metric] = recall_score(y_test, y_pred, average='weighted')
                    elif metric == "ROC-AUC":
                        if hasattr(pipeline.named_steps['model'], "predict_proba"):
                            if isinstance(pipeline.named_steps['model'], (OneVsRestClassifier, OneVsOneClassifier)):
                                scores[metric] = "N/A for OvR/OvO wrapper"
                            else:
                                y_proba = pipeline.predict_proba(X_test)
                                if len(np.unique(y_test)) == 2:
                                    scores[metric] = roc_auc_score(y_test, y_proba[:, 1])
                                else:
                                    scores[metric] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                        else:
                            scores[metric] = "N/A (no predict_proba)"
                elif problem_type == "Regresión":
                    if metric == 'RMSE': 
                        scores[metric] = np.sqrt(mean_squared_error(y_test, y_pred))
                    elif metric == 'R2': 
                        scores[metric] = r2_score(y_test, y_pred)
                    elif metric == 'MAE': 
                        scores[metric] = mean_absolute_error(y_test, y_pred)
                    elif metric == 'MSE':
                        scores[metric] = mean_squared_error(y_test, y_pred)
            except Exception as e:
                scores[metric] = f"Metric Error: {str(e)}"

        scores['Training Time (s)'] = round(time.time() - start_time, 2)
        scores['Test Size'] = len(X_test)
        scores['Train Size'] = len(X_train)

        # Almacenar modelo en object store usando ModelManager
        model_id, model_ref = ray.get(model_manager.store_model.remote(model_name, dataset_name, pipeline))
        ray.get(model_manager.save_scores.remote(model_name, dataset_name, scores))
        logger.info(f"[SUCCESS] Modelo {model_name} entrenado y almacenado con ID: {model_id}")
        
        return {
            'model_name': model_name,
            'model_id': model_id,
            'scores': scores,
            'model_ref': model_ref,
            'status': 'Success'
        }
        
    except Exception as e:
        logger.error(f"[FAILED] Error entrenando {model_name}: {e}", exc_info=True)
        raise


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics, test_size, random_state, features_to_exclude, transform_target, selected_models, estrategia, dataset_name,model_manager):
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.metrics = metrics
        self.test_size = test_size
        self.random_state = random_state
        self.features_to_exclude = features_to_exclude
        self.transform_target = transform_target
        self.selected_models = selected_models
        self.estrategia = estrategia
        self.dataset_name = dataset_name.replace(".csv", "")
        self.train_result_ref = None
        
        self.model_manager = model_manager
        logger.info("ModelManager actor inicializado")

    def get_models(self):
        rs = self.random_state
        base_models = {
        # Clasificadores que soportan predict_proba por defecto o con ajustes
        "RandomForest": RandomForestClassifier(random_state=rs),
        "GradientBoosting": GradientBoostingClassifier(random_state=rs),
        "AdaBoost": AdaBoostClassifier(random_state=rs),
        "ExtraTrees": ExtraTreesClassifier(random_state=rs),
        "DecisionTree": DecisionTreeClassifier(random_state=rs),
        "LogisticRegression": LogisticRegression(random_state=rs, max_iter=1000),
        "SGD": SGDClassifier(random_state=rs, loss='log_loss'),  
        "SVM": SVC(probability=True, random_state=rs), 
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP": MLPClassifier(random_state=rs, max_iter=500),
        
        "PassiveAggressive": PassiveAggressiveClassifier(random_state=rs), 
        "KNN": KNeighborsClassifier(),  
        "LinearSVM": LinearSVC(random_state=rs, dual="auto"), 
        
        # Regresores (no aplican para probabilidades)
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=rs),
        "Lasso": Lasso(random_state=rs),
        "RandomForestRegressor": RandomForestRegressor(random_state=rs),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=rs),
    }
        
        selected_base_models = {name: model for name, model in base_models.items() if name in self.selected_models}
        logger.info(f"Modelos seleccionados: {list(selected_base_models.keys())}")
        
        if self.problem_type == "Clasificación" and self.estrategia and len(np.unique(self.df[self.target_column])) > 2:
            multiclass_models = {}
            for strategy in self.estrategia:
                for name, model in selected_base_models.items():
                    if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                        if strategy == "One-vs-Rest": 
                            multiclass_models[f"{name}_OvR"] = OneVsRestClassifier(model)
                        elif strategy == "One-vs-One": 
                            multiclass_models[f"{name}_OvO"] = OneVsOneClassifier(model)
            return multiclass_models
        
        return selected_base_models
    
    def save_results(self, results):
        """Guardar resultados simplificados"""
        d = {}
        if self.train_result_ref:
            d = ray.get(self.train_result_ref)
        
        for result in results:
            if result.get('status') == 'Success':
                # Los modelos ya están almacenados en ModelManager, solo guardamos metadatos
                model_data = {
                    'model_name': result['model_name'],
                    'model_id': result['model_id'],
                    'scores': result['scores'],
                    'status': result['status'],
                    'shape_dataset': self.df.shape,
                    'columns': list(self.df.columns),
                    'data_type': [str(x) for x in self.df.dtypes.tolist()],
                    'columns_to_exclude': self.features_to_exclude,
                    'target': self.target_column,
                    'dataset_name': self.dataset_name
                }
                d[result['model_name']] = model_data
                
        self.train_result_ref = ray.put(d)

    def get_model_by_name(self, model_name):
        """Obtener modelo específico por nombre usando ModelManager"""
        try:
            model_ref = ray.get(self.model_manager.get_model.remote(model_name, self.dataset_name))
            if model_ref:
                return ray.get(model_ref)
            else:
                logger.warning(f"Modelo {model_name} no encontrado para dataset {self.dataset_name}")
                return None
        except Exception as e:
            logger.error(f"Error obteniendo modelo {model_name}: {e}")
            return None
    
    def search_models(self, model_name_pattern=None, dataset_name_pattern=None):
        """Buscar modelos usando patrones"""
        try:
            matching_ids = ray.get(self.model_manager.search_models.remote(
                model_name=model_name_pattern, 
                dataset_name=dataset_name_pattern
            ))
            return matching_ids
        except Exception as e:
            logger.error(f"Error buscando modelos: {e}")
            return []
    
    def list_all_models(self):
        """Listar todos los IDs de modelos disponibles"""
        try:
            all_ids = ray.get(self.model_manager.list_all_model_ids.remote())
            return all_ids
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            return []
    
    def get_model_registry_stats(self):
        """Obtener estadísticas del registry de modelos"""
        try:
            stats = ray.get(self.model_manager.get_registry_stats.remote())
            return stats
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"total_models": 0, "model_ids": []}
    
    def remove_model(self, model_name):
        """Remover modelo del registry"""
        try:
            removed = ray.get(self.model_manager.remove_model.remote(model_name, self.dataset_name))
            if removed:
                logger.info(f"Modelo {model_name} removido exitosamente")
            return removed
        except Exception as e:
            logger.error(f"Error removiendo modelo {model_name}: {e}")
            return False


    def train(self):
        try:
            y = self.df[self.target_column]
            X = self.df.drop(columns=[self.target_column] + self.features_to_exclude)
            
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            preprocessor_standard = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features), 
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                ], 
                remainder='passthrough'
            )
            
            preprocessor_non_negative = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numeric_features), 
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                ], 
                remainder='passthrough'
            )

            non_negative_models = {"MultinomialNB", "ComplementNB"}
            logger.info(f"Se usará MinMaxScaler para los modelos: {non_negative_models}")

            models_to_train = self.get_models()
            class_labels = None
            if self.problem_type == "Clasificación":
                class_labels = sorted(y.unique().tolist())
 
            X_ref = ray.put(X)
            y_ref = ray.put(y)

            futures = []
            for name, model in models_to_train.items():
                base_model_name = name.split('_')[0]
                
                if base_model_name in non_negative_models:
                    preprocessor_for_task = preprocessor_non_negative
                else:
                    preprocessor_for_task = preprocessor_standard

                task = {
                    "model_name": name,
                    "preprocessor": preprocessor_for_task, 
                    "model": model,
                    "X_ref": X_ref,  
                    "y_ref": y_ref,  
                    'class_labels': class_labels,
                    "metrics_to_calc": self.metrics,       
                    "problem_type": self.problem_type,
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "dataset_name": self.dataset_name,
                    "model_manager": self.model_manager  # Pasar ModelManager a la tarea
                }
                future = train_and_evaluate_model.remote(task)
                futures.append(future)
            
            logger.info(f"Iniciando entrenamiento de {len(futures)} modelos...")
            results = ray.get(futures)
            
            successful_results = [result for result in results if result.get('status') == 'Success']
            logger.info(f"Entrenamiento completado. {len(successful_results)} modelos entrenados exitosamente.")
            
            self.save_results(successful_results)
            
            # Devolver directamente el diccionario de resultados, no envuelto en lista
            if self.train_result_ref:
                return ray.get(self.train_result_ref)
            else:
                # Convertir successful_results a formato diccionario si no hay train_result_ref
                result_dict = {}
                for result in successful_results:
                    if 'model_name' in result:
                        result_dict[result['model_name']] = result
                return result_dict
            
        except Exception as e:
            logger.error(f"Error fatal en el Trainer: {e}", exc_info=True)
            return [{"status": "Fatal Error", "error": str(e)}]




def create_global_model_manager():
    """Crear un ModelManager global para usar en toda la aplicación"""
    return ModelManager.options(name="model_manager",get_if_exists=True,lifetime="detached").remote()

def search_and_load_model(model_manager, model_name, dataset_name):
    """Función de conveniencia para buscar y cargar un modelo específico"""
    try:
        # Crear el ID esperado
        expected_id = f"{dataset_name}_{model_name}"
        
        # Verificar si existe
        all_ids = ray.get(model_manager.list_all_model_ids.remote())
        
        if expected_id in all_ids:
            model_ref = ray.get(model_manager.get_model.remote(model_name, dataset_name))
            if model_ref:
                model = ray.get(model_ref)
                logger.info(f"Modelo {expected_id} cargado exitosamente")
                return model
        
        logger.warning(f"Modelo {expected_id} no encontrado. Modelos disponibles: {all_ids}")
        return None
        
    except Exception as e:
        logger.error(f"Error buscando/cargando modelo {model_name} para dataset {dataset_name}: {e}")
        return None

def predict_with_stored_model(model_manager, model_name, dataset_name, X_new):
    """Realizar predicciones usando un modelo almacenado"""
    try:
        model = search_and_load_model(model_manager, model_name, dataset_name)
        if model:
            predictions = model.predict(X_new)
            logger.info(f"Predicciones realizadas con modelo {model_name}")
            return predictions
        else:
            logger.error(f"No se pudo cargar el modelo {model_name} para hacer predicciones")
            return None
    except Exception as e:
        logger.error(f"Error realizando predicciones: {e}")
        return None



