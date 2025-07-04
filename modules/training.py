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
                                  LinearRegression, Ridge, Lasso, ElasticNet,
                                  HuberRegressor,RANSACRegressor,TheilSenRegressor,
                                  ARDRegression,BayesianRidge,PassiveAggressiveRegressor,SGDRegressor)
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC,SVR, LinearSVR
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                                HistGradientBoostingRegressor,
                              ExtraTreesClassifier, 
                              RandomForestRegressor, GradientBoostingRegressor,
                                GradientBoostingRegressor,
                                AdaBoostRegressor,
                                ExtraTreesRegressor,
                                BaggingRegressor,
                                VotingRegressor,)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@ray.remote(max_restarts=4, max_task_retries=4)
class ModelManager:
    """Actor para gestionar el mapeo de modelos en el Ray object store"""
    
    def __init__(self):
        self.model_registry = {} 
        self.scores={}
        self.dataset_atribuetes={}
        
    def create_model_id(self, model_name, dataset_name):
        """Crear ID único para el modelo basado en nombre del modelo y dataset"""
        return f"{dataset_name}_{model_name}"
    def save_atributes(self,name_model,dataset_name,**kwargs):
        """Guardar atributos adicionales del modelo"""
        id = self.create_model_id(name_model, dataset_name)
        kwargs_ref = ray.put(kwargs) 
        self.dataset_atribuetes[id] = kwargs_ref


    def get_atributes(self, id):
        return ray.get(self.dataset_atribuetes.get(id, {}))
    
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
            return ray.get(score_ref)  
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

        if model_id in self.model_registry:
            del self.model_registry[model_id]
            deleted = True
            logger.info(f"Modelo {model_id} eliminado del registry")

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
    start_overall_time = time.time()
    max_training_time = task.get("max_training_time", 120)  # 2 minutos por modelo por defecto
    
    try:
        preprocessor = task["preprocessor"]
        model = task["model"]
        data_type = task['data_type']
        X_df_ref = task["X_ref"]
        y_series_ref = task["y_ref"]
        metrics_to_calc = task["metrics_to_calc"]
        problem_type = task["problem_type"]
        test_size = task["test_size"]
        random_state = task["random_state"]
        dataset_name = task["dataset_name"]
        model_manager = task["model_manager"]
        columns = task.get('columns', [])
        target_column = task.get('target', None)
        features_to_exclude = task.get('columns_to_exclude', [])
        transform_target = task.get('transform_target', False)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        X_df = ray.get(X_df_ref)
        y_series = ray.get(y_series_ref)

     
        y_transformed = y_series.copy()
        log_transform_applied = False
        
        if transform_target and problem_type == "Regresión":
            try:
           
                if (y_series > 0).all():
                    y_transformed = np.log1p(y_series) 
                    log_transform_applied = True
                    logger.info(f"Transformación logarítmica aplicada al target para {model_name}")
                else:
                    logger.warning(f"No se puede aplicar transformación logarítmica a {model_name}: contiene valores <= 0")
            except Exception as e:
                logger.warning(f"Error aplicando transformación logarítmica a {model_name}: {e}")

        use_stratify = False
        if problem_type == "Clasificación":
            class_counts = pd.Series(y_transformed).value_counts()
            min_class_count = class_counts.min()
        
            min_test_samples_needed = len(class_counts)  
            total_test_samples = int(len(y_transformed) * test_size)
            
            if min_class_count >= 2 and total_test_samples >= min_test_samples_needed:
                use_stratify = True
                logger.info(f"Usando estratificación. Clases: {len(class_counts)}, Min muestras por clase: {min_class_count}")
            else:
                logger.warning(f"No se puede usar estratificación. Min muestras por clase: {min_class_count}, "
                             f"Muestras test necesarias: {min_test_samples_needed}, Disponibles: {total_test_samples}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_transformed, test_size=test_size, random_state=random_state,
            stratify=y_transformed if use_stratify else None
        )

        start_time = time.time()
        
        # Verificar timeout antes del entrenamiento
        elapsed_time = time.time() - start_overall_time
        if elapsed_time > max_training_time:
            raise TimeoutError(f"Timeout alcanzado antes del entrenamiento para {model_name}")
        
        logger.info(f"Iniciando entrenamiento de {model_name} (tiempo límite: {max_training_time}s)")
        pipeline.fit(X_train, y_train)
        
        # Verificar timeout después del entrenamiento
        elapsed_time = time.time() - start_overall_time
        if elapsed_time > max_training_time:
            logger.warning(f"Timeout alcanzado después del entrenamiento para {model_name}, continuando con predicciones...")
        
        y_pred = pipeline.predict(X_test)
        
        y_test_original = y_test.copy()
        y_pred_original = y_pred.copy()
        
        if log_transform_applied:
            try:
                y_test_original = np.expm1(y_test)  
                y_pred_original = np.expm1(y_pred)
                logger.info(f"Transformación logarítmica invertida para métricas de {model_name}")
            except Exception as e:
                logger.warning(f"Error invirtiendo transformación logarítmica en {model_name}: {e}")
        
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
                        scores[metric] = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                    elif metric == 'R2': 
                        scores[metric] = r2_score(y_test_original, y_pred_original)
                    elif metric == 'MAE': 
                        scores[metric] = mean_absolute_error(y_test_original, y_pred_original)
                    elif metric == 'MSE':
                        scores[metric] = mean_squared_error(y_test_original, y_pred_original)
                    elif metric == 'MAPE':
                        
                        scores[metric] = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            except Exception as e:
                scores[metric] = f"Metric Error: {str(e)}"

        scores['Training Time (s)'] = round(time.time() - start_time, 2)
        scores['Test Size'] = len(X_test)
        scores['Train Size'] = len(X_train)
        
        if log_transform_applied:
            scores['Log Transform Applied'] = True
        
        model_id, model_ref = ray.get(model_manager.store_model.remote(model_name, dataset_name, pipeline))
        ray.get(model_manager.save_scores.remote(model_name, dataset_name, scores))
        ray.get(model_manager.save_atributes.remote(
            model_name,
            dataset_name,
            columns=columns,
            target=target_column,
            columns_to_exclude=features_to_exclude,
            dataset=dataset_name,
            data_type=data_type,
            hyperparams_used=task.get('hyperparams_used', {}),
            log_transform_applied=log_transform_applied
        ))
        logger.info(f"[SUCCESS] Modelo {model_name} entrenado y almacenado con ID: {model_id}")
        
        return {
            'model_name': model_name,
            'model_id': model_id,
            'scores': scores,
            'model_ref': model_ref,
            'status': 'Success'
        }
        
    except TimeoutError as e:
        logger.error(f"[TIMEOUT] Timeout entrenando {model_name}: {e}")
        return {
            'model_name': model_name,
            'status': 'Timeout',
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"[FAILED] Error entrenando {model_name}: {e}", exc_info=True)
        return {
            'model_name': model_name,
            'status': 'Failed',
            'error': str(e)
        }


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics, test_size, random_state, features_to_exclude, transform_target, selected_models, estrategia, dataset_name, model_manager, hyperparams=None, training_timeout=300):
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
        self.hyperparams = hyperparams or {}
        self.training_timeout = training_timeout  
        
        self.model_manager = model_manager
        logger.info("ModelManager actor inicializado")
        logger.info(f"Hiperparámetros recibidos para {len(self.hyperparams)} modelos")
        logger.info(f"Timeout de entrenamiento configurado: {self.training_timeout} segundos")

    def apply_hyperparams_to_model(self, model_name, model):
        """Aplica hiperparámetros personalizados a un modelo"""
        if model_name in self.hyperparams:
            params = self.hyperparams[model_name]
            try:
            
                valid_params = {}
                model_params = model.get_params().keys()
                
                for param_name, param_value in params.items():
                    if param_name in model_params:
                        if param_value == "None":
                            valid_params[param_name] = None
                        elif param_value == "True":
                            valid_params[param_name] = True
                        elif param_value == "False":
                            valid_params[param_name] = False
                        elif isinstance(param_value, str) and param_value.replace('.', '').replace('-', '').isdigit():

                            if '.' in param_value:
                                valid_params[param_name] = float(param_value)
                            else:
                                valid_params[param_name] = int(param_value)
                        else:
                            valid_params[param_name] = param_value
                    else:
                        logger.warning(f"Parámetro '{param_name}' no válido para {model_name}. Parámetros disponibles: {list(model_params)}")
                
                if valid_params:
                    model.set_params(**valid_params)
                    logger.info(f"Aplicados hiperparámetros a {model_name}: {valid_params}")
                else:
                    logger.info(f"No se aplicaron hiperparámetros a {model_name} (ningún parámetro válido)")
                    
            except Exception as e:
                logger.warning(f"Error aplicando hiperparámetros a {model_name}: {e}")
        else:
            logger.info(f"No hay hiperparámetros personalizados para {model_name}, usando valores por defecto")
        
        return model

    def get_models(self):
        rs = self.random_state
        base_models = {
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
                
                # Regresores
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=rs),
                "Lasso": Lasso(random_state=rs),
                "ElasticNet": ElasticNet(random_state=rs),
                "HuberRegressor": HuberRegressor(),
                "RANSACRegressor": RANSACRegressor(random_state=rs),
                "TheilSenRegressor": TheilSenRegressor(random_state=rs),
                "ARDRegression": ARDRegression(),
                "BayesianRidge": BayesianRidge(),
                "PassiveAggressiveRegressor": PassiveAggressiveRegressor(random_state=rs),
                "SGDRegressor": SGDRegressor(random_state=rs),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=rs),
                "ExtraTreesRegressor": ExtraTreesRegressor(random_state=rs),
                "RandomForestRegressor": RandomForestRegressor(random_state=rs),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=rs),
                "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=rs),
                "AdaBoostRegressor": AdaBoostRegressor(random_state=rs),
                "SVR": SVR(),
                "LinearSVR": LinearSVR(random_state=rs),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "MLPRegressor": MLPRegressor(random_state=rs, max_iter=500),
                "BaggingRegressor": BaggingRegressor(random_state=rs),
                "VotingRegressor": VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor(random_state=rs))])
        }
                    
        selected_base_models = {name: model for name, model in base_models.items() if name in self.selected_models}
        logger.info(f"Modelos seleccionados: {list(selected_base_models.keys())}")
        
        for name, model in selected_base_models.items():
            selected_base_models[name] = self.apply_hyperparams_to_model(name, model)
        
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

                hyperparams_used = self.hyperparams.get(name, {})

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
                    "model_manager": self.model_manager,
                    'data_type':[str(x) for x in self.df.dtypes.tolist()],
                    'columns':self.df.columns.tolist(),
                    'target':self.target_column,
                    'columns_to_exclude':self.features_to_exclude,
                    'hyperparams_used': hyperparams_used,
                    'transform_target': self.transform_target,
                    'max_training_time': max(60, self.training_timeout // len(models_to_train))  # Distribuir tiempo entre modelos
                }
                future = train_and_evaluate_model.remote(task)
                futures.append(future)
            
            logger.info(f"Iniciando entrenamiento de {len(futures)} modelos...")
            
            # Obtener resultados con timeout
            try:
                timeout_seconds = self.training_timeout
                results = ray.get(futures, timeout=timeout_seconds)
                logger.info(f"Todos los modelos completados dentro del timeout de {timeout_seconds}s")
            except ray.exceptions.GetTimeoutError:
                logger.warning(f"Timeout de {timeout_seconds}s alcanzado. Obteniendo resultados parciales...")
                ready_futures, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=0)
                
                results = []
                # Obtener resultados de las tareas completadas
                if ready_futures:
                    completed_results = ray.get(ready_futures)
                    results.extend(completed_results)
                    logger.info(f"Obtenidos {len(completed_results)} resultados completados")
                
                # Cancelar tareas restantes
                if remaining_futures:
                    logger.warning(f"Cancelando {len(remaining_futures)} tareas no completadas")
                    for future in remaining_futures:
                        ray.cancel(future)
            
            successful_results = [result for result in results if result.get('status') == 'Success']
            timeout_results = [result for result in results if result.get('status') == 'Timeout']
            failed_results = [result for result in results if result.get('status') == 'Failed']
            
            logger.info(f"Entrenamiento completado. {len(successful_results)} modelos exitosos, "
                       f"{len(timeout_results)} timeouts, {len(failed_results)} fallos de {len(futures)} iniciados.")
            
            if timeout_results:
                timeout_models = [r['model_name'] for r in timeout_results]
                logger.warning(f"Modelos que excedieron el timeout: {timeout_models}")
            
            if failed_results:
                failed_models = [r['model_name'] for r in failed_results]
                logger.warning(f"Modelos que fallaron: {failed_models}")
            
            self.save_results(successful_results)

            if self.train_result_ref:
                return ray.get(self.train_result_ref)
            else:

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

        expected_id = f"{dataset_name}_{model_name}"

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



