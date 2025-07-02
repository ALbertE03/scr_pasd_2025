import ray
import pandas as pd
import numpy as np
import time
import warnings
import json
import shutil
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
from collections import defaultdict
from datetime import datetime
import pickle 
import os  
import subprocess
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



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
    
    try:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        X_df = ray.get(X_df_ref)
        y_series = ray.get(y_series_ref)

        # Dividir en train y test para evaluación
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=test_size, random_state=random_state,
            stratify=y_series if problem_type == "Clasificación" else None
        )

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        
        # Hacer predicciones en el conjunto de test
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
        base_results_dir = f'train_results/{dataset_name}'
        models_dir = os.path.join(base_results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        logger.info(f"[SUCCESS] Modelo {model_name} entrenado y guardado en {model_path}")
        
        return {
            'model_name': model_name,
            'scores': scores,
            'model_path': model_path,
            'status': 'Success'
        }
        
    except Exception as e:
        logger.error(f"[FAILED] Error entrenando {model_name}: {e}", exc_info=True)
        raise


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics, test_size, random_state, features_to_exclude, transform_target, selected_models, estrategia, dataset_name):
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
        "SGD": SGDClassifier(random_state=rs, loss='log_loss'),  # Requiere loss='log' o 'modified_huber' para predict_proba
        "SVM": SVC(probability=True, random_state=rs),  # Necesita probability=True
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP": MLPClassifier(random_state=rs, max_iter=500),
        
        # Clasificadores que NO soportan predict_proba (se excluyen o se dejan sin cambios)
        "PassiveAggressive": PassiveAggressiveClassifier(random_state=rs),  # No soporta probabilidades
        "KNN": KNeighborsClassifier(),  # Soporta predict_proba por defecto
        "LinearSVM": LinearSVC(random_state=rs, dual="auto"),  # No soporta predict_proba
        
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
        base_results_dir = f'train_results/{self.dataset_name}'
        os.makedirs(base_results_dir, exist_ok=True)
        logger.info(f"Saving results to: {base_results_dir}")

        # Guardar resultados en JSON
        d = {}
        try:
            with open(os.path.join('train_results', self.dataset_name, f'training_{self.dataset_name}_result.json'), 'r') as f:
                d = json.load(f)
        except:
            d = {}
 
        for result in results:
            if result.get('status') == 'Success':
                d[result['model_name']] = {
                    **result,
                    'shape_dataset': self.df.shape,
                    'columns': list(self.df.columns),
                    'data_type': [str(x) for x in self.df.dtypes.tolist()],
                    'columns_to_exclude': self.features_to_exclude,
                    'target': self.target_column
                }
                
        with open(os.path.join('train_results', self.dataset_name, f'training_{self.dataset_name}_result.json'), 'w') as f:
            json.dump(d, f)


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
                    "dataset_name": self.dataset_name
                }
                future = train_and_evaluate_model.remote(task)
                futures.append(future)
            
            logger.info(f"Iniciando entrenamiento de {len(futures)} modelos...")
            results = ray.get(futures)
            
            successful_results = [result for result in results if result.get('status') == 'Success']
            logger.info(f"Entrenamiento completado. {len(successful_results)} modelos entrenados exitosamente.")
            
            self.save_results(successful_results)
            return successful_results
            
        except Exception as e:
            logger.error(f"Error fatal en el Trainer: {e}", exc_info=True)
            return [{"status": "Fatal Error", "error": str(e)}]



