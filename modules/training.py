import ray
import pandas as pd
import numpy as np
import time
import warnings
from ray.exceptions import RayActorError
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, 
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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



@ray.remote(num_cpus=1, max_retries=3)
def train_and_evaluate_model(
         task
    ):
        model_name = task["model_name"]
        fold_idx = task["fold_idx"]
        preprocessor = task["preprocessor"]
        model = task["model"]
        X_df_ref = task["X_ref"]
        y_series_ref = task["y_ref"]
        train_indices = task["train_indices"]
        val_indices = task["val_indices"]
        metrics_to_calc = task["metrics_to_calc"]
        problem_type = task["problem_type"]
        try:
            
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            X_df = ray.get(X_df_ref)
            y_series = ray.get(y_series_ref)

            X_train, y_train = X_df.iloc[train_indices], y_series.iloc[train_indices]
            X_val, y_val = X_df.iloc[val_indices], y_series.iloc[val_indices]

            start_time = time.time()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            scores = {}
            for metric in metrics_to_calc:
                try:
                    if problem_type == "Clasificación":
                        if metric == "Accuracy": scores[metric] = accuracy_score(y_val, y_pred)
                        elif "F1" in metric: scores[metric] = f1_score(y_val, y_pred, average='weighted')
                        elif "Precision" in metric: scores[metric] = precision_score(y_val, y_pred, average='weighted')
                        elif "Recall" in metric: scores[metric] = recall_score(y_val, y_pred, average='weighted')
                        elif metric == "ROC-AUC":
                            if hasattr(pipeline.named_steps['model'], "predict_proba"):
                                if isinstance(pipeline.named_steps['model'], (OneVsRestClassifier, OneVsOneClassifier)):
                                    scores[metric] = "N/A for OvR/OvO wrapper"
                                else:
                                    y_proba = pipeline.predict_proba(X_val)
                                    if len(np.unique(y_val)) == 2:
                                        scores[metric] = roc_auc_score(y_val, y_proba[:, 1])
                                    else:
                                        scores[metric] = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
                            else:
                                scores[metric] = "N/A (no predict_proba)"
                    elif problem_type == "Regresión":
                        if metric == 'RMSE': scores[metric] = np.sqrt(mean_squared_error(y_val, y_pred))
                        elif metric == 'R2': scores[metric] = r2_score(y_val, y_pred)
                        elif metric == 'MAE': scores[metric] = mean_absolute_error(y_val, y_pred)
                except Exception as e:
                    scores[metric] = f"Metric Error: {str(e)}"

            scores['Training Time (s)'] = round(time.time() - start_time, 2)
            logger.info(f"[SUCCESS] Actor terminó {model_name} - Fold {fold_idx+1}.")
            return {'model_name':model_name,'fold_idx':fold_idx,'shape_dataset':X_df.shape,"scores":scores}
        except Exception as e:
            logger.error(f"[FAILED] Actor falló en {model_name} - Fold {fold_idx+1}. Error: {e}", exc_info=True)
            raise


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics, test_size, cv_folds, random_state, features_to_exclude, transform_target, selected_models, estrategia):
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.metrics = metrics
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.features_to_exclude = features_to_exclude
        self.transform_target = transform_target
        self.selected_models = selected_models
        self.estrategia = estrategia

    def get_models(self):
        rs = self.random_state
        base_models = {
            "RandomForest": RandomForestClassifier(random_state=rs), "GradientBoosting": GradientBoostingClassifier(random_state=rs),
            "AdaBoost": AdaBoostClassifier(random_state=rs), "ExtraTrees": ExtraTreesClassifier(random_state=rs),
            "DecisionTree": DecisionTreeClassifier(random_state=rs), "LogisticRegression": LogisticRegression(random_state=rs, max_iter=1000),
            "SGD": SGDClassifier(random_state=rs), "PassiveAggressive": PassiveAggressiveClassifier(random_state=rs),
            "KNN": KNeighborsClassifier(), "SVM": SVC(probability=True, random_state=rs),
            "LinearSVM": LinearSVC(random_state=rs, dual="auto"), "GaussianNB": GaussianNB(),
            "BernoulliNB": BernoulliNB(), "MultinomialNB": MultinomialNB(), "ComplementNB": ComplementNB(),
            "LDA": LinearDiscriminantAnalysis(), "QDA": QuadraticDiscriminantAnalysis(), "MLP": MLPClassifier(random_state=rs, max_iter=500),
            "LinearRegression": LinearRegression(), "Ridge": Ridge(random_state=rs), "Lasso": Lasso(random_state=rs),
            "RandomForestRegressor": RandomForestRegressor(random_state=rs), "GradientBoostingRegressor": GradientBoostingRegressor(random_state=rs),
        }
        
        selected_base_models = {name: model for name, model in base_models.items() if name in self.selected_models}
        logger.info(f"Modelos seleccionados: {list(selected_base_models.keys())}")
        if self.problem_type == "Clasificación" and self.estrategia and len(np.unique(self.df[self.target_column])) > 2:
            multiclass_models = {}
            for strategy in self.estrategia:
                for name, model in selected_base_models.items():
                    if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                        if strategy == "One-vs-Rest": multiclass_models[f"{name}_OvR"] = OneVsRestClassifier(model)
                        elif strategy == "One-vs-One": multiclass_models[f"{name}_OvO"] = OneVsOneClassifier(model)
            return multiclass_models
        
        return selected_base_models
   
    def train(self):
        try:
            y = self.df[self.target_column]
            X = self.df.drop(columns=[self.target_column] + self.features_to_exclude)
            
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            models_to_train = self.get_models()
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
            
            if self.problem_type == "Clasificación":
                kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                split_generator = list(kf.split(X, y))
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                split_generator = list(kf.split(X))
 
            X_ref = ray.put(X)
            y_ref = ray.put(y)

            futures = []
            for name, model in models_to_train.items():

                base_model_name = name.split('_')[0]
                
                if base_model_name in non_negative_models:
                    preprocessor_for_task = preprocessor_non_negative
                else:
                    preprocessor_for_task = preprocessor_standard

                for fold_idx, (train_indices, val_indices) in enumerate(split_generator):
                    task = {
                        "model_name": name,
                        "fold_idx": fold_idx,
                        "preprocessor": preprocessor_for_task, 
                        "model": model,
                        "X_ref": X_ref,  
                        "y_ref": y_ref,  
                        "train_indices": train_indices,
                        "val_indices": val_indices,
                        "metrics_to_calc": self.metrics,       
                        "problem_type": self.problem_type, 
                    }
                    future = train_and_evaluate_model.remote(task)
                    futures.append(future)
            
            results = []
            active_futures = {}
            
            completed, failed = 0,0
            while futures:
                try:
                    ready,not_ready = ray.wait(futures)
                    for result in ready:
                        try:
                            info = ray.get(result)
                            logger.info(info)
                            active_futures[info['model_name']]=info
                            results.append({'status':'Success',**info})
                            completed+=1
                        except Exception as e:
                            failed+=1
                    futures = not_ready
                    if len(futures)>0 and len(ready)==0:
                        time.sleep(5)
                except Exception as e:
                    logger.info(e)
                    logger.info("ocurrio un error esperando resultados")
                    break
            logger.info("completado")
            raw_results_by_model = defaultdict(list)
            for res in results:
                 if res.get('status') == 'Success':
                    raw_results_by_model[res['model_name']].append(res['scores'])
            
            logger.info("Entrenamiento y evaluación completados. Procesando resultados...")
            logger.info(f"Resultados obtenidos: {len(raw_results_by_model)} modelos procesados con éxito en al menos un fold.")

            final_results = []
            for model_name, fold_scores_list in raw_results_by_model.items():
                if not fold_scores_list:
                    continue 
                
                scores_df = pd.DataFrame(fold_scores_list)
                aggregated_scores = {}
                for metric_name in scores_df.columns:
                    numeric_series = pd.to_numeric(scores_df[metric_name], errors='coerce')
                    if numeric_series.isnull().all():
                        first_value = scores_df[metric_name].mode()
                        aggregated_scores[metric_name] = first_value.iloc[0] if not first_value.empty else "N/A"
                    else:
                        mean_val = numeric_series.mean()
                        std_val = numeric_series.std()
                        aggregated_scores[f"{metric_name}_mean"] = round(mean_val, 4) if pd.notna(mean_val) else 'N/A'
                        aggregated_scores[f"{metric_name}_std"] = round(std_val, 4) if pd.notna(std_val) else 'N/A'
                final_results.append({"model": model_name, "status": "Success", "scores": aggregated_scores})

            return final_results

        except Exception as e:
            logger.error(f"Error fatal en el Trainer: {e}", exc_info=True)
            return [{"status": "Fatal Error", "error": str(e)}]