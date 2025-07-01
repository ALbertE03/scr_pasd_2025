import ray
import pandas as pd
import numpy as np
import time
import warnings
import shutil
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
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
from sklearn.ensemble import VotingClassifier, VotingRegressor
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
            os.makedirs('temp', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'temp/{model_name}_{timestamp}_fold{fold_idx+1}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            y_pred = pipeline.predict(X_val)
            class_labels = task['class_labels']
            scores = {}
            for metric in metrics_to_calc:
                try:
                    
                    if problem_type == "Clasificación":
                        if metric == "matriz de confusion":
                            if class_labels:
                                cm = confusion_matrix(y_val, y_pred, labels=class_labels)
                                scores[metric] = cm.tolist()
                            else:
                                scores[metric] = "N/A (labels missing)"
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
            return {'model_name':model_name,'fold_idx':fold_idx,'shape_dataset':X_df.shape,"scores":scores,"path":model_path}
        except Exception as e:
            
            logger.error(f"[FAILED] Actor falló en {model_name} - Fold {fold_idx+1}. Error: {e}", exc_info=True)
            raise


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics, test_size, cv_folds, random_state, features_to_exclude, transform_target, selected_models, estrategia,dataset_name):
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
        self.dataset_name = dataset_name

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
    
    def save(self, raw_fold_results, aggregated_results):
    
        base_results_dir = f'train_results/{self.dataset_name}'
        models_dir = os.path.join(base_results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Saving results and models to: {base_results_dir}")

        for result in raw_fold_results:
            if result.get('status') == 'Success' and 'path' in result and result['path'] is not None:
                temp_path = result['path']
                if os.path.exists(temp_path):
                    model_name = result['model_name']
                    fold_idx = result['fold_idx']
                    new_filename = f"{model_name}_fold{fold_idx+1}.pkl"
                    permanent_path = os.path.join(models_dir, new_filename)
                    try:
                        shutil.move(temp_path, permanent_path)
                        result['path'] = permanent_path 
                    except Exception as e:
                        logger.error(f"Failed to move model file {temp_path}. Error: {e}")
        
        summary_path = os.path.join(base_results_dir, 'summary_results.csv')
        summary_df = pd.DataFrame(aggregated_results)
        if not summary_df.empty and 'scores' in summary_df.columns:
            scores_df = pd.json_normalize(summary_df['scores'])
            summary_df = pd.concat([summary_df.drop(columns=['scores']), scores_df], axis=1)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved aggregated results summary to {summary_path}")

        raw_results_path = os.path.join(base_results_dir, 'raw_fold_results.pkl')
        with open(raw_results_path, 'wb') as f:
            pickle.dump(raw_fold_results, f)
        logger.info(f"Saved raw fold-by-fold results to {raw_results_path}")

        if os.path.exists('temp'):
            try:
                shutil.rmtree('temp')
                logger.info("Cleaned up temporary 'temp' directory.")
            except Exception as e:
                logger.warning(f"Could not remove 'temp' directory. It may contain files from other processes. Error: {e}")


    def agregate(self):
        agrega = defaultdict(list)
        path = os.path.join("train_results",self.dataset_name,'models')
        models= os.listdir(path)
        for model in models:
                name = model.split("_")[0]
                agrega[name].append(model)
        logger.info(f"Modelos agregados: {agrega}")

        return agrega
    def create_voting_ensemble(self, voting_type='hard', weights=None):
        """
        Crea un ensemble por votación para cada grupo de modelos
        
        Args:
            voting_type (str): 'soft' o 'hard' (solo para clasificación)
            weights (list): Lista de pesos para los modelos (opcional)
        """
        models_dict = self.agregate()
        ensembles = {}
        path = os.path.join("train_results",self.dataset_name,'models')
        for model_name, models in models_dict.items():
            if not models:
                continue
 
            estimators = [(f"{model_name}_fold{idx}", model) 
                         for idx, model in enumerate(models)]
            
            try:
                if self.problem_type == 'Clasificación':
                        
                    ensemble = VotingClassifier(
                        estimators=estimators,
                        voting=voting_type,
                        weights=weights,
                        n_jobs=-1
                    )
                else:
                    
                    ensemble = VotingRegressor(
                        estimators=estimators,
                        weights=weights,
                        n_jobs=-1
                    )
                
                ensemble_name = f"{model_name}_voting_ensemble"
                ensemble_path = os.path.join(path, f"{ensemble_name}.pkl")
                with open(ensemble_path, 'wb') as f: 
                    pickle.dump(ensemble, f)

                ensembles[ensemble_name] = {
                    'base_models': [name for name, _ in estimators],
                    'ensemble_path': ensemble_path,
                    'voting_type': voting_type if self.problem_type == 'Clasificación' else None,
                    'weights': weights
                }
                
                print(f"Ensemble creado: {ensemble_name} con {len(models)} modelos base")
            
            except Exception as e:
                print(f"Error creando ensemble para {model_name}: {str(e)}")
                continue
        import re 
        for i in os.listdir(path):
                if re.match(r'.*fold\d*.pkl$', i):
                    os.remove(os.path.join(path,i))     
        return ensembles
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
            class_labels = None
            if self.problem_type == "Clasificación":
                class_labels = sorted(y.unique().tolist())
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
                        'class_labels':class_labels,
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
                if 'matriz de confusion' in self.metrics and class_labels:
                    cm_list = [
                        np.array(scores['matriz de confusion'])
                        for scores in fold_scores_list
                        if 'matriz de confusion' in scores and isinstance(scores['matriz de confusion'], list)
                    ]
                    if cm_list:
                        sum_cm = np.sum(cm_list, axis=0)
                        aggregated_scores['Confusion Matrix'] = {
                            "matrix": sum_cm.astype(int).tolist(),
                            "labels": class_labels
                        }

                scores_df = pd.DataFrame(fold_scores_list)
                for metric_name in scores_df.columns:
                    if metric_name == 'matriz de confusion': continue
                    
                    numeric_series = pd.to_numeric(scores_df[metric_name], errors='coerce')
                    if not numeric_series.isnull().all():
                        aggregated_scores[f"{metric_name}_mean"] = round(numeric_series.mean(), 4)
                        aggregated_scores[f"{metric_name}_std"] = round(numeric_series.std(), 4)
                    else:
                        aggregated_scores[metric_name] = scores_df[metric_name].mode().iloc[0] if not scores_df[metric_name].mode().empty else "N/A"
                        
                final_results.append({"model": model_name, "status": "Success", "scores": aggregated_scores})
            self.save(results, final_results)
            self.create_voting_ensemble()
            return final_results
        except Exception as e:
            logger.error(f"Error fatal en el Trainer: {e}", exc_info=True)
            return [{"status": "Fatal Error", "error": str(e)}]

        