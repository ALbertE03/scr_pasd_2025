import ray
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@ray.remote
def train_and_evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_calc: list,
    problem_type: str
):
    try:
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        scores = {}
        for metric in metrics_to_calc:
            try:
                if problem_type == "Clasificación":
                    if metric == "Accuracy": scores[metric] = accuracy_score(y_test, y_pred)
                    elif "F1" in metric: scores[metric] = f1_score(y_test, y_pred, average='weighted')
                    elif "Precision" in metric: scores[metric] = precision_score(y_test, y_pred, average='weighted')
                    elif "Recall" in metric: scores[metric] = recall_score(y_test, y_pred, average='weighted')
                    elif metric == "ROC-AUC":
                        if hasattr(pipeline.named_steps['model'], "predict_proba"):
                            if isinstance(pipeline.named_steps['model'], (OneVsRestClassifier, OneVsOneClassifier)):
                                scores[metric] = "N/A for OvR/OvO wrapper"
                            else:
                                y_proba = pipeline.predict_proba(X_test)
                                if y_proba.shape[1] == 2:
                                    scores[metric] = roc_auc_score(y_test, y_proba[:, 1])
                                else:
                                    scores[metric] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        else:
                            scores[metric] = "N/A (no predict_proba)"
                elif problem_type == "Regresión":
                    if metric == 'RMSE': scores[metric] = np.sqrt(mean_squared_error(y_test, y_pred))
                    elif metric == 'R2': scores[metric] = r2_score(y_test, y_pred)
                    elif metric == 'MAE': scores[metric] = mean_absolute_error(y_test, y_pred)
            except Exception as e:
                scores[metric] = f"Error: {str(e)}"

        scores['Training Time (s)'] = round(time.time() - start_time, 2)
        print(f"[SUCCESS] Model {model_name} trained. Scores: {scores}")
        return {"model": model_name, "status": "Success", "scores": scores, "pipeline": pipeline}
    except Exception as e:
        print(f"[FAILED] Model {model_name} failed. Error: {str(e)}")
        return {"model": model_name, "status": "Failed", "error": str(e)}


class Trainer:
    def __init__(self, df, target_column, problem_type, metrics,  test_size,cv_folds, random_state, features_to_exclude, transform_target, selected_models, estrategia):
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
        
        if self.problem_type == "Clasificación" and self.estrategia:
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

    def train(self):
        try:
            y = self.df[self.target_column]
            X = self.df.drop(columns=[self.target_column] + self.features_to_exclude)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if self.problem_type == "Clasificación" else None)
            if self.transform_target and self.problem_type == "Regresión":
                try:
                    X_train = np.log1p(X_train)
                except:
                    X_train = X_train
            numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)], remainder='passthrough')
            
            models_to_train = self.get_models()
            if not models_to_train:
                raise ValueError("No se seleccionaron modelos válidos para entrenar.")

            X_train_ref, y_train_ref, X_test_ref, y_test_ref = ray.put(X_train), ray.put(y_train), ray.put(X_test), ray.put(y_test)
            
            results_futures = []
            for name, model in models_to_train.items():
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                future = train_and_evaluate_model.remote(name, pipeline, X_train_ref, y_train_ref, X_test_ref, y_test_ref, self.metrics, self.problem_type)
                results_futures.append(future)

            raw_results = ray.get(results_futures)

            final_results = []
            for res in raw_results:
                if 'pipeline' in res:
                    del res['pipeline']
                final_results.append(res)

            return final_results
        except Exception as e:
            print(f"ERROR FATAL en Trainer.train: {e}")
            logger.error(f"ERROR FATAL en Trainer.train: {e}")
           
