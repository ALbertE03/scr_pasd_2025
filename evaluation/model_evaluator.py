import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.model_selection import cross_val_score
import ray

class ModelEvaluator:
    """Evaluates machine learning models using various metrics."""
    
    @staticmethod
    def evaluate(model: Any, X: np.ndarray, y: np.ndarray, 
                 metrics: Optional[List[str]] = None, 
                 task_type: str = "classification") -> Dict[str, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Trained model that implements predict method
            X: Features for evaluation
            y: True labels
            metrics: List of metric names to compute
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary of metric name to score
        """
        if metrics is None:
            if task_type == "classification":
                metrics = ["accuracy", "precision", "recall", "f1"]
            else:
                metrics = ["r2", "mse", "mae"]
        
        y_pred = model.predict(X)
        results = {}
        
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = float(accuracy_score(y, y_pred))
            elif metric == "precision":
                results[metric] = float(precision_score(y, y_pred, average='weighted'))
            elif metric == "recall":
                results[metric] = float(recall_score(y, y_pred, average='weighted'))
            elif metric == "f1":
                results[metric] = float(f1_score(y, y_pred, average='weighted'))
            elif metric == "r2":
                results[metric] = float(r2_score(y, y_pred))
            elif metric == "mse":
                results[metric] = float(np.mean((y - y_pred) ** 2))
            elif metric == "mae":
                results[metric] = float(np.mean(np.abs(y - y_pred)))
        
        return results
    
    @staticmethod
    @ray.remote
    def evaluate_distributed(model: Any, X: np.ndarray, y: np.ndarray,
                            metrics: Optional[List[str]] = None,
                            task_type: str = "classification") -> Dict[str, float]:
        """
        Distributed version of evaluate method using Ray.
        
        Args:
            model: Trained model that implements predict method
            X: Features for evaluation
            y: True labels
            metrics: List of metric names to compute
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary of metric name to score
        """
        return ModelEvaluator.evaluate(model, X, y, metrics, task_type)
    
    @staticmethod
    def cross_validate(model_factory: Callable[[], Any], X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: Union[str, List[str]] = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_factory: Function that returns a new model instance
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric(s) to use
            
        Returns:
            Dictionary of metric name to mean score
        """
        model = model_factory()
        
        if isinstance(scoring, list):
            results = {}
            for score in scoring:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=score)
                results[score] = float(np.mean(cv_scores))
            return results
        else:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return {scoring: float(np.mean(cv_scores))}