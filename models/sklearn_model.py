from sklearn.base import BaseEstimator
from typing import Any, Dict, Optional, Union
import ray
from models.base_model import Model

class SklearnModel(Model):
    """A wrapper for scikit-learn models that implements the Model interface."""
    
    def __init__(self, estimator: BaseEstimator, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a scikit-learn model wrapper.
        
        Args:
            estimator: The scikit-learn estimator class (not an instance)
            model_params: Parameters to initialize the estimator with
        """
        self.estimator_class = estimator
        self.model_params = model_params or {}
        self.model = self.estimator_class(**self.model_params)
        
    def fit(self, X, y, **kwargs):
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments to pass to the estimator's fit method
        
        Returns:
            self: The trained model
        """
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to make predictions on
        
        Returns:
            Predictions for the provided features
        """
        return self.model.predict(X)
    
    def get_params(self):
        """Get the parameters of this model."""
        return {
            "estimator_class": self.estimator_class,
            "model_params": self.model_params
        }
    
    @ray.remote
    def train_distributed(cls, estimator, model_params, X, y, **kwargs):
        """
        A Ray remote function to train a model in a distributed manner.
        
        Args:
            estimator: The scikit-learn estimator class
            model_params: Parameters for the estimator
            X: Training features
            y: Training labels
            **kwargs: Additional arguments for the fit method
            
        Returns:
            A trained SklearnModel instance
        """
        model = cls(estimator, model_params)
        return model.fit(X, y, **kwargs)