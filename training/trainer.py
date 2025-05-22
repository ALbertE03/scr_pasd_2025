import ray
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from models.sklearn_model import SklearnModel
from utils.data_loader import DataLoader

class DistributedTrainer:
    """Manages the distributed training of multiple models."""
    
    def __init__(self, init_ray: bool = True):
        """
        Initialize the distributed trainer.
        
        Args:
            init_ray: Whether to initialize Ray (set to False if Ray is already initialized)
        """
        if init_ray:
            ray.init()
    
    def train_models(self, 
                     dataset_path: str,
                     target_column: str,
                     models_config: List[Dict[str, Any]],
                     preprocessing_config: Optional[Dict[str, Any]] = None,
                     **dataset_kwargs) -> Dict[str, Any]:
        """
        Train multiple models in parallel on a dataset.
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Name of the target column
            models_config: List of model configurations, each containing:
                - 'name': Model name
                - 'estimator': Scikit-learn estimator class
                - 'params': Parameters for the estimator
            preprocessing_config: Configuration for data preprocessing
            **dataset_kwargs: Additional arguments for dataset loading
            
        Returns:
            Dictionary of trained models referenced by name
        """
        # Load and preprocess data in a distributed manner
        X_ref, y_ref = DataLoader.load_and_process_distributed.remote(
            dataset_path, target_column, preprocessing_config, **dataset_kwargs
        )
        
        # Start training tasks for all models
        training_refs = {}
        for model_config in models_config:
            model_name = model_config['name']
            training_refs[model_name] = SklearnModel.train_distributed.remote(
                SklearnModel,
                model_config['estimator'],
                model_config['params'],
                X_ref,
                y_ref
            )
        
        # Wait for all training tasks to complete
        trained_models = {}
        for model_name, model_ref in training_refs.items():
            trained_models[model_name] = ray.get(model_ref)
            
        return trained_models