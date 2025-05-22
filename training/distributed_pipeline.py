import ray
import numpy as np
import pandas as pd
import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.model_selection import train_test_split
from models.sklearn_model import SklearnModel
from models.model_registry import ModelRegistry
from evaluation.model_evaluator import ModelEvaluator
from utils.data_loader import DataLoader

# Try to import monitoring utilities
try:
    from utils.monitoring import get_system_resources, get_ray_status
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedTrainingPipeline:
    """Orchestrates the distributed training, evaluation, and registration of models."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None, init_ray: bool = True):
        """
        Initialize the training pipeline.
        
        Args:
            registry: Model registry instance
            init_ray: Whether to initialize Ray
        """
        self.registry = registry or ModelRegistry()
        if init_ray and not ray.is_initialized():
            ray.init()
            
        # Métricas de rendimiento
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "total_time": None,
            "data_loading_time": None,
            "training_times": {},
            "evaluation_times": {},
            "system_metrics": []
        }
        
        # Directorio para guardar métricas
        self.metrics_dir = "training_metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def _capture_system_metrics(self):
        """Capture system metrics if monitoring is available."""
        if not MONITORING_AVAILABLE:
            return None
            
        metrics = {
            "timestamp": time.time(),
            "system": get_system_resources(),
            "ray": get_ray_status()
        }
        self.performance_metrics["system_metrics"].append(metrics)
        return metrics
        
    def _save_metrics(self, experiment_name: str = None):
        """Save performance metrics to disk."""
        if not self.performance_metrics["start_time"]:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = experiment_name or f"training_run_{timestamp}"
        metrics_path = os.path.join(self.metrics_dir, f"{experiment_name}.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
            
        logger.info(f"Performance metrics saved to {metrics_path}")
        return metrics_path
    
    def run(self, dataset_path: str, target_column: str, 
            models_config: List[Dict[str, Any]],
            test_size: float = 0.2, random_state: int = 42,
            preprocessing_config: Optional[Dict[str, Any]] = None,
            evaluation_metrics: Optional[List[str]] = None,
            task_type: str = "classification",
            collect_metrics: bool = True,
            experiment_name: Optional[str] = None,
            **dataset_kwargs) -> Dict[str, str]:
        """
        Run a complete distributed training pipeline.
        
        Args:
            dataset_path: Path to the dataset
            target_column: Name of the target column
            models_config: List of model configurations
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            preprocessing_config: Configuration for data preprocessing
            evaluation_metrics: Metrics to compute during evaluation
            task_type: Type of ML task ("classification" or "regression")
            collect_metrics: Whether to collect performance metrics
            experiment_name: Name for the experiment (used for metrics files)
            **dataset_kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary mapping model names to their registry IDs
        """
        # Start performance metrics collection
        self.performance_metrics["start_time"] = time.time()
        if collect_metrics:
            self._capture_system_metrics()
        
        # Load data
        logger.info("Loading and preprocessing data...")
        data_load_start = time.time()
        # When calling a remote function, it returns one ObjectRef, not multiple
        data_ref = DataLoader.load_and_process_distributed.remote(
            dataset_path, target_column, preprocessing_config, **dataset_kwargs
        )
        
        # Get data from Ray object store - the remote function returns a tuple (X, y)
        X, y = ray.get(data_ref)
        data_load_end = time.time()
        self.performance_metrics["data_loading_time"] = data_load_end - data_load_start
        
        if collect_metrics:
            self._capture_system_metrics()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create references to data splits in Ray object store
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)
        X_test_ref = ray.put(X_test)
        y_test_ref = ray.put(y_test)
        
        # Train models in parallel
        logger.info(f"Training {len(models_config)} models in parallel...")
        training_start_times = {}
        training_refs = {}
        for model_config in models_config:
            model_name = model_config['name']
            logger.info(f"  Starting training for {model_name}...")
            training_start_times[model_name] = time.time()
            training_refs[model_name] = SklearnModel.train_distributed.remote(
                SklearnModel,
                model_config['estimator'],
                model_config.get('params', {}),
                X_train_ref,
                y_train_ref
            )
            
            if collect_metrics:
                self._capture_system_metrics()
        
        # Evaluate models in parallel
        logger.info("Training complete. Evaluating models...")
        model_refs = {}
        eval_refs = {}
        eval_start_times = {}
        registered_models = {}
        
        for model_name, model_ref in training_refs.items():
            # Get the trained model
            model = ray.get(model_ref)
            model_refs[model_name] = model
            
            # Record training time
            training_end_time = time.time()
            self.performance_metrics["training_times"][model_name] = training_end_time - training_start_times[model_name]
            
            # Start evaluation
            logger.info(f"  Evaluating {model_name}...")
            eval_start_times[model_name] = time.time()
            eval_refs[model_name] = ModelEvaluator.evaluate_distributed.remote(
                model, X_test_ref, y_test_ref, evaluation_metrics, task_type
            )
            
            if collect_metrics:
                self._capture_system_metrics()
        
        # Register models with their evaluation metrics
        for model_name, model in model_refs.items():
            metrics = ray.get(eval_refs[model_name])
            logger.info(f"  Registering {model_name} with metrics: {metrics}")
            
            # Record evaluation time
            eval_end_time = time.time()
            self.performance_metrics["evaluation_times"][model_name] = eval_end_time - eval_start_times[model_name]
            
            # Create dataset info
            dataset_info = {
                "source_path": dataset_path,
                "target_column": target_column,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "random_state": random_state
            }
            
            # Add performance metrics to model metadata
            model_perf_metrics = {
                "training_time": self.performance_metrics["training_times"].get(model_name),
                "evaluation_time": self.performance_metrics["evaluation_times"].get(model_name)
            }
            
            # Register the model
            model_id = self.registry.save_model(
                model_name=model_name,
                model=model,
                metrics=metrics,
                model_type=str(type(model.model).__name__),
                dataset_info=dataset_info,
                performance_metrics=model_perf_metrics
            )
            
            registered_models[model_name] = model_id
        
        # Finalize performance metrics
        self.performance_metrics["end_time"] = time.time()
        self.performance_metrics["total_time"] = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
        
        if collect_metrics:
            self._capture_system_metrics()
            self._save_metrics(experiment_name)
        
        logger.info(f"Pipeline completed successfully in {self.performance_metrics['total_time']:.2f} seconds!")
        return registered_models