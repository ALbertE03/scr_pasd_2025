import os
import pickle
import json
import datetime
from typing import Dict, List, Any, Optional
import ray

class ModelRegistry:
    """Manages the storage, retrieval, and metadata of trained models."""
    
    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store models and metadata
        """
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.metadata_file = os.path.join(registry_dir, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        """Load the metadata file or create a new one if it doesn't exist."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": {}}
            self._save_metadata()
    
    def _save_metadata(self):
        """Save the metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model_name: str, model: Any, metrics: Dict[str, float], 
                   model_type: str, dataset_info: Optional[Dict[str, Any]] = None,
                   performance_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model to the registry.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Evaluation metrics
            model_type: Type/class of the model
            dataset_info: Information about the dataset used for training
            performance_metrics: Performance metrics about training and evaluation times
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        model_path = os.path.join(self.registry_dir, f"{model_id}.pkl")
        
        # Save model to disk
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Update metadata
        self.metadata["models"][model_id] = {
            "name": model_name,
            "path": model_path,
            "type": model_type,
            "created_at": timestamp,
            "metrics": metrics,
            "dataset_info": dataset_info or {},
            "performance_metrics": performance_metrics or {},
            "status": "active"
        }
        self._save_metadata()
        
        return model_id
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            The loaded model
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self.metadata["models"][model_id]["path"]
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_metadata(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a specific model or all models.
        
        Args:
            model_id: ID of the model to get metadata for, or None for all models
            
        Returns:
            Model metadata
        """
        if model_id is None:
            return self.metadata
        
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        return self.metadata["models"][model_id]
    
    def list_models(self, filter_by: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List models in the registry, optionally filtered by metadata.
        
        Args:
            filter_by: Dictionary of metadata key-value pairs to filter by
            
        Returns:
            List of model IDs matching the filter
        """
        if filter_by is None:
            return list(self.metadata["models"].keys())
        
        filtered_models = []
        for model_id, metadata in self.metadata["models"].items():
            match = True
            for key, value in filter_by.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_models.append(model_id)
        
        return filtered_models
    
    def delete_model(self, model_id: str):
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self.metadata["models"][model_id]["path"]
        if os.path.exists(model_path):
            os.remove(model_path)
        
        del self.metadata["models"][model_id]
        self._save_metadata()
        
    def update_metadata(self, model_id: str, updated_metadata: Dict[str, Any]):
        """
        Update metadata for a model in the registry.
        
        Args:
            model_id: ID of the model to update
            updated_metadata: Dictionary of metadata to update or add
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Update the metadata with the new values
        self.metadata["models"][model_id].update(updated_metadata)
        self._save_metadata()