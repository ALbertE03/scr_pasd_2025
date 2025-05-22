import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import json
import logging
from models.model_registry import ModelRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model registry
model_registry = ModelRegistry()

# Create FastAPI app
app = FastAPI(title="Distributed ML Model Server",
             description="API for serving machine learning models trained with the distributed platform",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class PredictionRequest(BaseModel):
    model_id: str
    features: Union[List[List[float]], List[Dict[str, Any]]]
    
class PredictionResponse(BaseModel):
    predictions: List[Any]
    model_id: str
    
class ModelsResponse(BaseModel):
    models: Dict[str, Any]
    
class ModelDetailsResponse(BaseModel):
    model_id: str
    metadata: Dict[str, Any]

# In-memory cache for loaded models
loaded_models = {}

def get_model(model_id: str):
    """Get a model from cache or load it from registry."""
    if model_id not in loaded_models:
        try:
            logger.info(f"Loading model {model_id} from registry")
            loaded_models[model_id] = model_registry.load_model(model_id)
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found or failed to load")
    
    return loaded_models[model_id]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a model from the registry."""
    try:
        model = get_model(request.model_id)
        
        # Convert the features to numpy array
        if isinstance(request.features[0], dict):
            # Handle dict format (with column names)
            import pandas as pd
            features = pd.DataFrame(request.features)
            X = features.to_numpy()
        else:
            # Handle list format (without column names)
            X = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        return {"predictions": predictions, "model_id": request.model_id}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all models in the registry."""
    try:
        metadata = model_registry.get_metadata()
        return {"models": metadata["models"]}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/{model_id}", response_model=ModelDetailsResponse)
async def get_model_details(model_id: str):
    """Get details about a specific model."""
    try:
        metadata = model_registry.get_metadata(model_id)
        return {"model_id": model_id, "metadata": metadata}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Model server starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Model server shutting down")

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    uvicorn.run("server.model_server:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()