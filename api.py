import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import psutil
import ray
from modules.training import Trainer
import subprocess
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Distributed ML API",
    description="API para interactuar con modelos de Machine Learning entrenados en cluster Ray",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


trainer = None
models_cache = {}
models_directory = "models"
inference_stats_file = "inference_stats.json"


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Retorna un favicon """
    return JSONResponse(content={"message": "No favicon available"}, status_code=204)


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Nombre del modelo a usar")
    features: List[List[float]] = Field(..., description="Características para predicción")
    return_probabilities: bool = Field(False, description="Retornar probabilidades además de predicciones")

class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    training_time: float
    timestamp: str
    status: str

class ClusterInfo(BaseModel):
    connected: bool
    resources: dict
    nodes: list
    alive_nodes: list
    dead_nodes: list
    total_cpus: int
    total_memory: float
    total_gpus: int
    node_count: int
    alive_node_count: int
    dead_node_count: int
    resource_usage: dict
    timestamp: float


class PredictionResponse(BaseModel):
    model_name: str
    predictions: List[Union[int, float]]
    probabilities: Optional[List[List[float]]] = None
    feature_count: int
    prediction_time: float
    model_path: Optional[str] = None


def save_inference_stats(model_name: str, prediction_time: float, n_samples: int, accuracy: float = None):
    """Guarda estadísticas de inferencia en tiempo real"""
    try:
        if os.path.exists(inference_stats_file):
            with open(inference_stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}

        if model_name not in stats:
            stats[model_name] = []

        inference_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction_time": prediction_time,
            "n_samples": n_samples,
            "avg_time_per_sample": prediction_time / n_samples if n_samples > 0 else prediction_time,
            "accuracy": accuracy
        }
        
        stats[model_name].append(inference_entry)

        with open(inference_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Estadísticas de inferencia guardadas para modelo {model_name}")
        
    except Exception as e:
        logger.error(f"Error guardando estadísticas de inferencia: {e}")


def get_inference_stats(model_name: str = None) -> Dict:
    """Obtiene estadísticas de inferencia guardadas"""
    try:
        if not os.path.exists(inference_stats_file):
            return {}
        
        with open(inference_stats_file, 'r') as f:
            stats = json.load(f)
        
        if model_name:
            return {model_name: stats.get(model_name, [])}
        else:
            return stats
            
    except Exception as e:
        logger.error(f"Error cargando estadísticas de inferencia: {e}")
        return {}



def load_model_from_file(model_name: str, dataset_name: str = None, models_dir: str = "models"):
    """Carga un modelo desde archivo con búsqueda mejorada"""
    path = 'train_results'
    os.makedirs(path, exist_ok=True)
 
    if not os.listdir(path):
        logger.info("No se encontraron datasets en el directorio train_results")
        return "",""
    
    # Extract the base model name if it contains dataset info
    base_model_name = model_name.split("_")[0] if "_" in model_name else model_name
    
    for dataset in os.listdir(path):
        path1 = os.path.join(path, dataset)
     
        if not os.path.isdir(path1):
            continue
        
        # Look in the models subdirectory within each dataset
        models_path = os.path.join(path1, 'models')
        if not os.path.exists(models_path) or not os.path.isdir(models_path):
            continue
            
        for model_file_name in os.listdir(models_path):
            if not model_file_name.endswith(".pkl"):
                continue
            
            # Check if the base model name matches the beginning of the file name
            file_base_name = model_file_name.split("_")[0]
            if file_base_name != base_model_name:
                continue
                
            model_path = os.path.join(models_path, model_file_name)        
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Modelo {model_name} cargado desde {model_path}")
                return model, model_path
            except Exception as e:
                raise Exception(f"Error cargando modelo {model_name} desde {model_path}: {str(e)}")
    raise FileNotFoundError(f"Modelo {model_name} no encontrado en train_results")



def get_available_models() -> Dict[str, Dict]:
    """Obtiene información de todos los modelos disponibles con búsqueda exhaustiva"""
    models_info = {}
    path = 'train_results'

    os.makedirs(path, exist_ok=True)
 
    if not os.listdir(path):
        logger.info("No se encontraron datasets en el directorio train_result")
        return models_info
    
    for dataset in os.listdir(path):
        path1 = os.path.join(path, dataset,'models')
     
        if not os.path.isdir(path1):
            continue
            
        for model_file_name in os.listdir(path1):
            if not model_file_name.endswith(".pkl"):
                continue
            
            model_path = os.path.join(path1, model_file_name)
            model_file = Path(model_path)
            logger.info(f"Procesando modelo: {model_file_name} en {path1}")
            try:
                
                
                unique_name = f"{model_file_name.split('_')[0]}_{dataset}"
                model_info = {
                    "model_name": unique_name,
                    "dataset": dataset,
                    "file_path": str(model_path),
                    "directory": str(path1),
                    "file_size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                    "modified_time": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                    "scores": {}  # Initialize with empty scores dict
                }
                
                for result_file_name in os.listdir(os.path.join(path,dataset)):
                    if not result_file_name.endswith('.json'):
                        continue

                    result_file_path = os.path.join(os.path.join(path,dataset), result_file_name)
                    try:
                        with open(result_file_path, 'r') as f:
                            results = json.load(f)
                            logger.info(f"Loaded results from {result_file_path}")
                            
                            # Extract base model name from file (e.g., "RandomForest" from "RandomForest.pkl")
                            model_key = model_file_name.replace('.pkl', '').split('_')[0]
                            
                            if model_key in results:
                                result = results[model_key]
                                logger.info(f"Found result for model {model_key} with scores: {result.get('scores', {})}")
                                
                                # Update model_info with result data, but preserve the unique_name and dataset
                                original_model_name = model_info['model_name']
                                original_dataset = model_info['dataset']
                                
                                model_info.update(result)
                                
                                # Restore the unique identifiers
                                model_info['model_name'] = original_model_name
                                model_info['dataset'] = original_dataset
                                
                                # Ensure scores structure exists and is valid
                                if 'scores' in result and isinstance(result['scores'], dict):
                                    model_info['scores'] = result['scores']
                                    logger.info(f"Successfully loaded scores for {model_key}: {model_info['scores']}")
                                else:
                                    logger.warning(f"Model {model_key} missing or invalid 'scores' in result data")
                                    model_info['scores'] = {}
                                
                                break  # Found the result, no need to check other files
                                
                    except Exception as e:
                        logger.warning(f"Error cargando métricas desde {result_file_path}: {e}")
                
                models_info[unique_name] = model_info
            
            except Exception as e:
                logger.warning(f"Error cargando modelo {model_path}: {e}")
                continue
    
    logger.info(f"Encontrados {len(models_info)} modelos en total")
    return models_info


@app.get("/", summary="Información de la API")
async def root():
    """Endpoint raíz con información básica de la API"""
    return {
        "message": "API de Modelos de Machine Learning Distribuidos",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "models": "/models",
            "predict": "/predict",
            "train": "/train",
            "cluster": "/cluster/status"
        }
    }


@app.get("/health", summary="Estado de salud de la API")
async def health_check():
    """Verifica el estado de salud de la API y sus dependencias"""
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ray_initialized": ray.is_initialized(),
        "models_available": len(get_available_models())
    }
    
    try:
        cluster_info = ray.cluster_resources() if ray.is_initialized() else {}
        health_status["ray_cluster"] = "connected" if cluster_info else "disconnected"
        health_status["cluster_resources"] = cluster_info
    except Exception as e:
        health_status["ray_cluster"] = f"error: {str(e)}"
    
    return health_status




@app.get("/models/{model_name}", summary="Información de un modelo específico")
async def get_model_info(model_name: str):
    """Obtiene información detallada de un modelo específico"""
    try:
        models = get_available_models()
        
        if model_name not in models:
            matching_models = [k for k in models.keys() if model_name in k or k.startswith(f"{model_name}")]
            if not matching_models:
                available_models = list(models.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Modelo {model_name} no encontrado. Modelos disponibles: {available_models[:10]}{'...' if len(available_models) > 10 else ''}"
                )
            model_name = matching_models[0]
        
        model_info = models[model_name]

        try:
            model, model_path = load_model_from_file(
                model_info["model_name"], 
                model_info.get("dataset",'')
            )
            model_info["model_type"] = type(model).__name__
            model_info["model_parameters"] = getattr(model, 'get_params', lambda: {})()
            model_info["loaded_from"] = model_path
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo para información adicional: {e}")
            model_info["load_error"] = str(e)
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, summary="Realizar predicciones")
async def predict(request: PredictionRequest):
    """Realiza predicciones usando un modelo entrenado"""
    
    start_time = time.time()
    logger.error('enviando {} features para predicción'.format(request.features))
    logger.error('modelo: {}'.format(request.model_name))
    try:
        models = get_available_models()

        model_key = None
        if request.model_name in models:
            model_key = request.model_name
        
        if not model_key:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Modelo {request.model_name} no encontrado. Modelos disponibles: {available_models[:10]}{'...' if len(available_models) > 10 else ''}"
            )

        model_info = models[model_key]
        model, model_path = load_model_from_file(
            model_info["model_name"], 
            model_info.get("dataset")
        )

        X = np.array(request.features)
        logger.info(f"Realizando predicción con {X.shape[0]} muestras y {X.shape[1]} características")
        logger.info(X)
        columns = model_info.get('columns', [])
        target = model_info.get('target', '')
        predictions = None
        prediction_error = None
        used_dataframe = False  # Track which format worked for prediction
        
        feature_columns = []
        if target and target in columns:
            feature_columns = [col for col in columns if col != target]
        else:
            feature_columns = columns[:-1] if columns else []
        
        # First try with numpy array
        try:
            predictions = model.predict(X)
            logger.info(f'Predicción exitosa con numpy array: {predictions}')
            used_dataframe = False
        except Exception as e:
            prediction_error = str(e)
            logger.warning(f"Predicción con numpy array falló: {e}")
        
            # Try with DataFrame if numpy failed
            try:
                if feature_columns and len(feature_columns) == X.shape[1]:
                    X_df = pd.DataFrame(X, columns=feature_columns)
                    predictions = model.predict(X_df)
                    logger.info(f'Predicción exitosa con DataFrame: {predictions}')
                    used_dataframe = True
                else:
                    logger.warning(f"No se pudieron obtener nombres de columnas válidos. Columnas disponibles: {feature_columns}, características esperadas: {X.shape[1]}")
                    raise Exception(f"Modelo requiere DataFrame con nombres de columnas pero no se pudieron determinar. Error original: {prediction_error}")
                    
            except Exception as df_error:
                logger.error(f"Predicción con DataFrame también falló: {df_error}")
                raise Exception(f"Predicción falló con numpy array ({prediction_error}) y DataFrame ({str(df_error)})")
        
        if predictions is None:
            raise Exception("No se pudieron realizar predicciones")
        prediction_time = time.time() - start_time
        
        response_data = {
            "model_name": model_key,
            "predictions": predictions.tolist(),
            "feature_count": X.shape[1],
            "prediction_time": prediction_time,
            "model_path": model_path
        }
        
        model_accuracy = model_info.get('scores', {}).get("Accuracy")
        save_inference_stats(model_key, prediction_time, X.shape[0], model_accuracy)

        if request.return_probabilities:
            logger.error(f"Calculando probabilidades para modelo: {model_key}")
            logger.error(f"Tipo de modelo: {type(model).__name__}")
            logger.error(f"Tiene predict_proba: {hasattr(model, 'predict_proba')}")
            
            try:

                if not hasattr(model, 'predict_proba'):
                    logger.error(f"Modelo {model_key} ({type(model).__name__}) no tiene método predict_proba")
                    response_data["probabilities"] = None
                else:
                    if used_dataframe:
                        X_df = pd.DataFrame(X, columns=feature_columns)
                        logger.info(f"Usando DataFrame para probabilidades: {X_df.columns.tolist()}")
                        probabilities = model.predict_proba(X_df)
                        logger.error(probabilities)
                        logger.error(f"Probabilidades calculadas con DataFrame para {X.shape[0]} muestras")
                    else:
                        logger.info(f"Usando numpy array para probabilidades: shape {X.shape}")
                        probabilities = model.predict_proba(X)
                        logger.info(f"Probabilidades calculadas con numpy array para {X.shape[0]} muestras")
                    
                    logger.info(f"Probabilidades shape: {probabilities.shape}")
                    logger.info(f"Probabilidades sample: {probabilities[0] if len(probabilities) > 0 else 'empty'}")
                    response_data["probabilities"] = probabilities.tolist()
                    logger.info(f"Probabilidades convertidas a lista exitosamente")
                    
            except AttributeError as attr_error:
                logger.warning(f"Modelo {model_key} no soporta predict_proba: {attr_error}")
                response_data["probabilities"] = None
            except Exception as prob_error:
                logger.error(f"Error calculando probabilidades: {prob_error}")
                logger.error(f"Tipo de excepción: {type(prob_error).__name__}")
                response_data["probabilities"] = None
        
        return PredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/cluster/status", response_model=ClusterInfo, summary="Estado del cluster Ray")
async def cluster_status():
    """Obtiene información extendida del estado del cluster Ray"""
    import time
    try:
        if not ray.is_initialized():
            head_address = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
            ray.init(address=f"ray://{head_address}:10001", ignore_reinit_error=True)
        
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node.get('Alive', False)]
        dead_nodes = [node for node in nodes if not node.get('Alive', False)]

        resource_usage = ray.available_resources() if hasattr(ray, 'available_resources') else {}
        return ClusterInfo(
            connected=True,
            resources=cluster_resources,
            nodes=nodes,
            alive_nodes=alive_nodes,
            dead_nodes=dead_nodes,
            total_cpus=int(cluster_resources.get('CPU', 0)),
            total_memory=cluster_resources.get('memory', 0),
            total_gpus=int(cluster_resources.get('GPU', 0)),
            node_count=len(nodes),
            alive_node_count=len(alive_nodes),
            dead_node_count=len(dead_nodes),
            resource_usage=resource_usage,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Error obteniendo estado del cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del cluster: {str(e)}")
    

@app.delete("/models/{model_name}", summary="Eliminar modelo", response_model=None)
async def delete_model(model_name: str):
    """Elimina un modelo entrenado del almacenamiento"""
    try:
        logger.info(f"Solicitud para eliminar modelo: {model_name}")
        models = get_available_models()
        logger.info(f"Total de modelos disponibles: {len(models)}")

        if model_name in models:
            model_key = model_name
            logger.info(f"Modelo encontrado exactamente: {model_key}")
        else:
            matching_models = []

            requested_parts = model_name.lower().split('_')
            
            for model_key in models.keys():
                model_parts = model_key.lower().split('_')

                if all(part in model_parts for part in requested_parts):
                    matching_models.append(model_key)
            
            if not matching_models:
                logger.warning(f"No se encontraron coincidencias para modelo: {model_name}")
                available_models = list(models.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Modelo {model_name} no encontrado. Modelos disponibles: {available_models[:10]}{'...' if len(available_models) > 10 else ''}"
                )
                
            if len(matching_models) > 1:
                logger.warning(f"Múltiples modelos coincidentes: {matching_models}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Múltiples modelos coincidentes ({len(matching_models)}). Por favor especifique mejor: {matching_models}"
                )
                
            model_key = matching_models[0]
            logger.info(f"Modelo encontrado por coincidencia parcial: {model_key}")
        
        model_info = models[model_key]
        model_path = model_info["file_path"]
        logger.info(f"Intentando eliminar archivo: {model_path}")
        
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Modelo {model_key} eliminado exitosamente de {model_path}")
                
                global models_cache
                models_cache = {} 
                logger.info(f"Cache de modelos limpiada completamente")
                

                if os.path.exists(model_path):
                    logger.error(f"ERROR: El archivo {model_path} todavía existe después de intentar eliminarlo")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"El archivo no pudo ser eliminado completamente: {model_path}"
                    )
                
                return {
                    "success": True,
                    "message": f"Modelo {model_key} eliminado exitosamente",
                    "deleted_file": model_path,
                    "original_request": model_name,
                    "timestamp": datetime.now().isoformat()
                }
            except PermissionError:
                logger.error(f"Error de permisos al eliminar archivo: {model_path}")
                raise HTTPException(
                    status_code=403, 
                    detail=f"Sin permisos para eliminar el archivo: {model_path}"
                )
            except Exception as file_error:
                logger.error(f"Error al eliminar archivo {model_path}: {file_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error al eliminar archivo: {str(file_error)}"
                )
        else:
            logger.warning(f"Archivo del modelo no encontrado: {model_path}")
            raise HTTPException(
                status_code=404, 
                detail=f"Archivo del modelo no encontrado: {model_path}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando modelo: {str(e)}")

@app.post("/predict/batch", summary="Predicciones en lote desde archivo")
async def predict_batch(
    model_name: str,
    file: UploadFile = File(..., description="Archivo CSV con características para predicción"),
    return_probabilities: bool = False
):
    """Realiza predicciones en lote desde un archivo CSV"""
    import time
    start_time = time.time()
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))

        models = get_available_models()
        
        model_key = None
        if model_name in models:
            model_key = model_name
        else:
            matching_models = [k for k in models.keys() if 
                             model_name in k or 
                             k.endswith(f"_{model_name}") or
                             k.startswith(f"{model_name}_")]
            if matching_models:
                model_key = matching_models[0]
                logger.info(f"Usando modelo {model_key} para predicción en lote de {model_name}")
        
        if not model_key:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Modelo {model_name} no encontrado. Modelos disponibles: {available_models[:10]}{'...' if len(available_models) > 10 else ''}"
            )
        
        model_info = models[model_key]
        model, model_path = load_model_from_file(
            model_info["model_name"], 
            model_info.get("dataset")
        )

        predictions = model.predict(df.values)
        
        response_data = {
            "model_name": model_key,
            "predictions": predictions.tolist(),
            "n_samples": len(df),
            "feature_count": df.shape[1],
            "prediction_time": time.time() - start_time,
            "filename": file.filename,
            "model_path": model_path
        }

        if return_probabilities:
            try:
                probabilities = model.predict_proba(df.values)
                response_data["probabilities"] = probabilities.tolist()
            except AttributeError:
                response_data["probabilities"] = None
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


@app.get("/models/dataset/{dataset_name}", summary="Modelos por dataset")
async def get_models_by_dataset(dataset_name: str):
    """Obtiene todos los modelos disponibles para un dataset específico"""
    try:
        all_models = get_available_models()
        dataset_models = {
            name: info for name, info in all_models.items() 
            if info.get("dataset") == dataset_name
        }
        
        if not dataset_models:
            available_datasets = list(set(info.get("dataset", "unknown") for info in all_models.values()))
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron modelos para dataset '{dataset_name}'. Datasets disponibles: {available_datasets}"
            )
        
        return {
            "dataset": dataset_name,
            "total_models": len(dataset_models),
            "models": dataset_models,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo modelos para dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo modelos: {str(e)}")


@app.get("/models/search/{query}", summary="Buscar modelos")
async def search_models(query: str):
    """Busca modelos por nombre, dataset o tipo"""
    try:
        all_models = get_available_models()
        query_lower = query.lower()
        
        matching_models = {}
        for name, info in all_models.items():
            if (query_lower in name.lower() or 
                query_lower in info.get("dataset", "").lower() or
                query_lower in info.get("model_type", "").lower() or
                query_lower in info.get("model_name", "").lower()):
                matching_models[name] = info
        
        return {
            "query": query,
            "total_matches": len(matching_models),
            "models": matching_models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:        
        logger.error(f"Error buscando modelos con query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@app.get("/inference-stats", summary="Estadísticas de inferencia")
async def get_inference_statistics(model_name: Optional[str] = None):
    """Obtiene estadísticas de inferencia en tiempo real"""
    try:
        stats = get_inference_stats(model_name)
        
        if not stats:
            return {
                "message": "No hay estadísticas de inferencia disponibles",
                "stats": {},
                "timestamp": datetime.now().isoformat()
            }

        aggregated_stats = {}
        
        for model, entries in stats.items():
            if not entries:
                continue

            prediction_times = [entry["prediction_time"] for entry in entries]
            avg_times_per_sample = [entry["avg_time_per_sample"] for entry in entries]
            total_samples = sum(entry["n_samples"] for entry in entries)
            
            aggregated_stats[model] = {
                "total_predictions": len(entries),
                "total_samples": total_samples,
                "avg_prediction_time": sum(prediction_times) / len(prediction_times),
                "min_prediction_time": min(prediction_times),
                "max_prediction_time": max(prediction_times),
                "avg_time_per_sample": sum(avg_times_per_sample) / len(avg_times_per_sample),
                "last_prediction": entries[-1]["timestamp"] if entries else None,
                "accuracy": entries[-1]["accuracy"] if entries and entries[-1]["accuracy"] else None,
                "recent_entries": entries[-50:] if len(entries) > 50 else entries  # Últimas 50 entradas
            }
        
        return {
            "stats": stats,
            "aggregated": aggregated_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de inferencia: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@app.post("/add/node", summary="Agregar nodo al cluster Ray")
async def add_node_to_cluster(worker_name: str = Query(..., description="Nombre del worker"), add_cpu: int = Query(..., description="CPUs para el worker")):
    """Agrega un nodo al cluster Ray (worker externo)"""
    try:
            
        command = f"docker run -d --name ray_worker_{worker_name.lower().replace(" ","_")} --hostname ray_worker_{worker_name} --network scr_pasd_2025_ray-network -e RAY_HEAD_SERVICE_HOST=ray-head -e NODE_ROLE=worker -e LEADER_NODE=false -e FAILOVER_PRIORITY=3 -e ENABLE_AUTO_FAILOVER=false --shm-size=2gb scr_pasd_2025-ray-head bash -c 'echo Worker externo iniciando... && echo Esperando al cluster principal... && sleep 10 && echo Conectando al cluster existente... && ray start --address=ray-head:6379 --num-cpus={add_cpu} && echo Worker externo conectado exitosamente! && tail -f /dev/null'"
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Worker externo 'ray_worker_{worker_name}' añadido exitosamente al cluster", "stdout": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        logger.error(f"Excepción al añadir worker externo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excepción al añadir worker externo: {str(e)}")

@app.delete("/remove/node", summary="Eliminar nodo del cluster Ray")
async def remove_node_from_cluster(node_name: str = Query(..., description="Nombre del worker")):
    try:
        if node_name.startswith("ray-head") or node_name == "ray-head":
            return {"success": False, "error": "No se puede eliminar el nodo principal (ray-head), ya que es necesario para el funcionamiento del cluster"}
        command = f"docker stop {node_name} && docker rm {node_name}"
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Nodo '{node_name}' eliminado exitosamente del cluster", "stdout": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        logger.error(f"Excepción al eliminar nodo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excepción al eliminar nodo: {str(e)}")
    
@app.get("/cluster/nodes", summary="Lista de nodos del cluster Ray")
async def get_all_nodes_ray(command: str = Query(..., description="Comando shell para listar nodos Ray")):
    """Obtiene la lista de nodos del cluster Ray ejecutando un comando shell"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            return {"success": True, "data": result.stdout.splitlines()}
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        logger.error(f"Excepción al obtener nodos del cluster: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excepción al obtener nodos del cluster: {str(e)}")

@app.get("/system/status", summary="Estado del sistema (host)")
async def system_status():
    """Devuelve información de uso de CPU, memoria y disco del sistema host"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available / (1024**3),  # GB
            "memory_total": memory.total / (1024**3),  # GB
            "disk_percent": disk.percent,
            "disk_free": disk.free / (1024**3),  # GB
            "disk_total": disk.total / (1024**3)  # GB
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del sistema: {str(e)}")

@app.post("/read/csv", summary="Leer archivo CSV")
async def read_csv(file: UploadFile = File(..., description="Archivo CSV a leer")):
    """Lee un archivo CSV y devuelve su contenido como JSON"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
        
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        df = df.replace({np.nan: None})
        return {"data": df.to_dict(orient='records')}
    except Exception as e:
        logger.error(f"Error leyendo archivo CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error leyendo archivo CSV: {str(e)}")

class TrainingRequest(BaseModel):
    dataset: List[Dict[str, Any]] = Field(..., description="Dataset en formato JSON (orient='records')")
    target_column: str = Field(..., description="Nombre de la columna objetivo")
    problem_type: str = Field(..., description="Tipo de problema: 'Clasificación' o 'Regresión'")
    metrics: List[str] = Field(..., description="Lista de métricas a calcular")
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = 42
    features_to_exclude: List[str] = []
    transform_target: bool = False
    selected_models: List[str] = Field(..., description="Lista de nombres de modelos a entrenar")
    estrategia:List[str]
    dataset_name:str

import json
import numpy as np
from fastapi.responses import JSONResponse

class NumpyJSONResponse(JSONResponse):
    """
    Una respuesta JSON personalizada para manejar tipos de datos de NumPy
    que no son compatibles con JSON estándar.
    """
    def render(self, content: any) -> bytes:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    # Si es NaN o Inf, conviértelo a None (null en JSON)
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        return json.dumps(content, cls=NumpyEncoder).encode("utf-8")
@app.post('/train/oneDataset', summary='Entrena varios modelos en el mismo dataset')
async def train(params: TrainingRequest):
    """
    Recibe los parámetros de entrenamiento, lanza el trabajo en segundo plano y devuelve un ID de experimento.
    """
    try:
        logger.info('inicio')

        df = pd.DataFrame(params.dataset)
        logger.info(f"Datos recibidos: {df.shape[0]} filas, {df.shape[1]} columnas")
        actor_params = {
            "df": df,
            "target_column": params.target_column,
            "problem_type": params.problem_type,
            "metrics": params.metrics,
             "test_size": params.test_size,
            "random_state": params.random_state,
            "features_to_exclude": params.features_to_exclude,
            "transform_target": params.transform_target,
            "selected_models": params.selected_models,
            'estrategia':params.estrategia,
            'dataset_name':params.dataset_name
        }
        trainer_actor = Trainer(**actor_params)
        result = trainer_actor.train()
        logger.info(f"Entrenamiento iniciado con ID: {result}")
        response_data= {
            "message": "Entrenamiento iniciado con éxito.",
            "data": result,
        }
        return NumpyJSONResponse(content=response_data)
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=500, detail=f"Error al iniciar el entrenamiento: {str(e)}")


@app.get("/models", summary="Listar modelos disponibles")
async def list_models():
    """Lista todos los modelos entrenados disponibles con sus métricas"""
    try:
        models = get_available_models()
        return {
            "total_models": len(models),
            "models": models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")
  

def main():
    """Función principal para ejecutar la API FastAPI"""
    logger.info("Iniciando API de Modelos de Machine Learning Distribuidos")
    
    os.makedirs("models", exist_ok=True)
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info"
    }
    
    logger.info(f"Servidor iniciando en http://{config['host']}:{config['port']}")
    logger.info(f"Documentación disponible en http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run("api:app", **config)


if __name__ == "__main__":
    main()
