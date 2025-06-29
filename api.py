import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import psutil

from train import DistributedMLTrainer
from modules.cluster import get_cluster_status
import ray


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
    """Retorna un favicon vacío para evitar errores 404 en navegadores"""
    return JSONResponse(content={"message": "No favicon available"}, status_code=204)


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Nombre del modelo a usar")
    features: List[List[float]] = Field(..., description="Características para predicción")
    return_probabilities: bool = Field(False, description="Retornar probabilidades además de predicciones")


class TrainingRequest(BaseModel):
    dataset_name: str = Field("iris", description="Nombre del dataset (iris, wine, breast_cancer, digits)")
    selected_models: Optional[List[str]] = Field(None, description="Lista de modelos a entrenar")
    test_size: float = Field(0.3, description="Proporción de datos para prueba", ge=0.1, le=0.5)


class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    cv_mean: float
    cv_std: float
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


def get_trainer():
    """Obtiene o inicializa el trainer distribuido"""
    global trainer
    if trainer is None:
        try:
            trainer = DistributedMLTrainer(enable_fault_tolerance=True)
            logger.info("Trainer distribuido inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando trainer: {e}")
            raise HTTPException(status_code=500, detail=f"Error inicializando trainer: {str(e)}")
    return trainer


def load_model_from_file(model_name: str, dataset_name: str = None, models_dir: str = "models"):
    """Carga un modelo desde archivo con búsqueda mejorada"""
    model_path = None

    search_paths = []

    if dataset_name:
        search_paths.extend([
            os.path.join("models", dataset_name, f"{model_name}.pkl"),
            os.path.join(f"models_{dataset_name}", f"{model_name}.pkl"),
            os.path.join("training_results", f"models_{dataset_name}", f"{model_name}.pkl")
        ])
    
    search_paths.extend([
        os.path.join("models", f"{model_name}.pkl"),
        os.path.join("models", "iris", f"{model_name}.pkl"),
        os.path.join("models", "wine", f"{model_name}.pkl"),
        os.path.join("models", "breast_cancer", f"{model_name}.pkl"),
        os.path.join("models", "digits", f"{model_name}.pkl"),
        os.path.join("models_iris", f"{model_name}.pkl"),
        os.path.join("models_wine", f"{model_name}.pkl"),
        os.path.join("models_breast_cancer", f"{model_name}.pkl"),
        os.path.join("models_digits", f"{model_name}.pkl"),
        os.path.join("training_results", "models_iris", f"{model_name}.pkl"),
        os.path.join("training_results", "models_wine", f"{model_name}.pkl"),
        os.path.join("training_results", "models_breast_cancer", f"{model_name}.pkl"),
        os.path.join("training_results", "models_digits", f"{model_name}.pkl")
    ])
    
    for path in search_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        available_models = get_available_models()
        available_names = list(available_models.keys())
        raise FileNotFoundError(
            f"Modelo {model_name} no encontrado. "
            f"Modelos disponibles: {available_names[:10]}{'...' if len(available_names) > 10 else ''}"
        )
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modelo {model_name} cargado desde {model_path}")
        return model, model_path
    except Exception as e:
        raise Exception(f"Error cargando modelo {model_name} desde {model_path}: {str(e)}")


def get_available_models() -> Dict[str, Dict]:
    """Obtiene información de todos los modelos disponibles con búsqueda exhaustiva"""
    models_info = {}
    
    search_directories = [
        "models",
        "models_iris", "models_wine", "models_breast_cancer", "models_digits",
        os.path.join("training_results", "models_iris"),
        os.path.join("training_results", "models_wine"),
        os.path.join("training_results", "models_breast_cancer"),
        os.path.join("training_results", "models_digits")
    ]
    
    base_models_dir = Path("models")
    if base_models_dir.exists():
        for subdir in base_models_dir.iterdir():
            if subdir.is_dir():
                search_directories.append(str(subdir))
    
    for directory in search_directories:
        directory_path = Path(directory)
        if not directory_path.exists():
            continue

        if "iris" in directory:
            dataset_name = "iris"
        elif "wine" in directory:
            dataset_name = "wine"
        elif "breast_cancer" in directory:
            dataset_name = "breast_cancer"
        elif "digits" in directory:
            dataset_name = "digits"
        else:
            dataset_name = directory_path.name if directory_path.name != "models" else "unknown"
        
        for model_file in directory_path.glob("*.pkl"):
            model_name = model_file.stem
            

            unique_model_name = f"{dataset_name}_{model_name}" if dataset_name != "unknown" else model_name

            if model_name in [info.get("model_name") for info in models_info.values()]:
                unique_model_name = f"{dataset_name}_{model_name}"
            
            model_info = {
                "model_name": model_name,
                "unique_name": unique_model_name,
                "dataset": dataset_name,
                "file_path": str(model_file),
                "directory": str(directory_path),
                "file_size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                "modified_time": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            }
            
            result_files = [
                f"results_{dataset_name}.json",
                os.path.join("training_results", f"results_{dataset_name}.json")
            ]
            for result_file in result_files:
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                            
                            # Handle different formats of results file
                            if model_name in results:
                                # Standard format where model name is a direct key
                                result = results[model_name]
                                model_info.update({
                                    "accuracy": result.get("accuracy", 0),
                                    "cv_mean": result.get("cv_mean", 0),
                                    "cv_std": result.get("cv_std", 0),
                                    "training_time": result.get("training_time", 0),
                                    "status": result.get("status", "success"),
                                    "timestamp": result.get("timestamp", "")
                                })
                                break
                            elif dataset_name in results and model_name in results[dataset_name]:
                                # Format from run function where dataset is the top-level key
                                result = results[dataset_name][model_name]
                                model_info.update({
                                    "accuracy": result.get("val_accuracy", result.get("accuracy", 0)),
                                    "cv_mean": result.get("val_accuracy", 0),
                                    "cv_std": 0.0,
                                    "training_time": result.get("training_time", 0),
                                    "status": "success",
                                    "timestamp": datetime.now().isoformat()
                                })
                                break
                    except Exception as e:
                        logger.warning(f"Error cargando métricas desde {result_file}: {e}")
            
            models_info[unique_model_name] = model_info
    
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

@app.post("/data/load",summary='carga un dataset .csv')
async def laod_data(path:str):
    if not path.endswith('.csv'):
        raise HTTPException(status_code=500,details='error abriendo el archivo, tiene que ser un csv') 
    #enviar al backend
# tengo que crear otra para configurar parametros


@app.get("/models/{model_name}", summary="Información de un modelo específico")
async def get_model_info(model_name: str):
    """Obtiene información detallada de un modelo específico"""
    try:
        models = get_available_models()
        
        if model_name not in models:
            matching_models = [k for k in models.keys() if model_name in k or k.endswith(f"_{model_name}")]
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
                model_info.get("dataset")
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
    import time
    start_time = time.time()
    
    try:
        models = get_available_models()

        model_key = None
        if request.model_name in models:
            model_key = request.model_name
        else:
            matching_models = [k for k in models.keys() if 
                             request.model_name in k or 
                             k.endswith(f"_{request.model_name}") or
                             k.startswith(f"{request.model_name}_")]
            if matching_models:
                model_key = matching_models[0]
                logger.info(f"Usando modelo {model_key} para solicitud de {request.model_name}")
        
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
        predictions = model.predict(X)
        
        prediction_time = time.time() - start_time
        
        response_data = {
            "model_name": model_key,
            "predictions": predictions.tolist(),
            "feature_count": X.shape[1],
            "prediction_time": prediction_time,
            "model_path": model_path
        }
        
        model_accuracy = model_info.get("accuracy")
        save_inference_stats(model_key, prediction_time, X.shape[0], model_accuracy)

        if request.return_probabilities:
            try:
                probabilities = model.predict_proba(X)
                response_data["probabilities"] = probabilities.tolist()
                logger.info(f"Probabilidades calculadas para {X.shape[0]} muestras")
            except AttributeError:
                logger.warning(f"Modelo {model_key} no soporta predict_proba")
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


@app.get("/datasets", summary="Datasets disponibles")
async def list_datasets():
    """Lista los datasets disponibles para entrenamiento"""
    try:
        trainer = get_trainer()
        datasets = trainer.get_available_datasets()
        
        dataset_info = {}
        for name, dataset in datasets.items():
            dataset_info[name] = {
                "name": name,
                "n_samples": dataset.data.shape[0],
                "n_features": dataset.data.shape[1],
                "n_classes": len(np.unique(dataset.target)),
                "feature_names": getattr(dataset, 'feature_names', []).tolist() if hasattr(dataset, 'feature_names') else [],
                "target_names": getattr(dataset, 'target_names', []).tolist() if hasattr(dataset, 'target_names') else []
            }
        
        return {
            "total_datasets": len(dataset_info),
            "datasets": dataset_info
        }
        
    except Exception as e:
        logger.error(f"Error listando datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando datasets: {str(e)}")


@app.delete("/models/{model_name}", summary="Eliminar modelo", response_model=None)
async def delete_model(model_name: str):
    """Elimina un modelo entrenado del almacenamiento"""
    try:
        logger.info(f"Solicitud para eliminar modelo: {model_name}")
        models = get_available_models()
        logger.info(f"Total de modelos disponibles: {len(models)}")

        model_key = None
        if model_name in models:
            model_key = model_name
            logger.info(f"Modelo encontrado exactamente: {model_key}")
        else:
            matching_models = [k for k in models.keys() if 
                             model_name in k or 
                             k.endswith(f"_{model_name}") or
                             k.startswith(f"{model_name}_")]
            if matching_models:
                model_key = matching_models[0]
                logger.info(f"Modelo encontrado por coincidencia parcial: {model_key}")
            else:
                logger.warning(f"No se encontraron coincidencias para modelo: {model_name}")
        
        if not model_key:
            available_models = list(models.keys())
            logger.warning(f"Modelo no encontrado. Disponibles: {available_models}")
            raise HTTPException(
                status_code=404, 
                detail=f"Modelo {model_name} no encontrado. Modelos disponibles: {available_models[:10]}{'...' if len(available_models) > 10 else ''}"
            )
        
        model_info = models[model_key]
        model_path = model_info["file_path"]
        logger.info(f"Intentando eliminar archivo: {model_path}")
        
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Modelo {model_key} eliminado exitosamente de {model_path}")
                  # Limpiar caché si existe
                global models_cache
                models_cache = {}  # Limpiar toda la caché para forzar recargar todos los modelos
                logger.info(f"Cache de modelos limpiada completamente")
                
                # Verificar que el modelo ya no existe
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
    import subprocess
    """Agrega un nodo al cluster Ray (worker externo)"""
    try:
            
        command = f"docker run -d --name ray_worker_{worker_name.lower().replace(" ","_")} --hostname ray_worker_{worker_name} --network scr_pasd_2025_ray-network -e RAY_HEAD_SERVICE_HOST=ray-head -e NODE_ROLE=worker -e LEADER_NODE=false -e FAILOVER_PRIORITY=3 -e ENABLE_AUTO_FAILOVER=false --shm-size=2gb scr_pasd_2025-ray-head bash -c 'echo Worker externo iniciando... && echo Esperando al cluster principal... && sleep 10 && echo Conectando al cluster existente... && ray start --address=ray-head:6379 --num-cpus={add_cpu} --object-manager-port=8076 --node-manager-port=8077 --min-worker-port=10002 --max-worker-port=19999 && echo Worker externo conectado exitosamente! && tail -f /dev/null'"
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
        import subprocess
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
