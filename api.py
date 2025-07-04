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
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator, ValidationError
import uvicorn
import psutil
import ray
from modules.training import Trainer, ModelManager, create_global_model_manager, search_and_load_model, predict_with_stored_model
import subprocess
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Distributed ML API",
    description="API para interactuar con modelos de Machine Learning entrenados en cluster Ray",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Maneja errores de validación de Pydantic con mensajes más claros"""
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_details.append(f"Campo '{field}': {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Error de validación en los datos de entrada",
            "errors": error_details,
            "type": "validation_error"
        }
    )

trainer = None
models_cache = {}
models_directory = "models"
inference_stats_file = "inference_stats.json"
global_model_manager = None


def get_or_create_model_manager():
    """Obtiene o crea una instancia del ModelManager como actor Ray"""
    try:
        
        model_manager = ray.get_actor("model_manager")
        logger.info("ModelManager existente encontrado")
        return model_manager
    except ValueError:
        logger.info("Creando nuevo ModelManager")
        model_manager = create_global_model_manager()
        return model_manager


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Retorna un favicon """
    return JSONResponse(content={"message": "No favicon available"}, status_code=204)


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
            
                    if np.isnan(obj):
                        return None
                    if np.isinf(obj):
                        return None  
                    try:
                        float_val = float(obj)
                        if np.isnan(float_val) or np.isinf(float_val):
                            return None
                        return float_val
                    except (ValueError, OverflowError):
                        return None
                if isinstance(obj, np.ndarray):
                    
                    array_clean = np.nan_to_num(obj, nan=None, posinf=None, neginf=None)
                    return array_clean.tolist()
                if isinstance(obj, (complex, np.complex64, np.complex128)):
       
                    return {"real": float(obj.real), "imag": float(obj.imag)}
                if isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return obj
                return super(NumpyEncoder, self).default(obj)

        return json.dumps(content, cls=NumpyEncoder, allow_nan=False).encode("utf-8")
class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Nombre del modelo a usar")
    features: List[List[Union[int, float, str]]] = Field(..., description="Características para predicción (pueden incluir valores categóricos)")
    feature_names: Optional[List[str]] = Field(None, description="Nombres de las características") 
    return_probabilities: bool = Field(False, description="Retornar probabilidades además de predicciones")
    
    @validator('features')
    def validate_features_basic(cls, v):
        """Validaciones básicas de las características"""
        if not v:
            raise ValueError("No se proporcionaron características")
            
        if len(v) == 0:
            raise ValueError("La lista de características está vacía")
            
        feature_lengths = [len(row) for row in v]
        if len(set(feature_lengths)) > 1:
            raise ValueError(f"Todas las filas deben tener la misma cantidad de características. Encontradas: {set(feature_lengths)}")
            
        return v

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

        safe_prediction_time = prediction_time if prediction_time is not None and np.isfinite(prediction_time) else 0.0
        safe_n_samples = max(1, n_samples) if n_samples is not None else 1
        safe_accuracy = accuracy if accuracy is not None and np.isfinite(accuracy) else None
        
        inference_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction_time": safe_prediction_time,
            "n_samples": n_samples,
            "avg_time_per_sample": safe_prediction_time / safe_n_samples,
            "accuracy": safe_accuracy
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

    base_model_name = model_name.split("_")[0] if "_" in model_name else model_name
    
    for dataset in os.listdir(path):
        path1 = os.path.join(path, dataset)
     
        if not os.path.isdir(path1):
            continue

        models_path = os.path.join(path1, 'models')
        if not os.path.exists(models_path) or not os.path.isdir(models_path):
            continue
            
        for model_file_name in os.listdir(models_path):
            if not model_file_name.endswith(".pkl"):
                continue

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
    """Obtiene información de todos los modelos disponibles usando ModelManager o fallback a archivos"""
    try:
        try:
            model_manager = ray.get_actor("model_manager")
            logger.info("Usando ModelManager para obtener modelos disponibles")
            
            all_model_ids = ray.get(model_manager.list_all_model_ids.remote())
            models_info = {}    
            for model_id in all_model_ids:
                try:
                    scores = ray.get(model_manager.get_scores.remote(model_id))
                    atributes = ray.get(model_manager.get_atributes.remote(model_id))
                    if scores is None:
                        scores = {}
                        logger.warning(f"No se encontraron scores para modelo {model_id}")
                    
                    if not isinstance(scores, dict):
                        logger.warning(f"Scores para modelo {model_id} no es un diccionario, convirtiendo")
                        scores = {}
                    
                    safe_scores = {}
                    for key, value in scores.items():
                        try:
                            if isinstance(value, (int, float, str, bool, list)):
                                safe_scores[key] = value
                            elif hasattr(value, 'tolist'):  
                                safe_scores[key] = value.tolist()
                            else:
                                safe_scores[key] = str(value)
                        except Exception as convert_error:
                            logger.warning(f"Error convirtiendo score {key}: {convert_error}")
                            safe_scores[key] = str(value)
                    
                    parts = model_id.split('_')
                    
                    model_name = parts[-1]
                    dataset_name = "_".join(parts[:-1])

                    object_size = None
                    try:
                        model_ref = ray.get(model_manager.get_model.remote(model_name, dataset_name))
                        if model_ref is not None:
                            model_obj = ray.get(model_ref)
                            import pickle
                            model_bytes = pickle.dumps(model_obj)
                            object_size = len(model_bytes)
                    except Exception as size_error:
                        logger.warning(f"No se pudo obtener el tamaño del objeto para modelo {model_id}: {size_error}")

                    models_info[model_id] = {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "model_id": model_id,
                            "status": "available_in_object_store",
                            "timestamp": datetime.now().isoformat(),
                            'scores': safe_scores,
                            "object_size": object_size,
                            "object_size_mb": round(object_size / (1024 * 1024), 2) if object_size else None,
                            "object_size_kb": round(object_size / 1024, 2) if object_size else None,
                            **atributes
                        }
                        
                except Exception as e:
                    logger.error(f"Error procesando modelo {model_id}: {e}")
                    continue
            
            logger.info(f"Encontrados {len(models_info)} modelos en ModelManager")
            return models_info
            
        except ValueError as ve:
            logger.warning(f"ModelManager no disponible: {ve}. Usando fallback a archivos.")
            return get_models_from_files()
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos del ModelManager: {e}")
        logger.info("Intentando fallback a archivos...")
        return get_models_from_files()


def get_models_from_files() -> Dict[str, Dict]:
    """Fallback: Obtiene modelos disponibles desde archivos en el sistema"""
    models_info = {}
    path = 'train_results'
    
    try:
        if not os.path.exists(path):
            logger.info(f"Directorio {path} no existe")
            return models_info
            
        if not os.listdir(path):
            logger.info("No se encontraron datasets en el directorio train_results")
            return models_info

        for dataset in os.listdir(path):
            dataset_path = os.path.join(path, dataset)
         
            if not os.path.isdir(dataset_path):
                continue

            models_path = os.path.join(dataset_path, 'models')
            if not os.path.exists(models_path) or not os.path.isdir(models_path):
                continue
                
            for model_file_name in os.listdir(models_path):
                if not model_file_name.endswith(".pkl"):
                    continue

                model_path = os.path.join(models_path, model_file_name)
                model_base_name = model_file_name.replace('.pkl', '')
                model_id = f"{dataset}_{model_base_name}"
                
                try:
                    
                    file_stat = os.stat(model_path)
                    file_size = file_stat.st_size
                    file_modified = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    models_info[model_id] = {
                        "model_name": model_base_name,
                        "dataset_name": dataset,
                        "model_id": model_id,
                        "status": "available_in_file",
                        "timestamp": file_modified.isoformat(),
                        "file_path": model_path,
                        "file_size": file_size,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "file_size_kb": round(file_size / 1024, 2),
                        "scores": {}
                    }
                except Exception as e:
                    logger.error(f"Error procesando archivo {model_path}: {e}")
                    continue
        
        logger.info(f"Encontrados {len(models_info)} modelos en archivos")
        return models_info
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos desde archivos: {e}")
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
        
        model = None
        model_path = None
        
        try:
            model_manager = ray.get_actor("model_manager")
            parts = model_key.split('_')
            model_name = parts[-1]
            dataset_name = "_".join(parts[:-1])
            
            logger.info(f"Intentando cargar modelo desde ModelManager: {model_name}, dataset: {dataset_name}")
            model_ref = ray.get(model_manager.get_model.remote(model_name, dataset_name))
            
            if model_ref is not None:
                model = ray.get(model_ref)
                model_path = f"object_store://{model_key}"
                logger.info(f"Modelo {model_key} cargado exitosamente desde ModelManager")
            else:
                logger.warning(f"Modelo {model_key} no encontrado en ModelManager")
                
        except ValueError:
            logger.warning("ModelManager no disponible, usando fallback a archivos")
        except Exception as e:
            logger.warning(f"Error cargando desde ModelManager: {e}, usando fallback a archivos")
    
        if model is None:
            raise HTTPException(status_code=404, detail=f"Modelo {model_key} no encontrado")

        X = np.array(request.features)
        logger.info(request.feature_names)
        df = pd.DataFrame(X, columns=request.feature_names)
        logger.info(df)
        try:
            predictions = model.predict(df)
            logger.info(f'Predicción exitosa: {len(predictions)} resultados')
        except Exception as prediction_error:
            logger.error(f"Error en predicción: {prediction_error}")
            raise Exception(f"Error realizando predicción: {str(prediction_error)}")
        
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

        return PredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/cluster/status", response_model=ClusterInfo, summary="Estado del cluster Ray")
async def cluster_status():
    """Obtiene información extendida del estado del cluster Ray"""
    
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
    """Elimina un modelo entrenado del almacenamiento y del object store"""
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
        deletion_results = {
            "success": True,
            "message": f"Modelo {model_key} eliminado exitosamente",
            "original_request": model_name,
            "timestamp": datetime.now().isoformat(),
            "deleted_from_object_store": False,
            "deleted_from_file": False,
            "object_store_error": None,
            "file_error": None
        }

        try:
            model_manager = ray.get_actor("model_manager")
            delete_result = ray.get(model_manager.delete_model.remote(model_key))
            if delete_result:
                logger.info(f"Modelo {model_key} eliminado exitosamente del object store")
                deletion_results["deleted_from_object_store"] = True
            else:
                logger.warning(f"No se pudo eliminar el modelo {model_key} del object store")
                deletion_results["object_store_error"] = "No se pudo eliminar del object store"
        except ValueError:
            logger.info("ModelManager no disponible, saltando eliminación del object store")
            deletion_results["object_store_error"] = "ModelManager no disponible"
        except Exception as object_store_error:
            logger.error(f"Error eliminando modelo del object store: {object_store_error}")
            deletion_results["object_store_error"] = str(object_store_error)

        model_path = model_info.get("file_path")
        if model_path:
            logger.info(f"Intentando eliminar archivo: {model_path}")
            
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    logger.info(f"Archivo {model_path} eliminado exitosamente")
                    deletion_results["deleted_from_file"] = True
                    deletion_results["deleted_file"] = model_path
                    
                    if os.path.exists(model_path):
                        logger.error(f"ERROR: El archivo {model_path} todavía existe después de intentar eliminarlo")
                        deletion_results["file_error"] = f"El archivo no pudo ser eliminado completamente: {model_path}"
                        
                except PermissionError as perm_error:
                    logger.error(f"Error de permisos al eliminar archivo: {model_path}")
                    deletion_results["file_error"] = f"Sin permisos para eliminar el archivo: {model_path}"
                except Exception as file_error:
                    logger.error(f"Error al eliminar archivo {model_path}: {file_error}")
                    deletion_results["file_error"] = str(file_error)
            else:
                logger.warning(f"Archivo del modelo no encontrado: {model_path}")
                deletion_results["file_error"] = f"Archivo del modelo no encontrado: {model_path}"
        else:
            logger.info("No hay archivo asociado al modelo (solo en object store)")

        
        global models_cache
        models_cache = {} 
        logger.info(f"Cache de modelos limpiada completamente")

        if not deletion_results["deleted_from_object_store"] and not deletion_results["deleted_from_file"]:
            deletion_results["success"] = False
            error_details = []
            if deletion_results["object_store_error"]:
                error_details.append(f"Object store: {deletion_results['object_store_error']}")
            if deletion_results["file_error"]:
                error_details.append(f"Archivo: {deletion_results['file_error']}")
            
            raise HTTPException(
                status_code=500, 
                detail=f"No se pudo eliminar el modelo de ninguna ubicación. Errores: {'; '.join(error_details)}"
            )
        
        return deletion_results
            
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


@app.get("/inference-stats", summary="Estadísticas de inferencia", response_class=NumpyJSONResponse)
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

            valid_entries = []
            for entry in entries:
                if (entry.get("prediction_time") is not None and 
                    np.isfinite(entry.get("prediction_time", 0)) and
                    entry.get("n_samples") is not None and
                    entry.get("n_samples", 0) > 0):
                    valid_entries.append(entry)
            
            if not valid_entries:
                continue
                
            prediction_times = [entry["prediction_time"] for entry in valid_entries]
            avg_times_per_sample = [entry["avg_time_per_sample"] for entry in valid_entries 
                                  if entry.get("avg_time_per_sample") is not None and 
                                  np.isfinite(entry.get("avg_time_per_sample", 0))]
            total_samples = sum(entry["n_samples"] for entry in valid_entries)
            
            safe_avg_prediction_time = (sum(prediction_times) / len(prediction_times) 
                                      if prediction_times else 0.0)
            safe_avg_time_per_sample = (sum(avg_times_per_sample) / len(avg_times_per_sample) 
                                      if avg_times_per_sample else 0.0)
            
            aggregated_stats[model] = {
                "total_predictions": len(valid_entries),
                "total_samples": total_samples,
                "avg_prediction_time": safe_avg_prediction_time,
                "min_prediction_time": min(prediction_times) if prediction_times else 0.0,
                "max_prediction_time": max(prediction_times) if prediction_times else 0.0,
                "avg_time_per_sample": safe_avg_time_per_sample,
                "last_prediction": valid_entries[-1]["timestamp"] if valid_entries else None,
                "accuracy": (valid_entries[-1]["accuracy"] 
                           if valid_entries and valid_entries[-1].get("accuracy") is not None 
                           else None),
                "recent_entries": valid_entries[-50:] if len(valid_entries) > 50 else valid_entries 
            }
        
        response_data = {
            "stats": clean_data_for_json(stats),
            "aggregated": clean_data_for_json(aggregated_stats),
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de inferencia: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")
   
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

@app.get("/system/status", summary="Estado del sistema (host)", response_class=NumpyJSONResponse)
async def system_status():
    """Devuelve información de uso de CPU, memoria y disco del sistema host"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available / (1024**3),  # GB
            "memory_total": memory.total / (1024**3),  # GB
            "disk_percent": disk.percent,
            "disk_free": disk.free / (1024**3),  # GB
            "disk_total": disk.total / (1024**3)  # GB
        }
        
        return clean_data_for_json(system_data)
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
    hyperparams: Optional[Dict[str, Dict[str, Any]]] = Field(default={}, description="Hiperparámetros personalizados por modelo")



def clean_data_for_json(data):
    """
    Recursively clean data to ensure JSON compliance by handling NaN, inf, and other problematic values.
    """
    if isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.ndarray):
        cleaned_array = np.nan_to_num(data, nan=None, posinf=None, neginf=None)
        return cleaned_array.tolist()
    elif isinstance(data, (complex, np.complex64, np.complex128)):
        real_part = float(data.real) if np.isfinite(data.real) else None
        imag_part = float(data.imag) if np.isfinite(data.imag) else None
        return {"real": real_part, "imag": imag_part}
    else:
        return data

@app.post('/train/oneDataset', summary='Entrena varios modelos en el mismo dataset')
async def train(params: TrainingRequest):
    """
    Recibe los parámetros de entrenamiento, lanza el trabajo en segundo plano y devuelve un ID de experimento.
    """
    try:
        logger.info('Iniciando entrenamiento con ModelManager')
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        df = pd.DataFrame(params.dataset)
        logger.info(f"Datos recibidos: {df.shape[0]} filas, {df.shape[1]} columnas")
        model_manager = get_or_create_model_manager()

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
            'estrategia': params.estrategia,
            'dataset_name': params.dataset_name,
            'model_manager': model_manager,
            'hyperparams': params.hyperparams
        }
        
        trainer_actor = Trainer(**actor_params)
        
        result = trainer_actor.train()
        
        stats = trainer_actor.get_model_registry_stats()
        logger.info(f"Entrenamiento completado. Estadísticas del ModelManager: {stats}")
        
        response_data = {
            "message": "Entrenamiento completado con éxito usando ModelManager.",
            "data": result,
            "model_manager_stats": stats,
            "models_trained": len([r for r in result if isinstance(r, dict) and r.get('status') == 'Success']) if isinstance(result, list) else 0
        }
        
        return NumpyJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}", exc_info=True)
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

def main(port:int):
    """Función principal para ejecutar la API FastAPI"""
    logger.info("Iniciando API de Modelos de Machine Learning Distribuidos")
    
    os.makedirs("models", exist_ok=True)
    
    config = {
        "host": "0.0.0.0",
        "port": port,
        "reload": True,
        "log_level": "info"
    }
    
    logger.info(f"Servidor iniciando en http://{config['host']}:{config['port']}")
    logger.info(f"Documentación disponible en http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run("api:app", **config)



def prepare_prediction_data(features, feature_names=None, model_info=None):
    """
    Prepara los datos para predicción de manera simple y robusta
    """
    try:
        if feature_names and len(feature_names) == len(features[0]):
            df = pd.DataFrame(features, columns=feature_names)
        else:
            df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(len(features[0]))])
        
        logger.info(f"DataFrame creado: {df.shape}")
 
        for col in df.columns:
      
            df[col] = df[col].fillna(0)

            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
            
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except:
                  
                    df[col] = 0
        
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        logger.info(f"Datos procesados: {df.shape}, tipos: {df.dtypes.to_dict()}")
        return df
        
    except Exception as e:
        logger.error(f"Error en preparación de datos: {e}")
        fallback_df = pd.DataFrame(0, index=range(len(features)), columns=range(len(features[0])))
        logger.warning("Usando fallback con ceros")
        return fallback_df

if __name__ == "__main__":
    main(8000)
