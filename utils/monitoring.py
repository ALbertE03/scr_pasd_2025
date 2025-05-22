"""
Utilidades para la monitorización del sistema y el cluster Ray.
"""
import ray
import psutil
import time
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ray_status() -> Dict[str, Any]:
    """
    Obtiene el estado actual del cluster Ray.
    
    Returns:
        Dict[str, Any]: Diccionario con información del estado del cluster
    """
    if not ray.is_initialized():
        return {
            "initialized": False,
            "message": "Ray no está inicializado",
            "nodes": [],
            "resources": {}
        }
    
    try:
        # Obtener información sobre los nodos
        nodes = ray.nodes()
        
        # Obtener recursos disponibles y totales
        available_resources = ray.available_resources()
        total_resources = ray.cluster_resources()
        
        # Calcular recursos utilizados
        used_resources = {}
        for resource in total_resources:
            if resource in available_resources:
                used_resources[resource] = total_resources[resource] - available_resources.get(resource, 0)
            else:
                used_resources[resource] = total_resources[resource]
        
        # Dashboard URL
        try:
            dashboard_url = ray.get_dashboard_url()
        except:
            dashboard_url = None
        
        return {
            "initialized": True,
            "dashboard_url": dashboard_url,
            "nodes": nodes,
            "node_count": len(nodes),
            "available_resources": available_resources,
            "total_resources": total_resources,
            "used_resources": used_resources,
        }
    except Exception as e:
        logger.error(f"Error al obtener estado de Ray: {str(e)}")
        return {
            "initialized": True,
            "error": str(e),
            "nodes": [],
            "resources": {}
        }

def get_system_resources() -> Dict[str, Any]:
    """
    Obtiene información sobre los recursos del sistema local.
    
    Returns:
        Dict[str, Any]: Diccionario con información de recursos del sistema
    """
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # Network info
        net_io = psutil.net_io_counters()
        
        # Process info
        current_process = psutil.Process()
        process_memory = current_process.memory_info()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            },
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms
            }
        }
    except Exception as e:
        logger.error(f"Error al obtener recursos del sistema: {str(e)}")
        return {
            "error": str(e)
        }

def monitor_training_progress(model_ids: List[str], registry, interval: int = 5, max_time: int = 600) -> Dict[str, Any]:
    """
    Monitorea el progreso del entrenamiento de modelos.
    
    Args:
        model_ids: Lista de IDs de modelos a monitorear
        registry: Instancia del registro de modelos
        interval: Intervalo de monitoreo en segundos
        max_time: Tiempo máximo de monitoreo en segundos
        
    Returns:
        Dict[str, Any]: Resultados del monitoreo
    """
    start_time = time.time()
    results = {
        "models": {},
        "system_stats": [],
        "ray_stats": []
    }
    
    while time.time() - start_time < max_time:
        # Recolectar estadísticas del sistema
        system_resources = get_system_resources()
        system_resources["timestamp"] = time.time() - start_time
        results["system_stats"].append(system_resources)
        
        # Recolectar estadísticas de Ray
        ray_status = get_ray_status()
        ray_status["timestamp"] = time.time() - start_time
        results["ray_stats"].append(ray_status)
        
        # Comprobar el estado de cada modelo
        all_complete = True
        for model_id in model_ids:
            try:
                metadata = registry.get_metadata(model_id)
                status = metadata.get("status", "unknown")
                
                if model_id not in results["models"]:
                    results["models"][model_id] = []
                
                results["models"][model_id].append({
                    "timestamp": time.time() - start_time,
                    "status": status,
                    "metrics": metadata.get("metrics", {})
                })
                
                if status != "active":
                    all_complete = False
            except Exception as e:
                logger.error(f"Error al obtener metadata del modelo {model_id}: {str(e)}")
        
        # Si todos los modelos están completos, terminar
        if all_complete:
            break
            
        # Esperar antes de la siguiente comprobación
        time.sleep(interval)
    
    return results
