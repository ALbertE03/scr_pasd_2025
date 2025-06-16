import ray
import os
import time
import subprocess
import logging
from threading import Thread
import docker
import streamlit as st
import json
from typing import Dict, List, Optional, Union
import requests
import socket

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ray_failover')

# Variables globales para la configuración del failover
HEAD_NODE_NAME = "ray-head"
HEAD_NODE_ADDRESS = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
HEAD_NODE_PORT = 6379
BACKUP_NODES_FILE = "backup_nodes.json"
HEALTH_CHECK_INTERVAL = 5  # segundos
FAILOVER_MONITOR_ACTIVE = False
monitoring_thread = None

class FailoverStatus:
    """Clase para mantener el estado del failover"""
    def __init__(self):
        self.is_primary = False  # True si este nodo es el líder actual
        self.original_head_healthy = True
        self.active_head_node = HEAD_NODE_NAME
        self.backup_nodes = []
        self.last_failover_time = None
        self.failover_count = 0
        self.monitoring_active = False

# Estado global de failover
failover_status = FailoverStatus()

def get_docker_client():
    """Obtiene el cliente Docker"""
    try:
        return docker.from_env()
    except Exception as e:
        logger.error(f"Error al conectar con Docker: {e}")
        return None

def is_head_node_healthy() -> bool:
    """Verifica si el nodo líder está saludable"""
    try:
        # Intentar conexión TCP al puerto de Ray en el nodo líder
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # Timeout de 2 segundos
        result = sock.connect_ex((HEAD_NODE_ADDRESS, HEAD_NODE_PORT))
        sock.close()
        
        if result == 0:
            # Puerto abierto, intentar conectarse a Ray
            try:
                # Intentar conectarse al dashboard de Ray como prueba adicional
                response = requests.get(f"http://{HEAD_NODE_ADDRESS}:8265/api/cluster_status", timeout=3)
                return response.status_code == 200
            except:
                # Si hay error al conectar al dashboard pero el puerto está abierto,
                # asumimos que Ray está iniciando o el dashboard no está disponible
                return True
        return False
    except Exception as e:
        logger.error(f"Error al verificar salud del nodo líder: {e}")
        return False

def get_eligible_nodes_for_promotion() -> List[str]:
    """Obtiene la lista de nodos que pueden ser promovidos a líder"""
    try:
        client = get_docker_client()
        if not client:
            return []
        
        # Obtener todos los contenedores con la etiqueta ray-worker
        containers = client.containers.list(
            filters={"name": "ray-worker", "status": "running"}
        )
        
        # Ordenar por tiempo de creación (para tener consistencia en las promociones)
        containers.sort(key=lambda x: x.attrs['Created'])
        
        # Devolver los nombres de los contenedores
        return [container.name for container in containers]
    except Exception as e:
        logger.error(f"Error al obtener nodos elegibles: {e}")
        return []

def save_backup_nodes(nodes: List[str]):
    """Guarda la lista de nodos de respaldo en un archivo"""
    try:
        with open(BACKUP_NODES_FILE, 'w') as f:
            json.dump({"backup_nodes": nodes}, f)
    except Exception as e:
        logger.error(f"Error al guardar nodos de respaldo: {e}")

def load_backup_nodes() -> List[str]:
    """Carga la lista de nodos de respaldo desde el archivo"""
    try:
        if os.path.exists(BACKUP_NODES_FILE):
            with open(BACKUP_NODES_FILE, 'r') as f:
                data = json.load(f)
                return data.get("backup_nodes", [])
        return []
    except Exception as e:
        logger.error(f"Error al cargar nodos de respaldo: {e}")
        return []

def promote_node_to_leader(node_name: str) -> bool:
    """Promueve un nodo worker a líder"""
    try:
        logger.info(f"Promoviendo nodo {node_name} a líder...")
        
        # 1. Ejecutar comando para convertir el nodo en líder
        command = f"""
        docker exec {node_name} bash -c "
        ray stop &&
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=2 --object-manager-port=8076 --node-manager-port=8077 --min-worker-port=10002 --max-worker-port=19999
        "
        """
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error al promover nodo: {result.stderr}")
            return False
        
        # 2. Actualizar la configuración de red para redirigir el tráfico al nuevo líder
        update_network_config(node_name)
        
        # 3. Actualizar el estado global
        failover_status.is_primary = True
        failover_status.active_head_node = node_name
        failover_status.last_failover_time = time.time()
        failover_status.failover_count += 1
        
        logger.info(f"¡Nodo {node_name} promovido exitosamente a líder!")
        return True
    except Exception as e:
        logger.error(f"Error al promover nodo: {e}")
        return False

def update_network_config(new_head_node: str):
    """Actualiza la configuración de red para redireccionar al nuevo líder"""
    try:
        logger.info(f"Actualizando configuración de red para nuevo líder: {new_head_node}")
        
        # 1. Actualizar la variable de entorno en todos los workers
        client = get_docker_client()
        if not client:
            return
        
        workers = client.containers.list(
            filters={"name": "ray-worker", "status": "running"}
        )
        
        for worker in workers:
            if worker.name != new_head_node:
                logger.info(f"Actualizando configuración en worker: {worker.name}")
                # Detener Ray en el worker
                worker.exec_run("ray stop")
                # Actualizar la variable de entorno y reiniciar la conexión
                worker.exec_run(f"export RAY_HEAD_SERVICE_HOST={new_head_node}")
                # Reconectar al nuevo líder
                worker.exec_run(f"""
                ray start --address={new_head_node}:6379 --num-cpus=2 --object-manager-port=8076 --node-manager-port=8077 --min-worker-port=10002 --max-worker-port=19999
                """)
        
        # 2. Crear un proxy de red si es necesario (avanzado)
        # Este paso es más complejo y depende de la configuración de red específica
        
        logger.info("Configuración de red actualizada")
    except Exception as e:
        logger.error(f"Error al actualizar configuración de red: {e}")

def monitor_head_node():
    """Monitorea continuamente el estado del nodo líder y ejecuta failover si es necesario"""
    global failover_status
    
    logger.info("Iniciando monitorización de failover...")
    failover_status.monitoring_active = True
    
    # Cargar nodos de respaldo si existen
    backup_nodes = load_backup_nodes()
    if backup_nodes:
        failover_status.backup_nodes = backup_nodes
    
    try:
        while failover_status.monitoring_active:
            is_healthy = is_head_node_healthy()
            
            if not is_healthy and failover_status.original_head_healthy:
                logger.warning("¡ALERTA! Nodo líder principal caído, iniciando failover...")
                failover_status.original_head_healthy = False
                
                # Obtener nodos elegibles para promoción
                eligible_nodes = get_eligible_nodes_for_promotion()
                
                if eligible_nodes:
                    # Seleccionar el primer nodo elegible (podemos implementar una lógica más compleja)
                    new_leader = eligible_nodes[0]
                    logger.info(f"Seleccionado {new_leader} como nuevo líder")
                    
                    # Promover nodo
                    success = promote_node_to_leader(new_leader)
                    
                    if success:
                        # Actualizar nodos de respaldo
                        failover_status.backup_nodes = [node for node in eligible_nodes if node != new_leader]
                        save_backup_nodes(failover_status.backup_nodes)
                        
                        logger.info("Failover completado exitosamente")
                    else:
                        logger.error("Error durante el failover")
                else:
                    logger.error("No hay nodos elegibles para failover")
            
            # Si el nodo original vuelve a estar disponible, lo registramos
            elif is_healthy and not failover_status.original_head_healthy:
                logger.info("Nodo líder original recuperado")
                # Opcional: podemos implementar una reconexión al líder original
            
            time.sleep(HEALTH_CHECK_INTERVAL)
    except Exception as e:
        logger.error(f"Error en monitorización de failover: {e}")
    finally:
        failover_status.monitoring_active = False
        logger.info("Monitorización de failover detenida")

def start_failover_monitoring():
    """Inicia el hilo de monitorización de failover"""
    global monitoring_thread, FAILOVER_MONITOR_ACTIVE
    
    if monitoring_thread is None or not monitoring_thread.is_alive():
        FAILOVER_MONITOR_ACTIVE = True
        monitoring_thread = Thread(target=monitor_head_node, daemon=True)
        monitoring_thread.start()
        logger.info("Monitorización de failover iniciada")
        return True
    else:
        logger.warning("La monitorización de failover ya está activa")
        return False

def stop_failover_monitoring():
    """Detiene el hilo de monitorización de failover"""
    global failover_status, FAILOVER_MONITOR_ACTIVE
    
    if failover_status.monitoring_active:
        failover_status.monitoring_active = False
        FAILOVER_MONITOR_ACTIVE = False
        logger.info("Deteniendo monitorización de failover...")
        time.sleep(HEALTH_CHECK_INTERVAL + 1)  # Esperar a que termine la iteración
        return True
    else:
        logger.warning("La monitorización de failover no está activa")
        return False

def revert_to_original_head():
    """Revierte al nodo líder original si está disponible"""
    try:
        if not failover_status.original_head_healthy:
            is_healthy = is_head_node_healthy()
            if is_healthy:
                logger.info("El nodo líder original está disponible, revirtiendo...")
                
                # Actualizar configuración de red
                update_network_config(HEAD_NODE_NAME)
                
                # Actualizar estado
                failover_status.is_primary = False
                failover_status.original_head_healthy = True
                failover_status.active_head_node = HEAD_NODE_NAME
                
                logger.info("Reversión al nodo líder original completada")
                return True
            else:
                logger.warning("El nodo líder original no está disponible")
                return False
        else:
            logger.info("Ya estamos usando el nodo líder original")
            return True
    except Exception as e:
        logger.error(f"Error al revertir al nodo líder original: {e}")
        return False

def get_failover_status() -> Dict:
    """Obtiene el estado actual del sistema de failover"""
    return {
        "monitoring_active": failover_status.monitoring_active,
        "original_head_healthy": failover_status.original_head_healthy,
        "active_head_node": failover_status.active_head_node,
        "backup_nodes": failover_status.backup_nodes,
        "failover_count": failover_status.failover_count,
        "last_failover_time": failover_status.last_failover_time,
        "is_primary": failover_status.is_primary
    }

def render_failover_ui():
    """Renderiza la interfaz de usuario de failover en Streamlit"""
    st.header("Sistema de Failover - Ray Leader")
    
    with st.expander("ℹ️ Información sobre el Failover", expanded=True):
        st.info("""
        **Sistema de Failover para Ray**
        
        Este sistema monitoriza el estado del nodo líder (ray-head) y, en caso de fallo, 
        promueve automáticamente uno de los nodos worker a líder para mantener la continuidad del cluster.
        
        **Funcionamiento:**
        - El monitor verifica periódicamente el estado del nodo líder
        - Si el nodo líder falla, selecciona un worker para promoción
        - Reconfigura la red para que todos los nodos se conecten al nuevo líder
        - Mantiene el cluster operativo sin intervención manual
        
        **Beneficios:**
        - Elimina el punto único de fallo del cluster Ray
        - Garantiza alta disponibilidad para aplicaciones críticas
        - Minimiza el tiempo de inactividad en caso de fallos
        """)
    
    status = get_failover_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estado del Failover")
        st.metric(
            "Estado del Monitor", 
            "🟢 Activo" if status["monitoring_active"] else "🔴 Inactivo"
        )
        st.metric(
            "Estado del Nodo Líder Original", 
            "🟢 Saludable" if status["original_head_healthy"] else "🔴 Caído"
        )
        st.metric(
            "Nodo Líder Actual", 
            status["active_head_node"]
        )
    
    with col2:
        st.subheader("Estadísticas")
        st.metric(
            "Número de Failovers", 
            status["failover_count"]
        )
        
        if status["last_failover_time"]:
            last_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status["last_failover_time"]))
            st.metric(
                "Último Failover", 
                last_time
            )
        else:
            st.metric(
                "Último Failover", 
                "Nunca"
            )
        
        st.metric(
            "Nodos de Respaldo Disponibles", 
            len(status["backup_nodes"])
        )
    
    # Lista de nodos de respaldo
    if status["backup_nodes"]:
        st.subheader("📋 Nodos de Respaldo")
        for i, node in enumerate(status["backup_nodes"]):
            st.text(f"{i+1}. {node} {'(Próximo en línea de promoción)' if i == 0 else ''}")
    else:
        st.warning("No hay nodos de respaldo configurados. Se necesitan al menos 2 workers para tener failover.")

# Iniciar automáticamente la monitorización durante la carga del módulo
try:
    # Esto se ejecutará cuando se importe el módulo
    if not FAILOVER_MONITOR_ACTIVE:
        start_failover_monitoring()
        logger.info("Monitorización de failover iniciada automáticamente durante la carga del módulo")
except Exception as e:
    logger.error(f"Error al iniciar automáticamente la monitorización: {e}")

def configure_failover_at_startup():
    """Configura el sistema de failover al iniciar la aplicación"""
    # Verificar si somos el nodo líder original
    if os.getenv('HOSTNAME') == HEAD_NODE_NAME:
        failover_status.is_primary = True
    
    # Iniciar monitorización si estamos en modo automático (ahora siempre es automático)
    start_failover_monitoring()
    logger.info("Failover automático activado al inicio")
    
    logger.info("Sistema de failover configurado correctamente")
