import streamlit as st
import ray
import json
import time
import logging
import os
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ray_utils')

def get_unique_key(base_key):
    """Genera una key única incrementando un contador"""
    if 'widget_counter' not in st.session_state:
        st.session_state.widget_counter = 0
    st.session_state.widget_counter += 1
    return f"{base_key}_{st.session_state.widget_counter}"

def initialize_session_state():
    """Inicializa todas las variables de session_state necesarias"""
    
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'sequential_results' not in st.session_state:
        st.session_state.sequential_results = None
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = 'iris'
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
    

    if 'test_size' not in st.session_state:
        st.session_state.test_size = 0.3  
        st.session_state.enable_fault_tolerance = True
        
    if 'enable_leader_failover' not in st.session_state:
        st.session_state.enable_leader_failover = True

    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    if 'fault_config' not in st.session_state:
        st.session_state.fault_config = {
            'max_retries': 3,
            'timeout_seconds': 300,
            'enable_reconstruction': True,
            'enable_auto_retry': True,
            'monitoring_interval': 30
        }

    if 'fault_logs' not in st.session_state:
        st.session_state.fault_logs = []
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    
    if 'current_leader' not in st.session_state:
        st.session_state.current_leader = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')

    if 'widget_counter' not in st.session_state:
        st.session_state.widget_counter = 0

    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
        
def load_custom_styles():
    """Carga los estilos CSS personalizados para la aplicación"""
    st.markdown("""
    <style>
        /* Estilos para modo oscuro y claro con detección automática */
        
        /* Estilos globales - Modo oscuro */
        .stApp {
            background: linear-gradient(135deg, #1e1e2e 0%, #171728 100%);
        }
        
        /* Contenedor principal con fondo - Compatible con modo oscuro */
        .main .block-container {
            background: rgba(30, 30, 46, 0.7);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        /* Tarjetas con gradientes y efectos sutiles - Modo oscuro */
        .metric-card {
            background: linear-gradient(120deg, #292a3e 0%, #1e1e2e 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border-left: 4px solid #bd93f9;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            color: #cdd6f4;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .success-card {
            background: linear-gradient(120deg, #20303b 0%, #264531 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #a6e3a1;
            border-left: 4px solid #a6e3a1;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .warning-card {
            background: linear-gradient(120deg, #302a22 0%, #3d2e1b 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #f9e2af;
            border-left: 4px solid #f9e2af;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .error-card {
            background: linear-gradient(120deg, #362231 0%, #3f1d1d 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #f38ba8;
            border-left: 4px solid #f38ba8;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .info-card {
            background: linear-gradient(120deg, #1e293b 0%, #1a2332 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #89b4fa;
            border-left: 4px solid #89b4fa;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        
        /* Estilo para los tabs - Modo oscuro */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-radius: 12px;
            padding: 0.6rem;
            background: rgba(30, 30, 46, 0.7);
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(189, 147, 249, 0.1);
            color: #cdd6f4;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(189, 147, 249, 0.15);
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            color: #1e1e2e;
            box-shadow: 0 3px 10px rgba(189, 147, 249, 0.4);
            transform: translateY(-2px);
            font-weight: 600;
        }
        
        /* Animaciones sutiles */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(189, 147, 249, 0.3); }
            70% { box-shadow: 0 0 0 10px rgba(189, 147, 249, 0); }
            100% { box-shadow: 0 0 0 0 rgba(189, 147, 249, 0); }
        }
        .dashboard-container {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Dashboard Header Mejorado - Modo oscuro */
        .dashboard-header {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: #1e1e2e;
            box-shadow: 0 8px 20px rgba(189, 147, 249, 0.3);
            text-align: center;
            animation: fadeIn 0.8s ease-out;
        }
        .dashboard-header h1 {
            font-weight: 700;
            letter-spacing: 1px;
            color: #1e1e2e;
        }
        .dashboard-header h3 {
            opacity: 0.9;
            font-weight: 500;
            color: #2d2d44;
        }
        
        /* Estilos para gráficos y visualizaciones - Modo oscuro */
        .plotly-chart-container {
            background: rgba(30, 30, 46, 0.7);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            border: 1px solid rgba(189, 147, 249, 0.1);
        }
        .plotly-chart-container:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.25);
            transform: translateY(-2px);
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        
        /* Mejorar métricos - Modo oscuro */
        [data-testid="stMetric"] {
            background: linear-gradient(120deg, #292a3e 0%, #1e1e2e 100%);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #cdd6f4;
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #f8f8f2;
            font-size: 1.8rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 500;
        }
        
        /* Etiquetas y contenido - Modo oscuro */
        h1, h2, h3 {
            color: #f8f8f2;
        }
        h1 {
            font-weight: 700;
        }
        h2 {
            font-weight: 600;
        }
        h3 {
            font-weight: 500;
        }
        
        /* Estilo para los selectboxes y demás widgets - Modo oscuro */
        .stSelectbox [data-baseweb="select"] {
            background-color: #292a3e;
            border-radius: 8px;
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        .stMultiSelect [data-baseweb="select"] {
            background-color: #292a3e;
            border-radius: 8px;
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        
        /* Checkbox y Radio - Modo oscuro */
        .stCheckbox [data-baseweb="checkbox"] {
            padding: 0.5rem;
            border-radius: 6px;
            transition: background 0.2s ease;
        }
        .stCheckbox [data-baseweb="checkbox"]:hover {
            background: rgba(189, 147, 249, 0.1);
        }
        .stRadio [data-baseweb="radio"] {
            padding: 0.5rem;
            border-radius: 6px;
            transition: background 0.2s ease;
        }
        .stRadio [data-baseweb="radio"]:hover {
            background: rgba(189, 147, 249, 0.1);
        }
        
        /* Sliders - Modo oscuro */
        .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background: #bd93f9;
            border: 2px solid #1e1e2e;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Sidebar - Modo oscuro */
        [data-testid="stSidebar"] {
            background-color: rgba(25, 25, 35, 0.8);
            border-right: 1px solid rgba(189, 147, 249, 0.1);
        }
        
        /* Botones - Modo oscuro */
        .stButton > button {
            border-radius: 8px;
            padding: 0.3rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stButton > [data-baseweb="button"][kind="primary"] {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            color: #1e1e2e;
        }
        
        /* Data tables - Modo oscuro */
        [data-testid="stTable"] {
            background-color: #292a3e;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(189, 147, 249, 0.1);
        }
        [data-testid="stTable"] table {
            border-collapse: separate;
            border-spacing: 0;
        }
        [data-testid="stTable"] thead th {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-weight: 600;
            border-bottom: 1px solid rgba(189, 147, 249, 0.3);
        }
        [data-testid="stTable"] tbody tr:nth-child(even) {
            background-color: rgba(189, 147, 249, 0.05);
        }
        [data-testid="stTable"] tbody tr:hover {
            background-color: rgba(189, 147, 249, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def detect_current_ray_leader():
    """Detecta quién es el líder actual del clúster Ray
    
    Esta función busca en los contenedores Docker para identificar el nodo líder,
    con preferencia por 'ray-leader-new' cuando existe.
    
    Returns:
        str: El nombre del contenedor que actúa como líder
    """
    try:
        import docker
        
        # Intentar con cliente Docker
        try:
            client = docker.from_env()
            
            # 1. Buscar primero el nuevo líder por nombre específico
            try:
                new_leader = client.containers.get('ray-leader-new')
                if new_leader.status == 'running':
                    logger.info("Detectado nuevo líder: ray-leader-new")
                    return 'ray-leader-new'
            except:
                pass
                
            # 2. Buscar el líder original
            try:
                orig_leader = client.containers.get('ray-head')
                if orig_leader.status == 'running':
                    logger.info("Detectado líder original: ray-head")
                    return 'ray-head'
            except:
                pass
            
            # 3. Buscar cualquier contenedor que tenga la variable de entorno LEADER_NODE=true
            containers = client.containers.list(filters={"status": "running"})
            for container in containers:
                if container.name.startswith("ray-"):
                    try:
                        # Inspeccionar variables de entorno
                        env_vars = container.attrs['Config']['Env']
                        for env in env_vars:
                            if env.startswith("LEADER_NODE=true"):
                                logger.info(f"Detectado líder por variable de entorno: {container.name}")
                                return container.name
                    except:
                        continue
            
            # Si llegamos aquí, no encontramos un líder definido
            logger.warning("No se pudo detectar un líder específico, usando valor por defecto")
            return os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
            
        except Exception as e:
            logger.error(f"Error al conectar con Docker: {e}")
            return os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
            
    except ImportError:
        logger.warning("Módulo docker no disponible, usando configuración por defecto")
        return os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')

def connect_to_ray_cluster(max_retries=3):
    """Conecta al clúster Ray con manejo automático de redirección al nuevo líder
    
    Esta función intenta conectar al clúster Ray y maneja el caso en que el nodo líder
    haya cambiado debido a un failover.
    
    Args:
        max_retries: Número máximo de intentos de conexión
        
    Returns:
        bool: True si la conexión fue exitosa, False en caso contrario
    """
    # Detectar el líder actual consultando Docker
    current_head = detect_current_ray_leader()
    st.session_state['current_leader'] = current_head
    
    logger.info(f"Intentando conectar al clúster Ray. Nodo líder detectado: {current_head}")
    
    # Si Ray ya está conectado, verificar si está conectado al líder correcto
    if ray.is_initialized():
        try:
            # Verificar si la conexión actual es válida
            ray.get(ray.put(1))
            logger.info(f"Ya conectado al clúster Ray (verificando si es el líder correcto)")
            
            # Obtener información de la conexión actual
            current_address = ray._private.worker.global_worker.node_ip_address
            logger.info(f"Dirección de conexión actual: {current_address}")
            
            # Si es el líder correcto, seguimos conectados, de lo contrario reconectamos
            if current_address == current_head or current_address == "127.0.0.1":
                logger.info(f"Mantenemos conexión actual a Ray")
                return True
                
            # Si no, desconectamos para reconectar al nuevo líder
            logger.info(f"Desconectando de {current_address} para reconectar a {current_head}")
            ray.shutdown()
        except:
            # La conexión existente falló, desconectar para reconectar
            logger.warning("Conexión existente a Ray inválida, reconectando...")
            ray.shutdown()
    
    # Intentar conexión con múltiples reintentos
    for attempt in range(max_retries):
        try:
            # Intentar conectar al líder conocido
            ray.init(address=f"{current_head}:6379", ignore_reinit_error=True)
            logger.info(f"Conectado exitosamente al clúster Ray ({current_head})")
            st.session_state.current_leader = current_head
            return True
        except Exception as e:
            logger.warning(f"Error al conectar a Ray en {current_head} (intento {attempt+1}/{max_retries}): {e}")
            
            # Si falla, intentar cargar el nuevo líder desde la configuración
            try:
                if os.path.exists("leader_config.json"):
                    with open("leader_config.json", "r") as f:
                        config = json.load(f)
                        new_head = config.get("active_head_node")
                        
                        if new_head and new_head != current_head:
                            logger.info(f"Detectado nuevo líder en config: {new_head}, intentando reconectar...")
                            current_head = new_head
                            st.session_state.current_leader = new_head
                            continue
            except:
                pass
                
            # Si estamos en el último intento, buscar cualquier nodo Ray disponible
            if attempt == max_retries - 1:
                from modules.failover import get_eligible_nodes_for_promotion
                eligible_nodes = get_eligible_nodes_for_promotion()
                
                if eligible_nodes:
                    node = eligible_nodes[0]
                    node_name = node["name"] if isinstance(node, dict) else node
                    logger.info(f"Intentando conectar a nodo alternativo: {node_name}")
                    
                    try:
                        ray.init(address=f"{node_name}:6379", ignore_reinit_error=True)
                        logger.info(f"Conectado exitosamente a nodo alternativo: {node_name}")
                        st.session_state.current_leader = node_name
                        return True
                    except Exception as alt_e:
                        logger.error(f"Error al conectar a nodo alternativo: {alt_e}")
            
            # Esperar antes del siguiente intento
            time.sleep(2)
    
    logger.error(f"No se pudo conectar al clúster Ray después de {max_retries} intentos")
    return False

def save_system_metrics_history(metrics):
    """Guarda las métricas del sistema en un archivo de historial"""
    try:
        history_file = 'system_metrics_history.json'
        
        # Cargar historial existente
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Agregar timestamp actual
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Agregar nueva entrada
        history.append(metrics)
        
        # Mantener solo las últimas 24 horas (asumiendo una entrada cada 5 minutos)
        max_entries = 24 * 12  # 288 entradas
        if len(history) > max_entries:
            history = history[-max_entries:]
        
        # Guardar historial actualizado
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error guardando historial de métricas: {e}")

def load_system_metrics_history():
    """Carga el historial de métricas del sistema"""
    try:
        history_file = 'system_metrics_history.json'
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Filtrar últimas 12 horas para el gráfico
            now = datetime.now()
            twelve_hours_ago = now - timedelta(hours=12)
            
            filtered_history = []
            for entry in history:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= twelve_hours_ago:
                    filtered_history.append(entry)
            
            return filtered_history
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error cargando historial de métricas: {e}")
        return []

def get_metrics_for_timeframe(hours=12):
    """Obtiene métricas para un marco de tiempo específico"""
    history = load_system_metrics_history()
    
    if not history:
        return {
            'timestamps': [],
            'cpu_values': [],
            'memory_values': [],
            'disk_values': []
        }
    
    timestamps = []
    cpu_values = []
    memory_values = []
    disk_values = []
    
    for entry in history:
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            timestamps.append(timestamp.strftime('%H:%M'))
            cpu_values.append(entry.get('cpu_percent', 0))
            memory_values.append(entry.get('memory_percent', 0))
            disk_values.append(entry.get('disk_percent', 0))
        except Exception as e:
            logger.warning(f"Error procesando entrada de historial: {e}")
            continue
    
    return {
        'timestamps': timestamps,
        'cpu_values': cpu_values,
        'memory_values': memory_values,
        'disk_values': disk_values
    }

def start_metrics_collection():
    return True
