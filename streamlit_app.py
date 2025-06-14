"""
Ray ML Cluster Dashboard
"""

import streamlit as st
import time
from datetime import datetime

from modules.utils import initialize_session_state, load_custom_styles, get_unique_key
from modules.cluster import get_cluster_status, get_system_metrics, render_cluster_status_tab
from modules.views import (
    render_overview_tab,
    render_training_tab,
    render_results_tab, 
    render_system_metrics_tab,
    render_fault_tolerance_tab
)
from modules.training import (
    load_training_results,
    run_distributed_training,
    run_sequential_training,
    get_fault_tolerance_stats
)

# Configuración de la página
st.set_page_config(
    page_title="Ray ML Cluster Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.ray.io/',
        'About': '### Ray ML Cluster Dashboard\nSistema para entrenamiento distribuido de modelos ML'
    }
)

# Cargar estilos CSS personalizados desde el módulo utils
load_custom_styles()

# Inicializar variables de sesión
initialize_session_state()

# Encabezado de la página
st.markdown("""
<div class="dashboard-header">
    <h1>🚀 Ray ML Cluster Dashboard</h1>
    <h3>Sistema de Entrenamiento ML Distribuido</h3>
</div>
""", unsafe_allow_html=True)

# Barra de menús principal
tab_titles = [
    "📊 Vista General",
    "🔍 Estado del Cluster",
    "🧠 Entrenamiento ML",
    "📈 Resultados",
    "💻 Métricas del Sistema",
    "🛡️ Tolerancia a Fallos"
]

tabs = st.tabs(tab_titles)

# Refrescar estado del cluster
cluster_status = get_cluster_status()
system_metrics = get_system_metrics()

# Añadir opción de auto-refresco
with st.sidebar:
    st.title("⚙️ Configuración")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle(
        "Auto-Refresh (30s)",
        value=st.session_state.auto_refresh,
        key="auto_refresh_toggle"
    )
    
    if auto_refresh:
        st.session_state.auto_refresh = True
        time.sleep(0.1)  # Pequeña pausa
        st.rerun()
    else:
        st.session_state.auto_refresh = False
    
    # Botón de refresh manual
    if st.button("🔄 Actualizar Ahora", key=get_unique_key("refresh_button")):
        st.experimental_rerun()
    
    # Información del cluster
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ Información del Cluster")
    
    if cluster_status['connected']:
        st.sidebar.success("✅ Conectado al cluster Ray")
        st.sidebar.metric(
            label="Nodos Activos",
            value=cluster_status['alive_node_count']
        )
    else:
        st.sidebar.error(f"❌ Cluster no conectado: {cluster_status.get('error', 'Error desconocido')}")
    
    # Hora de actualización
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")
    
    # Versión
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0 - Junio 2025")

# Renderizar cada pestaña usando las funciones de los módulos
with tabs[0]:
    render_overview_tab(cluster_status, system_metrics)

with tabs[1]:
    render_cluster_status_tab(cluster_status)

with tabs[2]:
    render_training_tab(cluster_status)

with tabs[3]:
    render_results_tab()

with tabs[4]:
    render_system_metrics_tab(system_metrics)

with tabs[5]:
    render_fault_tolerance_tab()

# Auto-refresh si está habilitado
if st.session_state.auto_refresh:
    time.sleep(30)  # Esperar 30 segundos
    st.rerun()