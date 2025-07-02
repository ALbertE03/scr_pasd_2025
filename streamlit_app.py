"""
Ray ML Cluster Dashboard
"""

import streamlit as st
import time
from modules.utils import initialize_session_state, load_custom_styles, get_unique_key
from modules.cluster import render_cluster_status_tab
from modules.views import (
    render_training_tab,
)
from modules.api_client import render_api_tab,APIClient
import time

time.sleep(5)


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


load_custom_styles()


initialize_session_state()

st.markdown("""
<div class="dashboard-header">
    <h1>🚀 Ray ML Cluster Dashboard</h1>
    <h3>Sistema de Entrenamiento ML Distribuido</h3>
</div>
""", unsafe_allow_html=True)


tab_titles = [
    "🔍 Estado del Cluster",
    "🧠 Entrenamiento ML",
    "🌐 API de Modelos",
]

api_client = APIClient()


cluster_status = api_client.get_cluster_status()
response = api_client.get_system_metrics()

if response["status"] == "success":
    system_metrics = response["data"]
else:
    system_metrics = {
        "cpu_percent": 0,
        "memory_percent": 0,
        "memory_available": 0,
            "memory_total": 0,
            "disk_percent": 0,
            "disk_free": 0,
            "disk_total": 0
        }

tabs = st.tabs(tab_titles) 
with st.sidebar:
    st.title("⚙️ Configuración")
    
    auto_refresh = st.toggle(
        "Auto-Refresh (10s)",
        value=st.session_state.auto_refresh,
        key="auto_refresh_toggle"
    )
    
    if auto_refresh:
        st.session_state.auto_refresh = True
        time.sleep(0.1)  
        st.rerun()
    else:
        st.session_state.auto_refresh = False
    
    if st.button("🔄 Actualizar Ahora", key=get_unique_key("refresh_button")):
        st.experimental_rerun()
    
    
try:
    with tabs[0]:
        render_cluster_status_tab(cluster_status['data'],system_metrics,api_client)
except Exception as e:
    st.error(f"Error en pestaña de Cluster: {e}")

try:
    with tabs[1]:
        render_training_tab(cluster_status["data"],api_client)
except Exception as e:
    st.error(f"Error en pestaña de Entrenamiento: {e}")
    st.write("Detalles del error:")
    st.exception(e)

try:
    with tabs[2]:
        render_api_tab(api_client)
except Exception as e:
    st.error(f"Error en pestaña de API: {e}")
#with tabs[3]:
 #   render_system_metrics_tab(system_metrics)

if st.session_state.auto_refresh:
    st.rerun()
    time.sleep(10)