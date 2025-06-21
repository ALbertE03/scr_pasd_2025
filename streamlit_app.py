"""
Ray ML Cluster Dashboard
"""

import streamlit as st
import time
from datetime import datetime

from modules.utils import initialize_session_state, load_custom_styles, get_unique_key
from modules.cluster import get_cluster_status, get_system_metrics, render_cluster_status_tab
from modules.views import (
    render_training_tab, 
    render_system_metrics_tab,
)
from modules.api_client import render_api_tab



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
    #"💻 Métricas del Sistema",
]

tabs = st.tabs(tab_titles)

cluster_status = get_cluster_status()
system_metrics = get_system_metrics()


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
    
    

with tabs[0]:
    render_cluster_status_tab(cluster_status,system_metrics)

with tabs[1]:
    render_training_tab(cluster_status)


with tabs[2]:
    render_api_tab()

#with tabs[3]:
 #   render_system_metrics_tab(system_metrics)

if st.session_state.auto_refresh:
    st.rerun()
    time.sleep(10)