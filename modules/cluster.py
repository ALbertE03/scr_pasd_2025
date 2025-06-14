import streamlit as st
import ray
import os
import psutil
import pandas as pd
import plotly.graph_objects as go

@st.cache_data(ttl=30)
def get_cluster_status():
    """Obtiene el estado del cluster Ray"""
    try:
        if not ray.is_initialized():
            # Intentar conectar al cluster
            head_address = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
            ray.init(address=f"ray://{head_address}:10001", ignore_reinit_error=True)
        
        # Obtener información del cluster
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        # Analizar nodos vivos vs muertos
        alive_nodes = []
        dead_nodes = []
        
        for node in nodes:
            if node.get('Alive', False):
                alive_nodes.append(node)
            else:
                dead_nodes.append(node)
        
        return {
            "connected": True,
            "resources": cluster_resources,
            "nodes": nodes,
            "alive_nodes": alive_nodes,
            "dead_nodes": dead_nodes,
            "total_cpus": cluster_resources.get('CPU', 0),
            "total_memory": cluster_resources.get('memory', 0),
            "total_gpus": cluster_resources.get('GPU', 0),
            "node_count": len(nodes),
            "alive_node_count": len(alive_nodes),
            "dead_node_count": len(dead_nodes)
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "resources": {},
            "nodes": [],
            "alive_nodes": [],
            "dead_nodes": [],
            "total_cpus": 0,
            "total_memory": 0,
            "total_gpus": 0,
            "node_count": 0,
            "alive_node_count": 0,
            "dead_node_count": 0
        }

@st.cache_data(ttl=10)
def get_system_metrics():
    """Obtiene métricas del sistema"""
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
        st.error(f"Error obteniendo métricas del sistema: {e}")
        return {}

def plot_cluster_metrics(cluster_status):
    """Crea gráficos de métricas del cluster"""
    if not cluster_status['connected']:
        st.error("Cluster no conectado")
        return
    
    # Métricas de recursos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gráfico de CPUs
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cluster_status['total_cpus'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPUs Totales", 'font': {'size': 24, 'color': '#4361ee'}},
            gauge={
                'axis': {'range': [None, 20], 'tickwidth': 1},
                'bar': {'color': "#4361ee"},
                'steps': [
                    {'range': [0, 10], 'color': "rgba(67, 97, 238, 0.2)"},
                    {'range': [10, 20], 'color': "rgba(67, 97, 238, 0.4)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 16
                }
            }
        ))
        fig_cpu.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cpu, use_container_width=True, key="cluster_cpu_gauge")
    
    with col2:
        # Gráfico de memoria
        memory_gb = cluster_status['total_memory'] / (1024**3) if cluster_status['total_memory'] else 0
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=memory_gb,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memoria Total (GB)", 'font': {'size': 24, 'color': '#38b000'}},
            number={'suffix': " GB"},
            gauge={
                'axis': {'range': [None, 32], 'tickwidth': 1},
                'bar': {'color': "#38b000"},
                'steps': [
                    {'range': [0, 16], 'color': "rgba(56, 176, 0, 0.2)"},
                    {'range': [16, 32], 'color': "rgba(56, 176, 0, 0.4)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 24
                }
            }
        ))
        fig_mem.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_mem, use_container_width=True, key="cluster_memory_gauge")
    
    with col3:
        # Gráfico de estado de nodos
        fig_nodes = go.Figure(data=[
            go.Bar(
                x=['Vivos', 'Muertos'],
                y=[cluster_status['alive_node_count'], cluster_status['dead_node_count']],
                marker_color=['#38b000', '#ff0054'],
                text=[cluster_status['alive_node_count'], cluster_status['dead_node_count']],
                textposition='auto',
                hoverinfo='y+text',
                width=[0.4, 0.4]
            )
        ])
        fig_nodes.update_layout(
            title={
                'text': "Estado de Nodos",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#333'}
            },
            yaxis_title="Cantidad",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_nodes, use_container_width=True, key="cluster_nodes_status")

def render_cluster_status_tab(cluster_status):
    """Renderiza la pestaña de estado detallado del cluster"""
    st.header("Estado Detallado del Cluster")
    
    if cluster_status['connected']:
        # Información de nodos
        st.subheader("Estado de los Nodos del Cluster")
        
        # Resumen rápido
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodos", cluster_status['node_count'])
        with col2:
            st.metric("Nodos Vivos", cluster_status['alive_node_count'], 
                     delta="✅ Operativos")
        with col3:
            st.metric("Nodos Muertos", cluster_status['dead_node_count'],
                     delta="⚠️ Fuera de línea" if cluster_status['dead_node_count'] > 0 else "✅ Ninguno")
        
        # Tabla detallada de nodos
        st.subheader("Detalle de Nodos")
        
        nodes_data = []
        for i, node in enumerate(cluster_status['nodes']):
            is_alive = node.get('Alive', False)
            node_id = node.get('NodeID', f'node-{i+1}')
            
            nodes_data.append({
                'ID': node_id,
                'Estado': "🟢 Vivo" if is_alive else "🔴 Muerto",
                'Dirección': node.get('NodeManagerAddress', 'N/A'),
                'CPU': node.get('Resources', {}).get('CPU', 0),
                'Memoria (MB)': node.get('Resources', {}).get('memory', 0) / (1024*1024) if node.get('Resources', {}).get('memory') else 0,
                'GPU': node.get('Resources', {}).get('GPU', 0),
                'Última Actualización': node.get('Timestamp', 'N/A')
            })
        if nodes_data:
            df_nodes = pd.DataFrame(nodes_data)
            st.dataframe(df_nodes, use_container_width=True)
            
            # Mostrar alertas específicas
            if cluster_status['dead_node_count'] > 0:
                st.warning(f"⚠️ Hay {cluster_status['dead_node_count']} nodo(s) que no responden. Esto puede afectar el rendimiento del cluster.")
            # Recursos del cluster
            st.subheader("Recursos Totales del Cluster")
            resources_df = pd.DataFrame([cluster_status['resources']]).T
            resources_df.columns = ['Cantidad Total']
        
            st.dataframe(resources_df)
        else:
            st.error("No se puede conectar al cluster Ray")
            st.info("Asegúrate de que el cluster esté ejecutándose y accesible") 
    else:
        st.error("Cluster no conectado")
        st.warning(f"Error: {cluster_status.get('error', 'Desconocido')}")
        st.info("Por favor verifica que el cluster Ray esté en ejecución y sea accesible desde esta máquina.")
