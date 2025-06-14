import streamlit as st
import ray
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime, timedelta
import psutil
from train import DistributedMLTrainer, train_model_remote
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pickle
import requests

# Configuración de la página
st.set_page_config(
    page_title="Ray ML Cluster Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }    .stDataFrame {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

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

def load_training_results():
    """Carga los resultados de entrenamiento guardados"""
    results = {}
    datasets = ['iris', 'wine', 'breast_cancer']
    
    for dataset in datasets:
        filename = f"results_{dataset}.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    results[dataset] = json.load(f)
            except Exception as e:
                st.warning(f"Error cargando resultados de {dataset}: {e}")
    
    return results

def plot_model_comparison(results_data):
    """Crea gráfico de comparación de modelos"""
    if not results_data:
        st.warning("No hay datos de entrenamiento disponibles")
        return
    
    # Preparar datos para el gráfico
    data = []
    for dataset, models in results_data.items():
        for model_name, metrics in models.items():
            data.append({
                'Dataset': dataset,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std'],
                'Training_Time': metrics['training_time']
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        st.warning("No hay datos para mostrar")
        return
    
    # Gráfico de accuracy por modelo y dataset
    fig1 = px.bar(
        df, 
        x='Model', 
        y='Accuracy', 
        color='Dataset',
        title='Accuracy por Modelo y Dataset',
        height=400
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico de tiempo de entrenamiento
    fig2 = px.scatter(
        df, 
        x='Training_Time', 
        y='Accuracy', 
        color='Dataset',
        size='CV_Mean',
        hover_data=['Model'],
        title='Accuracy vs Tiempo de Entrenamiento',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

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
            title={'text': "CPUs Totales"},
            gauge={'axis': {'range': [None, 20]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 10], 'color': "lightgray"},
                       {'range': [10, 20], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 16}}))
        fig_cpu.update_layout(height=300)
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Gráfico de memoria
        memory_gb = cluster_status['total_memory'] / (1024**3) if cluster_status['total_memory'] else 0
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=memory_gb,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memoria Total (GB)"},
            gauge={'axis': {'range': [None, 32]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 16], 'color': "lightgray"},
                       {'range': [16, 32], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 24}}))
        fig_mem.update_layout(height=300)
        st.plotly_chart(fig_mem, use_container_width=True)
    
    with col3:
        # Gráfico de estado de nodos
        fig_nodes = go.Figure(data=[
            go.Bar(
                x=['Vivos', 'Muertos'],
                y=[cluster_status['alive_node_count'], cluster_status['dead_node_count']],
                marker_color=['green', 'red'],
                text=[cluster_status['alive_node_count'], cluster_status['dead_node_count']],
                textposition='auto',
            )
        ])
        fig_nodes.update_layout(
            title="Estado de Nodos",
            yaxis_title="Cantidad",
            height=300
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

def run_distributed_training(dataset_name, selected_models):
    """Ejecuta entrenamiento distribuido"""
    try:
        trainer = DistributedMLTrainer()
        
        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Iniciando entrenamiento distribuido...")
        progress_bar.progress(10)
        
        # Ejecutar entrenamiento
        results = trainer.train_models_distributed(
            dataset_name=dataset_name,
            selected_models=selected_models
        )
        
        progress_bar.progress(80)
        status_text.text("Guardando resultados...")
        
        # Guardar resultados
        trainer.save_results(f"results_{dataset_name}.json")
        trainer.save_models(f"models_{dataset_name}")
        
        progress_bar.progress(100)
        status_text.text("¡Entrenamiento completado!")
        
        return results
        
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        return None

def main():
    """Función principal de la aplicación Streamlit"""
    
    # Título principal
    st.title("🚀 Ray ML Cluster Dashboard")
    st.markdown("Dashboard avanzado para monitoreo y control del cluster Ray de Machine Learning")
    
    # Sidebar
    st.sidebar.title("Configuración")
    
    # Botón de actualización
    if st.sidebar.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Obtener estado del cluster
    cluster_status = get_cluster_status()
    system_metrics = get_system_metrics()
      # Estado de conexión
    if cluster_status['connected']:
        st.sidebar.markdown('<div class="success-card">✅ Cluster Conectado</div>', unsafe_allow_html=True)
        
    else:
        st.sidebar.markdown('<div class="warning-card">⚠️ Cluster Desconectado</div>', unsafe_allow_html=True)
        st.sidebar.error(f"Error: {cluster_status.get('error', 'Desconocido')}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🖥️ Cluster Status", 
        "🚀 Training", 
        "📈 Results", 
        "🔧 System Metrics"
    ])
    
    with tab1:
        st.header("Vista General del Cluster")
          # Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Nodos Vivos",
                value=cluster_status['alive_node_count'],
                delta="Online" if cluster_status['alive_node_count'] > 0 else "Ninguno"
            )
        
        with col2:
            # Mostrar nodos muertos con indicador visual
            dead_count = cluster_status['dead_node_count']
            delta_color = "inverse" if dead_count > 0 else "normal"
            st.metric(
                label="Nodos Muertos",
                value=dead_count,
                delta="⚠️ Atención" if dead_count > 0 else "✅ OK",
                delta_color=delta_color
            )
        
        with col3:
            st.metric(
                label="CPUs Totales",
                value=f"{cluster_status['total_cpus']:.0f}",
                delta=f"{system_metrics.get('cpu_percent', 0):.1f}% uso" if system_metrics else "N/A"
            )
        
        with col4:
            memory_gb = cluster_status['total_memory'] / (1024**3) if cluster_status['total_memory'] else 0
            st.metric(
                label="Memoria Total",
                value=f"{memory_gb:.1f} GB",
                delta=f"{system_metrics.get('memory_percent', 0):.1f}% uso" if system_metrics else "N/A"
            )
        
        with col5:
            st.metric(
                label="GPUs",
                value=cluster_status['total_gpus'],
                delta="Disponibles" if cluster_status['total_gpus'] > 0 else "No disponibles"
            )
          # Gráficos de métricas del cluster
        if cluster_status['connected']:
            # Alerta si hay nodos muertos
            if cluster_status['dead_node_count'] > 0:
                st.error(
                    f"⚠️ **ATENCIÓN**: {cluster_status['dead_node_count']} nodo(s) están muertos. "
                    f"Solo {cluster_status['alive_node_count']} de {cluster_status['node_count']} nodos están operativos."
                )
            else:
                st.success(f"✅ Todos los {cluster_status['alive_node_count']} nodos están operativos")
            
            plot_cluster_metrics(cluster_status)
    
    with tab2:
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
    
    with tab3:
        st.header("Entrenamiento Distribuido")
        
        if not cluster_status['connected']:
            st.warning("Debes conectarte al cluster para ejecutar entrenamientos")
            return
        
        # Configuración del entrenamiento
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.selectbox(
                "Selecciona Dataset",
                options=['iris', 'wine', 'breast_cancer'],
                index=0
            )
        
        with col2:
            available_models = [
                'RandomForest', 'GradientBoosting', 'AdaBoost', 'ExtraTrees',
                'LogisticRegression', 'SGD', 'SVM', 'KNN', 'DecisionTree', 'NaiveBayes'
            ]
            selected_models = st.multiselect(
                "Selecciona Modelos",
                options=available_models,
                default=['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
            )
        
        # Botón de entrenamiento
        if st.button("🚀 Iniciar Entrenamiento Distribuido", type="primary"):
            if selected_models:
                with st.container():
                    results = run_distributed_training(dataset_name, selected_models)
                    if results:
                        st.success("¡Entrenamiento completado exitosamente!")
                        
                        # Mostrar resultados rápidos
                        st.subheader("Resultados del Entrenamiento")
                        results_df = pd.DataFrame([
                            {
                                'Modelo': name,
                                'Accuracy': f"{data['accuracy']:.4f}",
                                'CV Score': f"{data['cv_mean']:.4f} ± {data['cv_std']:.4f}",
                                'Tiempo (s)': f"{data['training_time']:.2f}"
                            }
                            for name, data in results.items()
                        ])
                        st.dataframe(results_df, use_container_width=True)
            else:
                st.warning("Por favor selecciona al menos un modelo")
    
    with tab4:
        st.header("Resultados de Entrenamiento")
        
        # Cargar resultados existentes
        training_results = load_training_results()
        
        if training_results:
            # Selector de dataset
            dataset_selector = st.selectbox(
                "Ver resultados de:",
                options=list(training_results.keys()),
                key="results_dataset"
            )
            
            if dataset_selector and dataset_selector in training_results:
                dataset_results = training_results[dataset_selector]
                
                # Tabla de resultados
                st.subheader(f"Resultados para {dataset_selector.title()}")
                
                results_data = []
                for model_name, metrics in dataset_results.items():
                    results_data.append({
                        'Modelo': model_name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'CV Mean': f"{metrics['cv_mean']:.4f}",
                        'CV Std': f"{metrics['cv_std']:.4f}",
                        'Tiempo (s)': f"{metrics['training_time']:.2f}",
                        'Timestamp': metrics.get('timestamp', 'N/A')
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Gráficos de comparación
                st.subheader("Análisis Visual")
                plot_model_comparison({dataset_selector: dataset_results})
                
                # Matriz de confusión para el mejor modelo
                best_model = max(dataset_results.items(), key=lambda x: x[1]['accuracy'])
                st.subheader(f"Matriz de Confusión - {best_model[0]} (Mejor Modelo)")
                
                confusion_matrix = np.array(best_model[1]['confusion_matrix'])
                fig_conf = px.imshow(
                    confusion_matrix,
                    title=f"Matriz de Confusión - {best_model[0]}",
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No hay resultados de entrenamiento disponibles. Ejecuta un entrenamiento primero.")
    
    with tab5:
        st.header("Métricas del Sistema")
        
        if system_metrics:
            # Métricas en tiempo real
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="CPU",
                    value=f"{system_metrics['cpu_percent']:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Memoria",
                    value=f"{system_metrics['memory_percent']:.1f}%",
                    delta=f"{system_metrics['memory_available']:.1f}GB libre"
                )
            
            with col3:
                st.metric(
                    label="Disco",
                    value=f"{system_metrics['disk_percent']:.1f}%",
                    delta=f"{system_metrics['disk_free']:.1f}GB libre"
                )
            
            # Gráficos de uso de recursos
            fig_resources = make_subplots(
                rows=1, cols=3,
                subplot_titles=("CPU", "Memoria", "Disco"),
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # CPU gauge
            fig_resources.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_metrics['cpu_percent'],
                    title={'text': "CPU %"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
            
            # Memory gauge
            fig_resources.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_metrics['memory_percent'],
                    title={'text': "Memoria %"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 85}}
                ),
                row=1, col=2
            )
            
            # Disk gauge
            fig_resources.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_metrics['disk_percent'],
                    title={'text': "Disco %"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkorange"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 80}}
                ),
                row=1, col=3
            )
            
            fig_resources.update_layout(height=400)
            st.plotly_chart(fig_resources, use_container_width=True)
        
        else:
            st.error("No se pudieron obtener las métricas del sistema")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Ray ML Cluster Dashboard** | "
        f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()