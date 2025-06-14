import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os 
from .cluster import plot_cluster_metrics, render_cluster_status_tab
from .training import (
    plot_model_comparison, 
    plot_cross_dataset_comparison, 
    run_distributed_training,
    run_distributed_training_advanced,
    run_sequential_training, 
    load_training_results, 
    load_execution_summary,
    plot_training_metrics,
    plot_inference_metrics,
    load_training_history
)

def render_overview_tab(cluster_status, system_metrics):
    """Renderiza la pestaña de vista general"""
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

def render_training_tab(cluster_status):
    """Renderiza la pestaña de entrenamiento con capacidades avanzadas"""
    st.header("🧠 Entrenamiento Distribuido")
    
    if not cluster_status['connected']:
        st.warning("Debes conectarte al cluster para ejecutar entrenamientos")
        return
    
    # Crear tabs para diferentes opciones de entrenamiento
    training_tabs = st.tabs([
        "🚀 Entrenamiento Avanzado",
        "🔄 Entrenamiento Básico",
        "📊 Métricas en Tiempo Real"
    ])
    
    with training_tabs[0]:
        render_advanced_training(cluster_status)
        

        
    with training_tabs[2]:
        render_realtime_metrics()
        
def render_advanced_training(cluster_status):
    """Renderiza la interfaz de entrenamiento avanzado"""
    st.subheader("🚀 Entrenamiento Distribuido Avanzado")
    
    st.markdown("""
    <div class="success-card">
        <h4>✅ Procesamiento de múltiples modelos en paralelo</h4>
        <p>Entrene varios modelos simultáneamente aprovechando la potencia del cluster distribuido</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Configuración del dataset
        datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'diabetes']
        selected_dataset = st.selectbox(
            "Dataset",
            options=datasets,
            index=datasets.index(st.session_state.current_dataset) if st.session_state.current_dataset in datasets else 0,
            key="advanced_dataset_select"
        )
        st.session_state.current_dataset = selected_dataset
        
        # Configuración de modelos
        model_options = [
            "RandomForest", 
            "GradientBoosting", 
            "SVM", 
            "LogisticRegression",
            "KNN", 
            "DecisionTree", 
            "XGBoost"
        ]
        
        selected_models = st.multiselect(
            "Modelos a entrenar en paralelo",
            options=model_options,
            default=st.session_state.selected_models[:4] if hasattr(st.session_state, 'selected_models') else model_options[:4],
            key="advanced_models_multiselect"
        )
        
        st.session_state.selected_models = selected_models
        
    with col2:
        # Opciones avanzadas
        st.session_state.test_size = st.slider(
            "% Datos de prueba",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.test_size,
            step=0.05,
            key="advanced_test_size_slider"
        )
        
        enable_fault_tolerance = st.checkbox(
            "🛡️ Habilitar Tolerancia a Fallos",
            value=st.session_state.enable_fault_tolerance,
            key="advanced_fault_tolerance_checkbox"
        )
        
        visualize_progress = st.checkbox(
            "📊 Mostrar Progreso en Tiempo Real",
            value=True,
            key="advanced_show_progress_checkbox"
        )
         
    # Sección de Hiperparámetros
    with st.expander("⚙️ Configuración de Hiperparámetros"):
        st.caption("Configure hiperparámetros específicos para cada modelo seleccionado")
        
        # Mostramos configuración para cada modelo seleccionado
        hyperparams = {}
        
        for model in selected_models:
            st.subheader(f"{model}")
            cols = st.columns(3)
            
            if model == "RandomForest":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de árboles ({model})", 10, 200, 100, 10),
                    "max_depth": cols[1].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1),
                    "min_samples_split": cols[2].slider(f"Min muestras para split ({model})", 2, 10, 2, 1)
                }
            elif model == "GradientBoosting":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 10, 200, 100, 10),
                    "learning_rate": cols[1].slider(f"Tasa de aprendizaje ({model})", 0.01, 0.3, 0.1, 0.01),
                    "max_depth": cols[2].slider(f"Profundidad máxima ({model})", 2, 10, 3, 1)
                }
            elif model == "SVM":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1),
                    "kernel": cols[1].selectbox(f"Kernel ({model})", ["linear", "rbf", "poly"], 1),
                    "gamma": cols[2].selectbox(f"Gamma ({model})", ["scale", "auto"], 0)
                }
            elif model == "LogisticRegression":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 1000, 100, 100),
                    "solver": cols[2].selectbox(f"Solver ({model})", ["lbfgs", "liblinear", "newton-cg"], 0)
                }
            elif model == "NeuralNetwork":
                hyperparams[model] = {
                    "hidden_layers": cols[0].slider(f"Capas ocultas ({model})", 1, 5, 2, 1),
                    "neurons": cols[1].slider(f"Neuronas por capa ({model})", 5, 100, 32, 5),
                    "epochs": cols[2].slider(f"Épocas ({model})", 10, 200, 50, 10)
                }
            else:
                hyperparams[model] = {
                    "param1": cols[0].slider(f"Parámetro 1 ({model})", 0.0, 1.0, 0.5, 0.1),
                    "param2": cols[1].slider(f"Parámetro 2 ({model})", 0.0, 1.0, 0.5, 0.1),
                    "param3": cols[2].slider(f"Parámetro 3 ({model})", 0.0, 1.0, 0.5, 0.1)
                }
    
    # Sección de distribución de datos
    with st.expander("🔄 Distribución de Datos"):
        st.caption("Configure cómo se distribuirán los datos entre los nodos")
        
        data_distribution_strategy = st.radio(
            "Estrategia de distribución",
            ["Auto (Balanceo)", "Asignar por clases", "Fragmentos aleatorios", "Fragmentos estratificados"],
            horizontal=True
        )
        
        if data_distribution_strategy == "Asignar por clases":
            st.info("Los datos se distribuirán asignando clases completas a cada nodo para entrenar modelos especializados")
        elif data_distribution_strategy == "Fragmentos aleatorios":
            st.info("Los datos se particionarán aleatoriamente entre los nodos")
        elif data_distribution_strategy == "Fragmentos estratificados":
            st.info("Los datos se particionarán manteniendo la distribución de clases entre los nodos")
    
    # Botón de inicio de entrenamiento
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        start_training = st.button(
            "🚀 Iniciar Entrenamiento Distribuido", 
            type="primary",
            key="advanced_start_training_button",
            disabled=st.session_state.training_in_progress
        )
    
    # Área para resultados
    results_container = st.container()
    
    if start_training:
        with results_container:
            st.session_state.training_in_progress = True
            
            st.markdown("""
            <div class="dashboard-container">
                <h3>🔄 Entrenando Modelos en Paralelo</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Ejecutar entrenamiento avanzado
            results, training_history = run_distributed_training_advanced(
                dataset_name=selected_dataset,
                selected_models=selected_models,
                hyperparameters=hyperparams,
                enable_fault_tolerance=enable_fault_tolerance
            )
            
            st.session_state.training_in_progress = False
            
            if results and len(results) > 0:
                st.success(f"✅ Entrenamiento completado exitosamente para el dataset {selected_dataset}")
                
                # Actualizar resultados en session_state
                st.session_state.training_results = {selected_dataset: results}
                
                # Mostrar métricas de entrenamiento
                st.subheader("📊 Métricas de Entrenamiento")
                plot_training_metrics(training_history, chart_prefix="advanced")
                
                # Comparación de modelos
                st.subheader("🔍 Comparación de Modelos")
                plot_model_comparison({selected_dataset: results}, chart_prefix="advanced")
                
                # Guardar resultados
                st.session_state.last_trained_dataset = selected_dataset
                st.session_state.last_training_history = training_history

def render_realtime_metrics():
    """Renderiza métricas en tiempo real de modelos en producción"""
    st.subheader("📈 Métricas en Tiempo Real")
    
    st.markdown("""
    <div class="info-card">
        <h4>🔍 Monitoreo de Rendimiento</h4>
        <p>Visualice las estadísticas de inferencia en producción y métricas de rendimiento</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selección de dataset para mostrar resultados
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Si hay un último dataset entrenado, seleccionarlo por defecto
        default_dataset = st.session_state.get('last_trained_dataset', 'iris')
        datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'diabetes']
        
        selected_dataset = st.selectbox(
            "Dataset",
            options=datasets,
            index=datasets.index(default_dataset) if default_dataset in datasets else 0,
            key="metrics_dataset_select"
        )
    
    with col2:
        # Botón para refrescar datos
        refresh = st.button("🔄 Refrescar Datos", key="refresh_metrics_button")
    
    # Cargar datos de entrenamiento
    training_history = load_training_history(selected_dataset)
    
    # Crear tabs para diferentes tipos de métricas
    metrics_tabs = st.tabs([
        "📊 Métricas de Entrenamiento", 
        "⚡ Métricas de Inferencia"
    ])
    
    with metrics_tabs[0]:
        if training_history:
            # Mostrar métricas de entrenamiento
            plot_training_metrics(training_history, chart_prefix="realtime")
        else:
            st.info(f"No hay datos de entrenamiento disponibles para {selected_dataset}. Entrene primero usando la pestaña de Entrenamiento Avanzado.")
        with metrics_tabs[1]:
        # Cargar datos reales del dataset seleccionado
            results = load_training_results()
            
            if results and selected_dataset in results:
                # Solo pasamos los datos del dataset seleccionado
                dataset_results = {selected_dataset: results[selected_dataset]}
                plot_inference_metrics(dataset_results, chart_prefix="realtime")
                st.caption("Nota: Los datos de inferencia se basan en el rendimiento real de los modelos entrenados")
            else:
                # No pasamos datos, la función manejará el caso de no tener modelos
                plot_inference_metrics({}, chart_prefix="realtime")

def render_results_tab():
    """Renderiza la pestaña de resultados"""
    st.header("Resultados de Entrenamiento")
    
    # Tabs dentro de la pestaña de resultados
    results_tab1, results_tab2, results_tab3 = st.tabs([
        "📊 Resultados por Dataset", 
        "📈 Comparación de Datasets", 
        "📁 Modelos Guardados"
    ])
    
    # Tab 1: Resultados por dataset
    with results_tab1:
        # Carga de resultados existentes
        training_results = load_training_results()
        
        if not training_results:
            st.info("No hay resultados de entrenamiento disponibles. Ejecuta un entrenamiento primero.")
        else:
            # Selector de dataset para resultados
            dataset_options = list(training_results.keys())
            if dataset_options:
                selected_dataset = st.selectbox(
                    "Selecciona dataset para ver resultados",
                    options=dataset_options,
                    index=0,
                    key="results_dataset_select"
                )
                
                # Mostrar resultados del dataset seleccionado
                dataset_results = training_results.get(selected_dataset, {})
                if dataset_results:
                    st.subheader(f"Resultados para dataset: {selected_dataset}")
                    
                    # Tabla de resultados
                    results_data = []
                    for model_name, metrics in dataset_results.items():
                        results_data.append({
                            'Modelo': model_name,
                            'Accuracy': f"{metrics['accuracy']:.4f}",
                            'CV (Media)': f"{metrics['cv_mean']:.4f}",
                            'CV (Std)': f"{metrics['cv_std']:.4f}",
                            'Tiempo (s)': f"{metrics['training_time']:.3f}"
                        })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualización gráfica
                        st.subheader("Visualización")
                        
                        # Gráficos específicos del dataset
                        plot_model_comparison({selected_dataset: dataset_results}, chart_prefix="single_dataset")
                    else:
                        st.warning("No hay datos de modelos para este dataset")
                else:
                    st.warning("No hay resultados disponibles para este dataset")
    
    # Tab 2: Comparación entre datasets
    with results_tab2:
        all_results = load_training_results()
        execution_summary = load_execution_summary()
        
        if not all_results:
            st.info("No hay resultados de entrenamiento disponibles para comparar datasets.")
        else:
            st.subheader("Comparación entre Datasets")
            
            # Mostrar resumen de ejecución si está disponible
            if execution_summary:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Datasets Procesados", 
                        execution_summary.get('total_datasets', 0)
                    )
                
                with col2:
                    st.metric(
                        "Datasets Exitosos", 
                        execution_summary.get('successful_datasets', 0),
                        delta=f"{execution_summary.get('success_rate', 0):.1f}% éxito"
                    )
                
                with col3:
                    st.metric(
                        "Tiempo Total (s)", 
                        f"{execution_summary.get('total_execution_time', 0):.2f}"
                    )
            
            # Gráficos de comparación entre datasets
            plot_cross_dataset_comparison(all_results)
    
    # Tab 3: Modelos guardados
    with results_tab3:
        st.subheader("Modelos Guardados")
        
        # Buscar modelos guardados
        model_files = []
        model_dirs = ['models_iris', 'models_wine', 'models_breast_cancer']
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for model_file in os.listdir(model_dir):
                    if model_file.endswith('.pkl'):
                        model_files.append({
                            'Dataset': model_dir.replace('models_', ''),
                            'Modelo': model_file,
                            'Tamaño': f"{os.path.getsize(os.path.join(model_dir, model_file)) / 1024:.2f} KB",
                            'Modificado': datetime.fromtimestamp(os.path.getmtime(os.path.join(model_dir, model_file))).strftime('%Y-%m-%d %H:%M:%S')
                        })
        
        if model_files:
            models_df = pd.DataFrame(model_files)
            st.dataframe(models_df, use_container_width=True)
            
            # Opción para descargar los modelos (simulada)
            if st.button("📥 Exportar Modelos Seleccionados", key="export_models_btn"):
                st.success("Modelos exportados correctamente")
                
                # Mostrar código de ejemplo de cómo cargar los modelos
                st.subheader("Código para cargar modelos:")
                st.code("""
                import pickle

                # Cargar modelo guardado
                with open('models_iris/RandomForest.pkl', 'rb') as f:
                    model = pickle.load(f)
                
                # Realizar predicción
                predictions = model.predict(X_test)
                """, language="python")
        else:
            st.info("No hay modelos guardados disponibles. Entrena y guarda modelos primero.")

def render_system_metrics_tab(system_metrics):
    """Renderiza la pestaña de métricas del sistema"""
    st.header("Métricas del Sistema")
    
    # Métricas principales del sistema host
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=system_metrics.get('cpu_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#4361ee"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(67, 97, 238, 0.2)"},
                    {'range': [50, 80], 'color': "rgba(67, 97, 238, 0.4)"},
                    {'range': [80, 100], 'color': "rgba(67, 97, 238, 0.6)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=system_metrics.get('memory_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memoria RAM (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#38b000"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(56, 176, 0, 0.2)"},
                    {'range': [50, 80], 'color': "rgba(56, 176, 0, 0.4)"},
                    {'range': [80, 100], 'color': "rgba(56, 176, 0, 0.6)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=system_metrics.get('disk_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Disco (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#9e0059"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(158, 0, 89, 0.2)"},
                    {'range': [50, 80], 'color': "rgba(158, 0, 89, 0.4)"},
                    {'range': [80, 100], 'color': "rgba(158, 0, 89, 0.6)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalles adicionales
    st.subheader("Detalles del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Memoria**")
        
        mem_data = {
            "Métrica": ["Total", "Disponible", "Usado"],
            "Valor (GB)": [
                system_metrics.get('memory_total', 0),
                system_metrics.get('memory_available', 0),
                system_metrics.get('memory_total', 0) - system_metrics.get('memory_available', 0)
            ]
        }
        mem_df = pd.DataFrame(mem_data)
        st.dataframe(mem_df, use_container_width=True)
    
    with col2:
        st.markdown("**Almacenamiento**")
        
        disk_data = {
            "Métrica": ["Total", "Libre", "Usado"],
            "Valor (GB)": [
                system_metrics.get('disk_total', 0),
                system_metrics.get('disk_free', 0),
                system_metrics.get('disk_total', 0) - system_metrics.get('disk_free', 0)
            ]
        }
        disk_df = pd.DataFrame(disk_data)
        st.dataframe(disk_df, use_container_width=True)
    
    # Métricas históricas (simuladas para este ejemplo)
    st.subheader("Métricas Históricas")
    
    # Simulación de datos históricos
    import numpy as np
    timestamps = [f"{i}:00" for i in range(9, 21)]  # 9 AM - 8 PM
    cpu_history = np.clip(system_metrics.get('cpu_percent', 50) + np.random.normal(0, 10, len(timestamps)), 0, 100)
    memory_history = np.clip(system_metrics.get('memory_percent', 40) + np.random.normal(0, 5, len(timestamps)), 0, 100)
    
    # Gráfico de líneas para datos históricos
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_history,
        mode='lines+markers',
        name='CPU (%)',
        line=dict(width=3, color='#4361ee'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_history,
        mode='lines+markers',
        name='Memoria (%)',
        line=dict(width=3, color='#38b000'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Histórico de Utilización (Hoy)',
        xaxis_title='Hora',
        yaxis_title='Utilización (%)',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, 100]
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_fault_tolerance_tab():
    """Renderiza la pestaña de tolerancia a fallos"""
    from .training import get_fault_tolerance_stats
    import os
    from datetime import datetime
    
    st.header("Monitoreo de Tolerancia a Fallos")
    
    # Obtener estadísticas de tolerancia a fallos
    fault_stats = get_fault_tolerance_stats()
    
    # Métricas de salud principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Valores para métricas (usar datos reales si están disponibles)
    failed_tasks = fault_stats.get('failed_tasks', 0) if fault_stats else 0
    recovered_tasks = fault_stats.get('recovered_tasks', 0) if fault_stats else 0
    cluster_nodes = fault_stats.get('total_nodes', 4) if fault_stats else 4  # Default a 4 si no hay datos
    alive_nodes = fault_stats.get('alive_nodes', cluster_nodes) if fault_stats else cluster_nodes
    
    with col1:
        st.metric(
            "Tareas Fallidas", 
            failed_tasks,
            delta="require atención" if failed_tasks > 0 else "todo ok"
        )
    
    with col2:
        recover_rate = (recovered_tasks / max(failed_tasks, 1)) * 100 if failed_tasks > 0 else 100
        st.metric(
            "Tareas Recuperadas", 
            recovered_tasks,
            delta=f"{recover_rate:.1f}% tasa de recuperación"
        )
    
    with col3:
        st.metric(
            "Nodos Vivos", 
            alive_nodes,
            delta=f"de {cluster_nodes} total"
        )
    
    with col4:
        uptime_percentage = (alive_nodes / max(cluster_nodes, 1)) * 100 if cluster_nodes > 0 else 0
        delta_color = "normal" if uptime_percentage >= 80 else "inverse"
        st.metric(
            "Uptime (%)", 
            f"{uptime_percentage:.1f}%",
            delta="Saludable" if uptime_percentage >= 80 else "Atención requerida",
            delta_color=delta_color
        )
    
    # Dashboard de salud del cluster
    st.subheader("🏥 Salud del Cluster")
    
    # Indicador visual de salud
    health_score = min(100, (alive_nodes / max(cluster_nodes, 1)) * 100 - (failed_tasks * 5))
    health_color = "green" if health_score >= 80 else "orange" if health_score >= 60 else "red"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Salud del Cluster"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': health_color},
                'steps': [
                    {'range': [0, 60], 'color': "lightcoral"},
                    {'range': [60, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 80
                }
            }
        ))                
        fig_health.update_layout(
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_health, use_container_width=True, key="fault_tolerance_health_gauge")
    
    with col2:
        # Log de eventos de tolerancia a fallos
        st.markdown("**📝 Log de Eventos Recientes**")
        
        # Simular algunos logs si no hay datos reales
        if not st.session_state.fault_logs:
            sample_logs = [
                {"time": "2025-06-14 10:30:15", "event": "Sistema iniciado", "type": "info"},
                {"time": "2025-06-14 10:32:20", "event": "Nodo ray-worker-1 conectado", "type": "success"},
                {"time": "2025-06-14 10:35:45", "event": "Reintento automático en RandomForest", "type": "warning"},
            ]
            st.session_state.fault_logs = sample_logs
        
        for log in st.session_state.fault_logs[-5:]:  # Mostrar últimos 5
            icon = "ℹ️" if log["type"] == "info" else "✅" if log["type"] == "success" else "⚠️"
            st.text(f"{icon} {log['time']}: {log['event']}")
    
    # Detalles de tareas fallidas
    if fault_stats and fault_stats.get('failed_task_details'):
        st.subheader("❌ Análisis de Fallos Detallado")
        
        # Tabs para diferentes vistas de fallos
        fault_tab1, fault_tab2, fault_tab3 = st.tabs(["📋 Lista de Fallos", "📊 Análisis", "🔍 Diagnóstico"])
        
        with fault_tab1:
            failed_tasks_data = []
            
            for failed_task in fault_stats['failed_task_details']:
                failed_tasks_data.append({
                    'Modelo': failed_task.get('model_name', 'N/A'),
                    'Nodo': failed_task.get('node_id', 'N/A'),
                    'Error': failed_task.get('error', 'Error desconocido')[:50] + "...",
                    'Timestamp': failed_task.get('timestamp', 'N/A'),
                    'Estado': failed_task.get('status', 'failed')
                })
            
            if failed_tasks_data:
                failed_df = pd.DataFrame(failed_tasks_data)
                st.dataframe(failed_df, use_container_width=True)
                  # Botón para exportar fallos
                if st.button("📥 Exportar Log de Fallos", key="fault_export_failures_btn"):
                    st.success("Log de fallos exportado a fault_log.json")
        
        with fault_tab2:
            if failed_tasks_data:
                # Gráfico de fallos por modelo
                failed_df = pd.DataFrame(failed_tasks_data)
                fail_counts = failed_df['Modelo'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_fails = px.bar(
                        x=fail_counts.index,
                        y=fail_counts.values,
                        title="Fallos por Modelo",
                        labels={'x': 'Modelo', 'y': 'Número de Fallos'},
                        color=fail_counts.values,
                        color_continuous_scale="Reds"
                    )
                    fig_fails.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_fails, use_container_width=True, key="fault_analysis_fails_by_model")
                
                with col2:
                    # Fallos por nodo
                    node_counts = failed_df['Nodo'].value_counts()                            
                    fig_nodes_fault = px.pie(
                        values=node_counts.values,
                        names=node_counts.index,
                        title="Distribución de Fallos por Nodo"
                    )
                    fig_nodes_fault.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_nodes_fault, use_container_width=True, key="fault_analysis_fails_by_node_pie")
        
        with fault_tab3:
            st.markdown("**🔍 Diagnóstico Automático**")
            
            # Análisis automático de patrones de fallo
            if failed_tasks_data:
                failed_df = pd.DataFrame(failed_tasks_data)
                total_failures = len(failed_tasks_data)
                most_failing_model = failed_df['Modelo'].mode().iloc[0] if not failed_df.empty else "N/A"
                most_failing_node = failed_df['Nodo'].mode().iloc[0] if not failed_df.empty else "N/A"
                
                st.info(f"**Análisis Automático:**")
                st.write(f"• Total de fallos detectados: {total_failures}")
                st.write(f"• Modelo con más fallos: {most_failing_model}")
                st.write(f"• Nodo con más fallos: {most_failing_node}")
                
                # Recomendaciones automáticas
                st.markdown("**💡 Recomendaciones:**")
                if total_failures > 10:
                    st.warning("- Considerar aumentar el timeout de entrenamiento")
                    st.warning("- Verificar la estabilidad de la red del cluster")
                if most_failing_node != "N/A":
                    st.warning(f"- Revisar la conectividad del nodo {most_failing_node}")
                if most_failing_model != "N/A":
                    st.info(f"- Optimizar parámetros del modelo {most_failing_model}")
    else:
        st.success("✅ No hay tareas fallidas registradas - Sistema funcionando correctamente")
    
    # Configuración avanzada de tolerancia a fallos
    st.subheader("⚙️ Configuración de Tolerancia a Fallos")        
    config_tab1, config_tab2, config_tab3 = st.tabs(["🔧 Básico", "🚀 Avanzado", "📊 Monitoreo"])
    
    with config_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.fault_config['max_retries'] = st.slider(
                "Máximo de reintentos",
                min_value=1,
                max_value=10,
                value=st.session_state.fault_config.get('max_retries', 3),
                key="fault_max_retries_slider"
            )
            
            st.session_state.fault_config['timeout_seconds'] = st.slider(
                "Timeout de tarea (segundos)",
                min_value=30,
                max_value=600,
                value=st.session_state.fault_config.get('timeout_seconds', 300),
                step=30,
                key="fault_timeout_slider"
            )
        
        with col2:
            st.session_state.fault_config['enable_reconstruction'] = st.checkbox(
                "Habilitar reconstrucción de objetos",
                value=st.session_state.fault_config.get('enable_reconstruction', True),
                key="fault_reconstruction_cb"
            )
            
            st.session_state.fault_config['enable_auto_retry'] = st.checkbox(
                "Reintento automático",
                value=st.session_state.fault_config.get('enable_auto_retry', True),
                key="fault_auto_retry_cb"
            )
    
    with config_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Intervalo de monitoreo (segundos)",
                min_value=5,
                max_value=120,
                value=st.session_state.fault_config.get('monitoring_interval', 30),
                key="fault_monitoring_interval"
            )
            
            st.text_input(
                "Emails para notificaciones",
                value=st.session_state.fault_config.get('notification_emails', ''),
                key="fault_notification_emails"
            )
        
        with col2:
            st.selectbox(
                "Estrategia de reintento",
                options=["Inmediato", "Backoff exponencial", "Intervalo fijo", "Aleatorio"],
                index=1,
                key="fault_retry_strategy"
            )
            
            st.selectbox(
                "Nivel de logging",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=1,
                key="fault_log_level"
            )
    
    with config_tab3:
        # Activar/desactivar monitoreo
        st.session_state.monitoring_active = st.toggle(
            "Activar monitoreo en vivo",
            value=st.session_state.monitoring_active,
            key="fault_monitoring_toggle"
        )
        
        if st.session_state.monitoring_active:
            st.info("Monitoreo activo - registrando eventos del cluster")
            
            # Simular algo de actividad de monitoreo
            st.markdown("**📊 Eventos en tiempo real**")
            
            # Placeholder para gráfico de eventos en tiempo real
            import random
            import numpy as np
            
            # Simulación de datos de monitoreo
            timestamps = [f"{i}" for i in range(1, 11)]  # 1-10 periodos
            events = np.random.randint(0, 5, size=10)  # Eventos aleatorios
            
            # Gráfico de eventos
            fig = px.bar(
                x=timestamps,
                y=events,
                title="Eventos por Periodo de Monitoreo",
                labels={'x': 'Periodo', 'y': 'Número de Eventos'}
            )
            fig.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_training_metrics(training_history, chart_prefix=""):
    """Visualiza métricas de rendimiento de los modelos"""
    if not training_history or not isinstance(training_history, dict):
        st.warning("No hay datos de historial de entrenamiento disponibles")
        return
    
    # Preparamos los datos para gráficas de barras comparativas
    metrics_data = []
    
    for model_name, history in training_history.items():
        if not history or not isinstance(history, dict):
            continue
        
        # Solo procesar si hay métricas disponibles
        if 'accuracy' not in history or history['accuracy'] is None:
            continue
            
        # Crear un registro para este modelo
        entry = {
            'Model': model_name,
            'Accuracy': float(history.get('accuracy', 0)),
            'Val_Accuracy': float(history.get('val_accuracy', 0)),
            'Loss': float(history.get('loss', 0)),
            'Val_Loss': float(history.get('val_loss', 0))
        }
        
        # Normalizar valores de accuracy
        if entry['Accuracy'] > 1:
            entry['Accuracy'] = min(1.0, entry['Accuracy'] / 100)
        if entry['Val_Accuracy'] > 1:
            entry['Val_Accuracy'] = min(1.0, entry['Val_Accuracy'] / 100)
            
        metrics_data.append(entry)
    
    if not metrics_data:
        st.warning("No hay suficientes datos de historial para visualizar")
        return
        
    df = pd.DataFrame(metrics_data)
    
    # Mostrar información del DataFrame para depuración
    with st.expander("Información del DataFrame generado", expanded=False):
        st.write("Columnas disponibles:", df.columns.tolist())
        st.write("Número de filas:", len(df))
        st.write("Valores únicos en 'Model':", df['Model'].unique().tolist())
        st.write("Datos completos:")
        st.write(df)
    
    # Crear columnas para los gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de precisión (barras)
        fig_accuracy = px.bar(
            df,
            x="Model",
            y=["Accuracy", "Val_Accuracy"],
            title="Comparación de Precisión entre Modelos",
            labels={"value": "Precisión", "variable": "Tipo"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Configurar aspecto
        fig_accuracy.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Modelo"),
            yaxis=dict(title="Precisión", tickformat=".0%", range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_gridcolor="rgba(0,0,0,0.1)",
            yaxis_gridcolor="rgba(0,0,0,0.1)",
            height=400,
            margin=dict(l=20, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig_accuracy, use_container_width=True, key=f"{chart_prefix}_accuracy_plot")
    
    with col2:
        # Gráfico de pérdida (barras)
        fig_loss = px.bar(
            df,
            x="Model",
            y=["Loss", "Val_Loss"],
            title="Comparación de Pérdida entre Modelos",
            labels={"value": "Pérdida", "variable": "Tipo"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Calcular el valor máximo para el eje y
        loss_max = df[["Loss", "Val_Loss"]].values.max() * 1.2 if len(df) > 0 else 2.0
        
        # Configurar aspecto
        fig_loss.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Modelo"),
            yaxis=dict(title="Pérdida", range=[0, loss_max]),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_gridcolor="rgba(0,0,0,0.1)",
            yaxis_gridcolor="rgba(0,0,0,0.1)",
            height=400,
            margin=dict(l=20, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig_loss, use_container_width=True, key=f"{chart_prefix}_loss_plot")
