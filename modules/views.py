import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .cluster import  get_system_metrics
from .training import (
    plot_model_comparison, 
    run_distributed_training_advanced,
    plot_training_metrics,

)
from .utils import save_system_metrics_history, get_metrics_for_timeframe


def render_training_tab(cluster_status):
    """Renderiza la pestaña de entrenamiento con capacidades avanzadas"""
    st.header("🧠 Entrenamiento Distribuido")
    
    if not cluster_status['connected']:
        st.warning("Debes conectarte al cluster para ejecutar entrenamientos")
        return
    
    training_tabs = st.tabs([
        "🚀 Entrenamiento Avanzado",
    ])
    
    with training_tabs[0]:
        render_advanced_training(cluster_status)
        

        
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
        datasets = ['iris', 'wine', 'breast_cancer', 'digits']
        selected_dataset = st.selectbox(
            "Dataset",
            options=datasets,
            index=datasets.index(st.session_state.current_dataset) if st.session_state.current_dataset in datasets else 0,
            key="advanced_dataset_select"
        )
        st.session_state.current_dataset = selected_dataset
        
        model_options = [
            # Modelos basados en árboles
            "RandomForest", 
            "GradientBoosting", 
            "AdaBoost",
            "ExtraTrees",
            "DecisionTree", 
            "XGBoost",
            
            # Modelos lineales
            "LogisticRegression",
            "SGD",
            "PassiveAggressive",
            
            # Basados en vecinos
            "KNN",
            
            # SVM
            "SVM",
            "LinearSVM",
            
            # Naive Bayes
            "GaussianNB",
            "BernoulliNB",
            "MultinomialNB",
            "ComplementNB",
            
            # Discriminant Analysis
            "LDA",
            "QDA",
            
            # Neural Networks
            "MLP",
            
            # Ensembles
            "Bagging",
            "Voting"
        ]
        
        selected_models = st.multiselect(
            "Modelos a entrenar en paralelo",
            options=model_options,
            default=st.session_state.selected_models[:4] if hasattr(st.session_state, 'selected_models') else model_options[:4],
            key="advanced_models_multiselect"
        )
        
        st.session_state.selected_models = selected_models
        
    with col2:
        st.session_state.test_size = st.slider(
            "% Datos de prueba",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.test_size,
            step=0.05,
            key="advanced_test_size_slider"
        )
        
         
    with st.expander("⚙️ Configuración de Hiperparámetros"):
        st.caption("Configure hiperparámetros específicos para cada modelo seleccionado")
        
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
            elif model == "AdaBoost":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 50, 200, 100, 10),
                    "learning_rate": cols[1].slider(f"Tasa de aprendizaje ({model})", 0.01, 1.0, 0.1, 0.01),
                    "algorithm": cols[2].selectbox(f"Algoritmo ({model})", ["SAMME"], 0)
                }
            elif model == "ExtraTrees":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de árboles ({model})", 10, 200, 100, 10),
                    "max_depth": cols[1].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1),
                    "min_samples_split": cols[2].slider(f"Min muestras para split ({model})", 2, 10, 2, 1)
                }
            elif model == "KNN":
                hyperparams[model] = {
                    "n_neighbors": cols[0].slider(f"Número de vecinos ({model})", 1, 20, 5, 1),
                    "weights": cols[1].selectbox(f"Pesos ({model})", ["uniform", "distance"], 0),
                    "algorithm": cols[2].selectbox(f"Algoritmo ({model})", ["auto", "ball_tree", "kd_tree", "brute"], 0)
                }
            elif model == "DecisionTree":
                hyperparams[model] = {
                    "max_depth": cols[0].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1),
                    "min_samples_split": cols[1].slider(f"Min muestras para split ({model})", 2, 10, 2, 1),
                    "criterion": cols[2].selectbox(f"Criterio ({model})", ["gini", "entropy"], 0)
                }
            elif model == "SGD":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.0001, 0.1, 0.0001, 0.0001, format="%.4f"),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 2000, 1000, 100),
                    "loss": cols[2].selectbox(f"Función de pérdida ({model})", ["hinge", "log_loss", "modified_huber"], 1)
                }           
            elif model == "PassiveAggressive":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 1000, 1000, 100),
                    "tol": cols[2].slider(f"Tolerancia ({model})", 1e-5, 1e-2, 1e-3, format="%.5f")
                }
            elif model == "LinearSVM":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 1000, 1000, 100),
                    "tol": cols[2].slider(f"Tolerancia ({model})", 1e-5, 1e-2, 1e-3, format="%.5f")
                }
            elif model == "GaussianNB":
                hyperparams[model] = {
                    "var_smoothing": cols[0].slider(f"Suavizado de varianza ({model})", 1e-12, 1e-6, 1e-9, format="%.2e")
                }
            elif model == "BernoulliNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True)
                }
            elif model == "MultinomialNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True)
                }
            elif model == "ComplementNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True)
                }
            elif model == "LDA":
                hyperparams[model] = {
                    "solver": cols[0].selectbox(f"Solver ({model})", ["svd", "lsqr", "eigen"], 0),
                    "shrinkage": cols[1].slider(f"Shrinkage ({model})", None, 1.0, None) if cols[1].checkbox(f"Usar shrinkage ({model})", False) else None
                }
            elif model == "QDA":
                hyperparams[model] = {
                    "reg_param": cols[0].slider(f"Parámetro de regularización ({model})", 0.0, 1.0, 0.0, 0.01)
                }
            elif model == "MLP":
                hyperparams[model] = {
                    "hidden_layer_sizes": (cols[0].slider(f"Neuronas capa oculta ({model})", 10, 200, 100, 10),),
                    "activation": cols[1].selectbox(f"Activación ({model})", ["relu", "tanh", "logistic"], 0),
                    "max_iter": cols[2].slider(f"Max iteraciones ({model})", 100, 500, 200, 50)
                }
            elif model == "Bagging":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 5, 50, 10, 5),
                    "max_samples": cols[1].slider(f"Max muestras ({model})", 0.1, 1.0, 1.0, 0.1),
                    "bootstrap": cols[2].checkbox(f"Bootstrap ({model})", True)
                }
            elif model == "Voting":
                hyperparams[model] = {
                    "voting": cols[0].selectbox(f"Tipo de votación ({model})", ["hard", "soft"], 1)
                }
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        start_training = st.button(
            "🚀 Iniciar Entrenamiento Distribuido", 
            type="primary",
            key="advanced_start_training_button",
        )
    
    results_container = st.container()
    
    if start_training:
        with results_container:
            st.session_state.training_in_progress = True
            
            st.markdown("""
            <div class="dashboard-container">
                <h3>🔄 Entrenando Modelos en Paralelo</h3>
            </div>
            """, unsafe_allow_html=True)
            
            results, training_history = run_distributed_training_advanced(
                dataset_name=selected_dataset,
                selected_models=selected_models,
                hyperparameters=hyperparams
            )
            
            st.session_state.training_in_progress = False
            
            if results and len(results) > 0:
                st.success(f"✅ Entrenamiento completado exitosamente para el dataset {selected_dataset}")

                st.session_state.training_results = {selected_dataset: results}

                st.subheader("📊 Métricas de Entrenamiento")
                plot_training_metrics(training_history, chart_prefix="advanced")

                st.subheader("🔍 Comparación de Modelos")
                plot_model_comparison({selected_dataset: results}, chart_prefix="advanced")
                
                st.session_state.last_trained_dataset = selected_dataset
                st.session_state.last_training_history = training_history

def render_system_metrics_tab(system_metrics):
    """Renderiza la pestaña de métricas del sistema"""
    st.header("Métricas del Sistema")
    
    # Inicializar recolección de métricas si es la primera vez
    initialize_metrics_collection()
    
    # Botón para refrescar métricas
    col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
    with col_refresh2:
        if st.button("🔄 Refrescar Métricas", key="refresh_system_metrics"):
            # Obtener métricas frescas del sistema
            fresh_metrics = get_system_metrics()
            if fresh_metrics:
                system_metrics.update(fresh_metrics)
                # Guardar en historial
                save_system_metrics_history(fresh_metrics)
                st.success("Métricas actualizadas")
                st.rerun()  # Refrescar la interfaz
    
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
    
    st.subheader("Métricas Históricas")
    
    # Obtener datos históricos reales
    historical_data = get_metrics_for_timeframe(hours=12)
    
    # Solo mostrar gráfico si hay datos históricos reales
    if historical_data['timestamps'] and len(historical_data['timestamps']) >= 2:
        timestamps = historical_data['timestamps']
        cpu_history = historical_data['cpu_values']
        memory_history = historical_data['memory_values']
        disk_history = historical_data['disk_values']

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
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=disk_history,
            mode='lines+markers',
            name='Disco (%)',
            line=dict(width=3, color='#9e0059'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Histórico de Utilización (Últimas 12 horas)',
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
        
        # Mostrar estadísticas de los datos históricos reales
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            if len(cpu_history) > 0:
                st.metric(
                    "CPU Promedio (12h)", 
                    f"{sum(cpu_history) / len(cpu_history):.1f}%",
                    f"{cpu_history[-1] - cpu_history[0]:.1f}%" if len(cpu_history) > 1 else None
                )
        
        with col_stats2:
            if len(memory_history) > 0:
                st.metric(
                    "Memoria Promedio (12h)", 
                    f"{sum(memory_history) / len(memory_history):.1f}%",
                    f"{memory_history[-1] - memory_history[0]:.1f}%" if len(memory_history) > 1 else None
                )
        
        with col_stats3:
            if len(disk_history) > 0:
                st.metric(
                    "Disco Promedio (12h)", 
                    f"{sum(disk_history) / len(disk_history):.1f}%",
                    f"{disk_history[-1] - disk_history[0]:.1f}%" if len(disk_history) > 1 else None
                )
                
        # Información adicional sobre la recolección de métricas
        with st.expander("ℹ️ Información sobre Métricas Históricas"):
            st.markdown("""
            **🔍 Fuente de Datos:**
            - Las métricas se obtienen en tiempo real usando `psutil`
            - Los datos históricos se almacenan localmente en `system_metrics_history.json`
            - Se mantienen datos de las últimas 24 horas
            
            **📊 Frecuencia de Actualización:**
            - Use el botón "Refrescar Métricas" para obtener datos actuales
            - Los datos históricos se acumulan automáticamente con cada actualización
            
            **📈 Datos Históricos Disponibles:**
            - Entradas registradas: {entries}
            - Período mostrado: Últimas 12 horas
            """.format(entries=len(historical_data['timestamps'])))
    else:
        # No hay suficientes datos históricos
        st.info("📊 **Datos históricos insuficientes**")
        st.markdown("""
        Para ver las métricas históricas, es necesario acumular datos a lo largo del tiempo.
        
        **¿Cómo generar datos históricos?**
        1. 🔄 Haga clic en "Refrescar Métricas" regularmente
        2. ⏰ Los datos se acumularán automáticamente con cada actualización
        3. 📈 En unas horas tendrá un gráfico histórico completo
        
        **Estado actual:**
        - Entradas de datos: {entries}
        - Tiempo mínimo requerido: 2+ entradas
        """.format(entries=len(historical_data['timestamps']) if historical_data['timestamps'] else 0))
        
        # Mostrar métricas actuales como referencia
        col_current1, col_current2, col_current3 = st.columns(3)
        
        with col_current1:
            st.metric(
                "CPU Actual", 
                f"{system_metrics.get('cpu_percent', 0):.1f}%"
            )
        
        with col_current2:
            st.metric(
                "Memoria Actual", 
                f"{system_metrics.get('memory_percent', 0):.1f}%"
            )
        
        with col_current3:
            st.metric(
                "Disco Actual", 
                f"{system_metrics.get('disk_percent', 0):.1f}%"
            )

def plot_training_metrics(training_history, chart_prefix=""):
    """Visualiza métricas de rendimiento de los modelos"""
    if not training_history or not isinstance(training_history, dict):
        st.warning("No hay datos de historial de entrenamiento disponibles")
        return

    metrics_data = []
    
    for model_name, history in training_history.items():
        if not history or not isinstance(history, dict):
            continue

        if 'accuracy' not in history or history['accuracy'] is None:
            continue
            
        entry = {
            'Model': model_name,
            'Accuracy': float(history.get('accuracy', 0)),
            'Val_Accuracy': float(history.get('val_accuracy', 0)),
            'Loss': float(history.get('loss', 0)),
            'Val_Loss': float(history.get('val_loss', 0))
        }

        if entry['Accuracy'] > 1:
            entry['Accuracy'] = min(1.0, entry['Accuracy'] / 100)
        if entry['Val_Accuracy'] > 1:
            entry['Val_Accuracy'] = min(1.0, entry['Val_Accuracy'] / 100)
            
        metrics_data.append(entry)
    
    if not metrics_data:
        st.warning("No hay suficientes datos de historial para visualizar")
        return
        
    df = pd.DataFrame(metrics_data)
    
    

    col1, col2 = st.columns(2)
    
    with col1:
        fig_accuracy = px.bar(
            df,
            x="Model",
            y=["Accuracy", "Val_Accuracy"],
            title="Comparación de Precisión entre Modelos",
            labels={"value": "Precisión", "variable": "Tipo"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        

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
        fig_loss = px.bar(
            df,
            x="Model",
            y=["Loss", "Val_Loss"],
            title="Comparación de Pérdida entre Modelos",
            labels={"value": "Pérdida", "variable": "Tipo"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        

        loss_max = df[["Loss", "Val_Loss"]].values.max() * 1.2 if len(df) > 0 else 2.0
        

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

def initialize_metrics_collection():
    """Inicializa la recolección de métricas del sistema si no existe historial"""
    try:
        historical_data = get_metrics_for_timeframe(hours=1)
        
        # Si no hay datos históricos, generar una entrada inicial
        if not historical_data['timestamps']:
            current_metrics = get_system_metrics()
            if current_metrics:
                save_system_metrics_history(current_metrics)
                st.success("🔄 Sistema de métricas inicializado")
        
        return True
    except Exception as e:
        st.warning(f"No se pudo inicializar el sistema de métricas: {e}")
        return False

def auto_update_metrics():
    """Actualiza automáticamente las métricas del sistema"""
    try:
        current_metrics = get_system_metrics()
        if current_metrics:
            save_system_metrics_history(current_metrics)
            return current_metrics
        return {}
    except Exception as e:
        st.error(f"Error actualizando métricas: {e}")
        return {}
