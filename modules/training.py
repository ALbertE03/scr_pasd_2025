import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from train import DistributedMLTrainer
import time
import pickle
import ray
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

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

def plot_model_comparison(results_data, chart_prefix="default"):
    """Crea gráfico de comparación de modelos"""
    if not results_data:
        st.warning("No hay datos de entrenamiento disponibles")
        return
    
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
    
    fig1 = px.bar(
        df, 
        x='Model', 
        y='Accuracy', 
        color='Dataset',
        title='Accuracy por Modelo y Dataset',
        height=400,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig1.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True, key=f"{chart_prefix}_model_comparison_accuracy_bar")

    fig2 = px.scatter(
        df, 
        x='Training_Time', 
        y='Accuracy', 
        color='Dataset',
        size='CV_Mean',
        hover_data=['Model'],
        title='Accuracy vs Tiempo de Entrenamiento',
        height=400,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"{chart_prefix}_model_comparison_accuracy_scatter")

def run_distributed_training(dataset_name, selected_models, enable_fault_tolerance=True):
    """Ejecuta entrenamiento distribuido con tolerancia a fallos"""
    try:
        trainer = DistributedMLTrainer(enable_fault_tolerance=enable_fault_tolerance)
        

        results = trainer.train_models_distributed(
            dataset_name=dataset_name,
            selected_models=selected_models
        )

        trainer.save_results(f"results_{dataset_name}.json")
        trainer.save_models(f"models_{dataset_name}")
        
        fault_stats = trainer.get_fault_tolerance_stats()
        if fault_stats and fault_stats.get('failed_tasks', 0) > 0:
            st.warning(f"⚠️ {fault_stats['failed_tasks']} tarea(s) fallaron pero el entrenamiento continuó")
            
            if 'fault_logs' in st.session_state:
                st.session_state.fault_logs.append({
                    "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "event": f"Entrenamiento completado con {fault_stats['failed_tasks']} fallos",
                    "type": "warning"
                })
        else:
            if 'fault_logs' in st.session_state:
                st.session_state.fault_logs.append({
                    "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "event": f"Entrenamiento exitoso en {dataset_name}",
                    "type": "success"
                })
        
        return results
        
    except Exception as e:
        if 'fault_logs' in st.session_state:
            st.session_state.fault_logs.append({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "event": f"Error en entrenamiento: {str(e)[:50]}...",
                "type": "error"
            })
        st.error(f"Error durante el entrenamiento: {e}")
        return None

def run_sequential_training(datasets_list, selected_models):
    """Ejecuta entrenamiento secuencial de múltiples datasets"""
    try:
        trainer = DistributedMLTrainer(enable_fault_tolerance=True)

        all_results, execution_summary = trainer.train_multiple_datasets_sequential(
            datasets_list=datasets_list,
            selected_models=selected_models
        )

        if 'fault_logs' in st.session_state:
            st.session_state.fault_logs.append({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "event": f"Entrenamiento secuencial completado: {execution_summary.get('successful_datasets', 0)}/{execution_summary.get('total_datasets', 0)} datasets",
                "type": "info"
            })
        
        return all_results, execution_summary
        
    except Exception as e:

        if 'fault_logs' in st.session_state:
            st.session_state.fault_logs.append({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "event": f"Error en entrenamiento secuencial: {str(e)[:50]}...",
                "type": "error"
            })
        st.error(f"Error durante el entrenamiento secuencial: {e}")
        return None, None

def load_execution_summary():
    """Carga el resumen de ejecución secuencial"""
    try:
        if os.path.exists("execution_summary.json"):
            with open("execution_summary.json", 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Error cargando resumen de ejecución: {e}")
    return None

def plot_cross_dataset_comparison(all_results):
    """Crea gráfico de comparación entre datasets"""
    if not all_results:
        st.warning("No hay datos de múltiples datasets disponibles")
        return

    comparison_data = []
    for dataset_name, results in all_results.items():
        for model_name, metrics in results.items():
            if metrics.get('status') == 'success':
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'CV_Mean': metrics['cv_mean'],
                    'Training_Time': metrics['training_time'],
                    'Node_ID': metrics.get('node_id', 'N/A')
                })
    
    if not comparison_data:
        st.warning("No hay datos exitosos para comparar")
        return    
    df = pd.DataFrame(comparison_data)
    

    fig1 = px.box(
        df, 
        x='Dataset', 
        y='Accuracy',
        color='Dataset',
        title='Distribución de Accuracy por Dataset',
        height=400,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True, key="cross_dataset_accuracy_box")
    
    best_models = df.loc[df.groupby('Dataset')['Accuracy'].idxmax()]
    fig2 = px.bar(
        best_models,
        x='Dataset',
        y='Accuracy',
        color='Model',
        title='Mejor Modelo por Dataset',
        text='Model',
        height=400,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, use_container_width=True, key="cross_dataset_best_models")

def get_fault_tolerance_stats():
    """Obtiene estadísticas de tolerancia a fallos del trainer"""
    try:
        trainer = DistributedMLTrainer()
        return trainer.get_fault_tolerance_stats()
    except Exception as e:
        st.error(f"Error obteniendo estadísticas de tolerancia a fallos: {e}")
        return {}

def plot_training_metrics(training_history, chart_prefix=""):
    """Visualiza métricas de rendimiento de los modelos"""
    if not training_history or not isinstance(training_history, dict):
        st.warning("No hay datos de historial de entrenamiento disponibles")
        return

    st.header("📊 Métricas de Entrenamiento")

    st.info(f"Procesando {len(training_history)} modelos para visualización")

    st.write("🔍 Modelos en el historial:")
    for idx, model_name in enumerate(training_history.keys()):
        st.write(f"  {idx+1}. {model_name}")

    st.markdown("---")

    metrics_data = []
    missing_models = []
    processed_models = []
    
    for model_name, history in training_history.items():
        if not history:
            st.warning(f"⚠️ Historia vacía para {model_name}")
            missing_models.append(model_name)
            continue
            
        if not isinstance(history, dict):
            st.warning(f"⚠️ Historia con formato inválido para {model_name}: {type(history)}")
            missing_models.append(model_name)
            continue

        fields_status = {
            'accuracy': 'accuracy' in history and history['accuracy'] is not None,
            'val_accuracy': 'val_accuracy' in history and history['val_accuracy'] is not None,
            'loss': 'loss' in history and history['loss'] is not None,
            'val_loss': 'val_loss' in history and history['val_loss'] is not None
        }

        if not fields_status['accuracy']:
            st.warning(f"⚠️ No hay métricas de accuracy para {model_name}")
            missing_models.append(model_name)
            continue

        accuracy = float(history.get('accuracy', 0))
        val_accuracy = float(history.get('val_accuracy', accuracy))  
        loss = float(history.get('loss', 0))
        val_loss = float(history.get('val_loss', loss))  
            
        entry = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Val_Accuracy': val_accuracy,
            'Loss': loss,
            'Val_Loss': val_loss
        }

        if entry['Accuracy'] > 1:
            entry['Accuracy'] = min(1.0, entry['Accuracy'] / 100)
        if entry['Val_Accuracy'] > 1:
            entry['Val_Accuracy'] = min(1.0, entry['Val_Accuracy'] / 100)
            
        metrics_data.append(entry)
        processed_models.append(model_name)
        st.success(f"✅ Añadido modelo {model_name} a la visualización")
    

    if missing_models:
        st.error(f"⚠️ Los siguientes modelos no pudieron ser incluidos: {', '.join(missing_models)}")
    
    if not metrics_data:
        st.warning("⛔ No hay suficientes datos válidos para visualizar")
        return

    df = pd.DataFrame(metrics_data)

    with st.expander("Diagnóstico de datos para visualización", expanded=True):
        st.write("Columnas disponibles:", df.columns.tolist())
        st.write(f"Número de modelos procesados: {len(processed_models)}")
        st.write(f"Número de filas en DataFrame: {len(df)}")
        st.write("Modelos incluidos en el DataFrame:", df['Model'].unique().tolist())
        
        if len(df) != len(processed_models):
            st.error(f"⚠️ Inconsistencia: {len(processed_models)} modelos procesados, pero {len(df)} en el DataFrame")
        
        st.subheader("Datos completos para graficar:")
        st.dataframe(df)

        st.subheader("Estado detallado por modelo")
        for model_name in training_history:
            st.write(f"Modelo: {model_name}")
            if model_name in df['Model'].values:
                st.success(f"✓ Incluido en la visualización")
                model_idx = df.index[df['Model'] == model_name].tolist()
                st.write(f"   Índice(s) en DataFrame: {model_idx}")
            else:
                st.error(f"✗ No incluido en la visualización")
                st.write(f"   Datos disponibles: {training_history[model_name]}")
    

    col1, col2 = st.columns(2)

    num_models = len(df)
    base_height = 400
    height_per_model = 30 
    chart_height = max(base_height, min(1200, base_height + (num_models - 6) * height_per_model))

    st.write(f"Número de modelos: {num_models}, altura del gráfico: {chart_height}px")
    
    with col1:

        if num_models > 10:
            fig_accuracy = px.bar(
                df,
                y="Model",  
                x=["Accuracy", "Val_Accuracy"],  
                title="Comparación de Precisión entre Modelos",
                labels={"value": "Precisión", "variable": "Tipo"},
                barmode="group",
                orientation='h', 
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_accuracy.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Precisión", tickformat=".0%", range=[0, 1]),
                yaxis=dict(title="Modelo", categoryorder='total ascending'),  
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(0,0,0,0.1)",
                yaxis_gridcolor="rgba(0,0,0,0.1)",
                height=chart_height,
                margin=dict(l=120, r=20, t=60, b=40)  
            )
        else:
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
                height=chart_height,
                margin=dict(l=20, r=20, t=60, b=40)
            )
        
        st.plotly_chart(fig_accuracy, use_container_width=True, key=f"{chart_prefix}_accuracy_plot")
    
    with col2:

        loss_max = df[["Loss", "Val_Loss"]].values.max() * 1.2 if len(df) > 0 else 2.0

        if num_models > 10:
            fig_loss = px.bar(
                df,
                y="Model",  
                x=["Loss", "Val_Loss"],  
                title="Comparación de Pérdida entre Modelos",
                labels={"value": "Pérdida", "variable": "Tipo"},
                barmode="group",
                orientation='h',  
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_loss.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Pérdida", range=[0, loss_max]),
                yaxis=dict(title="Modelo", categoryorder='total ascending'),  
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(0,0,0,0.1)",
                yaxis_gridcolor="rgba(0,0,0,0.1)",
                height=chart_height,
                margin=dict(l=120, r=20, t=60, b=40) 
            )
        else:
            fig_loss = px.bar(
                df,
                x="Model",
                y=["Loss", "Val_Loss"],
                title="Comparación de Pérdida entre Modelos",
                labels={"value": "Pérdida", "variable": "Tipo"},
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_loss.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Modelo"),
                yaxis=dict(title="Pérdida", range=[0, loss_max]),
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(0,0,0,0.1)",
                yaxis_gridcolor="rgba(0,0,0,0.1)",
                height=chart_height,
                margin=dict(l=20, r=20, t=60, b=40)
            )
        
        st.plotly_chart(fig_loss, use_container_width=True, key=f"{chart_prefix}_loss_plot")

def run_distributed_training_advanced(dataset_name, selected_models, hyperparameters=None, enable_fault_tolerance=True, progress_callback=None):
    """
    Ejecuta entrenamiento distribuido avanzado con monitoreo en tiempo real utilizando Ray
    
    Args:
        dataset_name: Nombre del dataset a utilizar
        selected_models: Lista de modelos a entrenar
        hyperparameters: Diccionario de hiperparámetros específicos para cada modelo
        enable_fault_tolerance: Si se habilita la tolerancia a fallos
        progress_callback: Función para reportar progreso
    """
   
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        
        results = {}
        training_history = {}

        status_text.text("⏳ Inicializando conexión con el cluster Ray...")
        trainer = DistributedMLTrainer(enable_fault_tolerance=enable_fault_tolerance)

        if not ray.is_initialized():
            status_text.text("⚠️ Conectando con cluster Ray...")
            try:
             
                head_address = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
                ray_port = os.getenv('RAY_HEAD_SERVICE_PORT', '10001')

                connection_attempts = [
                    f"ray://{head_address}:{ray_port}",  # Formato ray://
                    f"{head_address}:{ray_port}",        # Formato directo
                    "auto",                              # Autodetección
                    None                                 # Local
                ]
                
                connected = False
                for address in connection_attempts:
                    try:
                        if address:
                            status_text.text(f"⚠️ Intentando conectar a Ray en: {address}")
                            ray.init(address=address, ignore_reinit_error=True)
                        else:
                            status_text.text("⚠️ Iniciando Ray en modo local...")
                            ray.init(ignore_reinit_error=True)
                        
                        connected = True
                        status_text.text(f"✅ Conectado a Ray en: {ray.get_runtime_context().gcs_address}")
                        break
                    except Exception as e:
                        status_text.text(f"⚠️ No se pudo conectar a Ray en {address}: {str(e)}")
                        continue
                
                if not connected:
                    st.error("No se pudo conectar al cluster Ray después de varios intentos")
                    status_text.error("❌ No se pudo conectar con el cluster Ray")
                    return None, None
                
            except Exception as e:
                st.error(f"Error conectando con Ray: {str(e)}")
                status_text.error("❌ No se pudo conectar con el cluster Ray")
                return None, None
        try:
            cluster_info = ray.cluster_resources()
            cpu_count = int(cluster_info.get('CPU', 0))
            gpu_count = int(cluster_info.get('GPU', 0))
            nodes = ray.nodes()
            alive_nodes = [node for node in nodes if node.get('Alive', False)]
            
            status_text.text(f"🚀 Conectado al cluster Ray con {len(alive_nodes)} nodos y {cpu_count} CPUs disponibles")

            with st.expander("Detalles del cluster Ray", expanded=False):
                st.write(f"Nodos totales: {len(nodes)}")
                st.write(f"Nodos activos: {len(alive_nodes)}")
                st.write(f"CPUs disponibles: {cpu_count}")
                if gpu_count > 0:
                    st.write(f"GPUs disponibles: {gpu_count}")
                
                for i, node in enumerate(alive_nodes):
                    st.write(f"Nodo {i+1}: ID={node['NodeID'][:8]}, CPUs={node['Resources'].get('CPU', 0)}")
        except Exception as e:
            st.warning(f"No se pudo obtener información detallada del cluster Ray: {e}")

            alive_nodes = []
            cpu_count = 0
        

        datasets = trainer.get_available_datasets()
        if dataset_name not in datasets:
            status_text.error(f"❌ Dataset '{dataset_name}' no disponible")
            return None, None
            
        dataset = datasets[dataset_name]
        X, y = dataset.data, dataset.target

        available_models = trainer.get_available_models()
        models_to_train = {}
        
        for model_name in selected_models:
            if model_name in available_models:
                base_model = available_models[model_name]
                if hyperparameters and model_name in hyperparameters:
                    for param, value in hyperparameters[model_name].items():
                        if hasattr(base_model, param):
                            setattr(base_model, param, value)
                models_to_train[model_name] = base_model
        
        status_text.text(f"🧠 Preparando entrenamiento de {len(models_to_train)} modelos en {len(alive_nodes)} nodos")        # Inicializar historiales para cada modelo (ya no usamos listas para métricas)
        for model_name in models_to_train:
            training_history[model_name] = {
                "accuracy": None,
                "loss": None,
                "val_accuracy": None,
                "val_loss": None,
                "start_time": time.time(),
                "node_assignment": None
            }
        
        @ray.remote(num_cpus=1)
        def train_model_with_tracking(model, model_name, X, y, fold_idx, total_folds):
            """Función remota para entrenar un modelo y rastrear su progreso"""      
            try:

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=fold_idx)

                model.fit(X_train, y_train)

                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)

                train_acc = accuracy_score(y_train, train_preds)
                val_acc = accuracy_score(y_val, val_preds)

                try:
                    train_probs = model.predict_proba(X_train)
                    val_probs = model.predict_proba(X_val)
                    train_loss = log_loss(y_train, train_probs)
                    val_loss = log_loss(y_val, val_probs)
                except:

                    train_loss = 1.0 - train_acc
                    val_loss = 1.0 - val_acc

                train_acc = float(train_acc)
                val_acc = float(val_acc)
                train_loss = float(train_loss)
                val_loss = float(val_loss)

                if train_acc > 1:
                    train_acc = min(1.0, train_acc / 100)
                if val_acc > 1:
                    val_acc = min(1.0, val_acc / 100)

                train_loss = max(0, train_loss)
                val_loss = max(0, val_loss)

                if train_loss > 10:
                    train_loss = min(10.0, train_loss)
                if val_loss > 10:
                    val_loss = min(10.0, val_loss)
                return {
                    'model_name': model_name,
                    'fold': fold_idx,
                    'status': 'success',
                    'model': model, 
                    'metrics': {
                        'accuracy': train_acc,
                        'loss': train_loss,
                        'val_accuracy': val_acc,
                        'val_loss': val_loss
                    }
                }
            except Exception as e:
                return {
                    'model_name': model_name,
                    'fold': fold_idx,
                    'status': 'failed',
                    'error': str(e)
                }

        status_text.text("🚀 Iniciando entrenamiento distribuido con Ray...")
        total_tasks = len(models_to_train)  
        completed_tasks = 0
        tasks = []
        task_mapping = {} 
        completed_task_results = []  
        if alive_nodes:

            for i, (model_name, model) in enumerate(models_to_train.items()):

                node_idx = i % len(alive_nodes)
                node_id = alive_nodes[node_idx]['NodeID']
                training_history[model_name]['node_assignment'] = node_id
                status_text.text(f"🚀 Distribuyendo tarea: modelo {model_name} → nodo {node_id}")
                time.sleep(0.1)  
        else:

            for model_name, model in models_to_train.items():
                task = train_model_with_tracking.options(num_cpus=1).remote(model, model_name, X, y, 0, 1)
                
                tasks.append(task)
                task_mapping[task] = (model_name, 0)  

        while tasks:

            done_id, tasks = ray.wait(tasks, num_returns=1)
            try:                
                result = ray.get(done_id[0])
                completed_tasks += 1

                completed_task_results.append(result)

                progress = completed_tasks / total_tasks
                progress_bar.progress(progress)
                
                model_name = result['model_name']
                
                if result['status'] == 'success':                    
                    metrics = result['metrics']

                    accuracy = float(metrics['accuracy'])
                    val_accuracy = float(metrics['val_accuracy'])
                    loss = float(metrics['loss'])
                    val_loss = float(metrics['val_loss'])

                    if accuracy > 1:
                        accuracy = min(1.0, accuracy / 100)
                    if val_accuracy > 1:
                        val_accuracy = min(1.0, val_accuracy / 100)

                    loss = max(0, loss)
                    val_loss = max(0, val_loss)
                    training_history[model_name]['accuracy'] = accuracy
                    training_history[model_name]['loss'] = loss
                    training_history[model_name]['val_accuracy'] = val_accuracy
                    training_history[model_name]['val_loss'] = val_loss

                    with metrics_container.container():
                        st.caption(f"Entrenando {model_name} - Progreso {completed_tasks}/{total_tasks} ({(completed_tasks/total_tasks)*100:.0f}%)")

                        model_progress = {}
                        for m in training_history:

                            is_completed = training_history[m]['accuracy'] is not None
                            model_progress[m] = 100 if is_completed else 0  

                        for m, prog in model_progress.items():
                            st.caption(f"{m}: {'Completado' if prog == 100 else 'Pendiente'}")
                            st.progress(prog/100)

                        cols = st.columns(4)
                        acc = training_history[model_name]['accuracy']
                        val_acc = training_history[model_name]['val_accuracy']
                        loss = training_history[model_name]['loss']
                        val_loss = training_history[model_name]['val_loss']
                        cols[0].metric("Train Accuracy", f"{acc:.2%}")
                        cols[1].metric("Val Accuracy", f"{val_acc:.2%}")
                        cols[2].metric("Train Loss", f"{loss:.4f}")
                        cols[3].metric("Val Loss", f"{val_loss:.4f}")

                        elapsed = time.time() - training_history[model_name]['start_time']
                        st.caption(f"⏱️ Tiempo de entrenamiento: {elapsed:.1f} segundos")
                
                elif result['status'] == 'failed':
                    st.warning(f"Error en {model_name}, fold {result['fold']}: {result.get('error', 'Error desconocido')}")
                
            except Exception as e:
                st.error(f"Error procesando resultado: {str(e)}")

        for model_name in training_history:
            if training_history[model_name]['val_accuracy'] is not None:  
                start_time = training_history[model_name].get('start_time', time.time())
                end_time = time.time()
                training_duration = end_time - start_time
                
                results[model_name] = {
                    "accuracy": float(training_history[model_name]["val_accuracy"]),
                    "training_time": training_duration,
                    "cv_mean": float(training_history[model_name]["val_accuracy"]), 
                    "cv_std": 0.0,  
                    "timestamp": datetime.now().isoformat()
                }
            
        progress_bar.progress(1.0)
        status_text.success("✅ Entrenamiento completado exitosamente!")
        metrics_container.empty()
        if dataset_name and len(results) > 0:
            os.makedirs("training_results", exist_ok=True)

            history_filename = os.path.join("training_results", f"training_history_{dataset_name}.json")
            with open(history_filename, 'w') as f:
                json_history = {}
                for model_name, data in training_history.items():
                    json_history[model_name] = {
                        k: v for k, v in data.items() 
                        if isinstance(v, (list, dict, str, int, float)) or v is None
                    }
                json.dump(json_history, f)
            
            models_directory = os.path.join("training_results", f"models_{dataset_name}")
            os.makedirs(models_directory, exist_ok=True)
            
            best_models = {}
            for model_name in results.keys():
                try:
                    best_fold_idx = 0;
                    best_val_acc = 0;

                    for task in [t for t in completed_task_results if t['model_name'] == model_name]:
                        if task.get('status') == 'success' and 'metrics' in task:
                            val_acc = task['metrics'].get('val_accuracy', 0)
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_fold_idx = task.get('fold', 0)
                                if 'model' in task:
                                    best_models[model_name] = task['model']
                    
                    if model_name in best_models:
                        model_filename = os.path.join(models_directory, f"{model_name}.pkl")
                        with open(model_filename, 'wb') as f:
                            pickle.dump(best_models[model_name], f)
                        status_text.text(f"✅ Guardado mejor modelo {model_name} (fold {best_fold_idx}, val_acc: {best_val_acc:.2%})")
                    else:
                        st.warning(f"No se encontró un modelo válido para {model_name}")
                
                except Exception as e:
                    st.warning(f"Error al guardar el modelo {model_name}: {str(e)}")
            results_filename = os.path.join("training_results", f"results_{dataset_name}.json")
            with open(results_filename, 'w') as f:

                json_results = {}
                for model_name, result in results.items():
                    json_results[model_name] = {k: v for k, v in result.items()}
                json.dump(json_results, f)
            
            status_text.success(f"✅ Resultados y modelos guardados exitosamente!")

            save_path = os.path.abspath("training_results")
            st.success(f"""
            ## 💾 Modelos Guardados Exitosamente            
            Puede cargar estos modelos para inferencia o análisis posterior utilizando la pestaña de modelos.
            """)

        return results, training_history
        
    except Exception as e:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.error(f"❌ Error durante el entrenamiento: {e}")
        if metrics_container:
            metrics_container.empty()
        
        st.error(f"Error durante el entrenamiento distribuido avanzado: {e}")
        return None, None

def load_training_history(dataset_name):
    """Carga el historial de entrenamiento guardado"""
    history_filename = f"training_history_{dataset_name}.json"
    
    if os.path.exists(history_filename):
        try:
            with open(history_filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error cargando historial de entrenamiento: {e}")
    
    return None

def load_trained_model(dataset_name, model_name):
    """Carga un modelo entrenado desde el sistema de archivos"""
    model_path = os.path.join("training_results", f"models_{dataset_name}", f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        return None
        
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error cargando modelo {model_name} para dataset {dataset_name}: {str(e)}")
        return None

def get_trained_models_list(dataset_name):
    """Obtiene la lista de modelos entrenados disponibles para un dataset"""
    models_dir = os.path.join("training_results", f"models_{dataset_name}")
    
    if not os.path.exists(models_dir):
        return []
        
    try:

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

        model_names = [os.path.splitext(f)[0] for f in model_files]
        return sorted(model_names)
    except Exception as e:
        st.warning(f"Error al listar modelos entrenados: {str(e)}")
        return []

def plot_inference_metrics(inference_data, chart_prefix=""):
    """Visualiza métricas de inferencia de los modelos entrenados"""
    if not inference_data or not isinstance(inference_data, dict):
        st.warning("No hay datos de inferencia disponibles")
        return
    st.markdown("---")
    

    st.write(f"📊 Procesando {len(inference_data)} modelos para visualización de inferencia")

    metrics_data = []
    missing_models = []
    
    for model_name, results in inference_data.items():
        if not results or not isinstance(results, dict):
            st.warning(f"⚠️ Resultados inválidos para el modelo {model_name}")
            missing_models.append(model_name)
            continue
        
        metrics = {
            'Model': model_name,
            'Accuracy': float(results.get('accuracy', 0)),
            'Precision': float(results.get('precision', results.get('accuracy', 0))),
            'Recall': float(results.get('recall', results.get('accuracy', 0))),
            'F1': float(results.get('f1', results.get('accuracy', 0))),
            'Inference_Time': float(results.get('inference_time', 0))
        }
        
        for key in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if metrics[key] > 1:
                metrics[key] = min(1.0, metrics[key] / 100)
                
        metrics_data.append(metrics)
        st.write(f"✅ Añadido {model_name} a la visualización de inferencia")
    
    if missing_models:
        st.error(f"⚠️ Los siguientes modelos no pudieron ser incluidos en la visualización de inferencia: {', '.join(missing_models)}")
    
    if not metrics_data:
        st.warning("No hay suficientes datos de inferencia para visualizar")
        return
    
    df = pd.DataFrame(metrics_data)
    
    with st.expander("Información del DataFrame de inferencia", expanded=True):
        st.write("Columnas disponibles:", df.columns.tolist())
        st.write("Número de filas:", len(df))
        st.write("Valores únicos en 'Model':", df['Model'].unique().tolist())
        st.write("Datos completos:")
        st.write(df)
        
        st.subheader("Estado detallado por modelo (inferencia)")
        for model_name in inference_data:
            st.write(f"Modelo: {model_name}")
            if model_name in df['Model'].values:
                st.success("✓ Incluido en la visualización")
            else:
                st.error("✗ No incluido en la visualización")
                st.write("Datos disponibles:", inference_data[model_name])
    num_models = len(df)
    base_height = 400
    height_per_model = 30 
    chart_height = max(base_height, min(1200, base_height + (num_models - 6) * height_per_model))
    
    st.write(f"Número de modelos para inferencia: {num_models}, altura del gráfico: {chart_height}px")

    col1, col2 = st.columns(2)
    
    with col1:
        if num_models > 10:
            fig_metrics = px.bar(
                df,
                y="Model",  
                x=["Accuracy", "Precision", "Recall", "F1"], 
                title="Métricas de Rendimiento por Modelo",
                labels={"value": "Valor", "variable": "Métrica"},
                barmode="group",
                orientation='h',  
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_metrics.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Valor", tickformat=".0%", range=[0, 1]),
                yaxis=dict(title="Modelo", categoryorder='total ascending'), 
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(0,0,0,0.1)",
                yaxis_gridcolor="rgba(0,0,0,0.1)",
                height=chart_height,
                margin=dict(l=120, r=20, t=60, b=40)  
            )
        else:
            fig_metrics = px.bar(
                df,
                x="Model",
                y=["Accuracy", "Precision", "Recall", "F1"],
                title="Métricas de Rendimiento por Modelo",
                labels={"value": "Valor", "variable": "Métrica"},
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_metrics.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Modelo"),
                yaxis=dict(title="Valor", tickformat=".0%", range=[0, 1]),
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(0,0,0,0.1)",
                yaxis_gridcolor="rgba(0,0,0,0.1)",
                height=chart_height,
                margin=dict(l=20, r=20, t=60, b=40)
            )
        
        st.plotly_chart(fig_metrics, use_container_width=True, key=f"{chart_prefix}_metrics_plot")
    
    with col2:

        if 'Inference_Time' in df:
            max_time = df["Inference_Time"].max() * 1.2 if len(df) > 0 else 100
            if num_models > 10:
                fig_time = px.bar(
                    df,
                    y="Model", 
                    x="Inference_Time", 
                    title="Tiempo de Inferencia por Modelo",
                    labels={"Inference_Time": "Tiempo (ms)"},
                    orientation='h',  
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig_time.update_layout(
                    xaxis=dict(title="Tiempo (ms)", range=[0, max_time]),
                    yaxis=dict(title="Modelo", categoryorder='total ascending'), 
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_gridcolor="rgba(0,0,0,0.1)",
                    yaxis_gridcolor="rgba(0,0,0,0.1)",
                    height=chart_height,
                    margin=dict(l=120, r=20, t=60, b=40)  
                )
            else:
                fig_time = px.bar(
                    df,
                    x="Model",
                    y="Inference_Time",
                    title="Tiempo de Inferencia por Modelo",
                    labels={"Inference_Time": "Tiempo (ms)"},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig_time.update_layout(
                    xaxis=dict(title="Modelo"),
                    yaxis=dict(title="Tiempo (ms)", range=[0, max_time]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_gridcolor="rgba(0,0,0,0.1)",
                    yaxis_gridcolor="rgba(0,0,0,0.1)",
                    height=chart_height,
                    margin=dict(l=20, r=20, t=60, b=40)
                )
            
            st.plotly_chart(fig_time, use_container_width=True, key=f"{chart_prefix}_time_plot")
        else:
            st.warning("No hay datos de tiempo de inferencia disponibles")
