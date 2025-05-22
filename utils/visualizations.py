"""
Funciones de visualización para la plataforma de aprendizaje distribuido.
Incluye visualizaciones avanzadas 2D y 3D para análisis de rendimiento.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import datetime

def plot_training_metrics(metrics_file: str) -> Optional[Dict[str, Any]]:
    """
    Visualiza las métricas de entrenamiento desde un archivo JSON.
    
    Args:
        metrics_file: Ruta al archivo JSON de métricas
        
    Returns:
        Un diccionario con las métricas cargadas o None si hay un error
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        # Mostrar información general
        st.subheader("Información General")
        total_time = metrics.get("total_time", 0)
        data_loading_time = metrics.get("data_loading_time", 0)
        
        # Crear columnas para métricas principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tiempo Total", f"{total_time:.2f} s")
        with col2:
            st.metric("Tiempo de Carga", f"{data_loading_time:.2f} s")
        with col3:
            model_count = len(metrics.get("training_times", {}))
            st.metric("Modelos Entrenados", f"{model_count}")
            
        # Visualizar tiempos por modelo
        st.subheader("Tiempos por Modelo")
        
        if metrics.get("training_times") and metrics.get("evaluation_times"):
            # Crear DataFrame para visualización
            model_times = []
            for model_name, train_time in metrics["training_times"].items():
                eval_time = metrics["evaluation_times"].get(model_name, 0)
                model_times.append({
                    "Modelo": model_name,
                    "Tiempo de Entrenamiento": train_time,
                    "Tiempo de Evaluación": eval_time,
                    "Tiempo Total": train_time + eval_time
                })
                
            if model_times:
                model_times_df = pd.DataFrame(model_times)
                
                # Gráfico de barras para tiempos
                fig = px.bar(
                    model_times_df,
                    x="Modelo",
                    y=["Tiempo de Entrenamiento", "Tiempo de Evaluación"],
                    barmode="stack",
                    title="Tiempos de Entrenamiento y Evaluación por Modelo",
                    labels={"value": "Tiempo (s)", "variable": "Fase"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Visualizar métricas del sistema a lo largo del tiempo si están disponibles
        if metrics.get("system_metrics"):
            st.subheader("Métricas del Sistema Durante el Entrenamiento")
            
            system_metrics = metrics["system_metrics"]
            
            # Preparar datos para visualización
            timestamps = []
            cpu_usage = []
            memory_usage = []
            ray_resources = []
            
            base_time = system_metrics[0]["timestamp"] if system_metrics else 0
            
            for metric in system_metrics:
                rel_time = metric["timestamp"] - base_time
                timestamps.append(rel_time)
                
                if "system" in metric and "cpu" in metric["system"]:
                    cpu_usage.append(metric["system"]["cpu"]["percent"])
                else:
                    cpu_usage.append(None)
                
                if "system" in metric and "memory" in metric["system"]:
                    mem_percent = metric["system"]["memory"]["percent"]
                    memory_usage.append(mem_percent)
                else:
                    memory_usage.append(None)
                
                if "ray" in metric and "used_resources" in metric["ray"]:
                    try:
                        ray_cpu = metric["ray"]["used_resources"].get("CPU", 0)
                        ray_resources.append(ray_cpu)
                    except:
                        ray_resources.append(None)
                else:
                    ray_resources.append(None)
            
            # Crear gráficos de líneas para métricas del sistema
            if any(cpu_usage):
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cpu_usage,
                    mode='lines+markers',
                    name='CPU (%)'
                ))
                
                if any(memory_usage):
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=memory_usage,
                        mode='lines+markers',
                        name='Memoria (%)'
                    ))
                
                fig.update_layout(
                    title='Uso de Recursos del Sistema Durante el Entrenamiento',
                    xaxis_title='Tiempo (s)',
                    yaxis_title='Porcentaje (%)',
                    legend_title='Recurso'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Gráfico para recursos de Ray
            if any(ray_resources):
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=ray_resources,
                    mode='lines+markers',
                    name='CPUs de Ray Utilizadas'
                ))
                
                fig.update_layout(
                    title='Uso de Recursos de Ray Durante el Entrenamiento',
                    xaxis_title='Tiempo (s)',
                    yaxis_title='Unidades',
                    legend_title='Recurso'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        return metrics
    
    except Exception as e:
        st.error(f"Error al cargar o visualizar las métricas: {str(e)}")
        return None

def plot_model_comparison(model_registry, model_ids: List[str]):
    """
    Visualiza una comparación detallada entre modelos.
    
    Args:
        model_registry: Instancia del registro de modelos
        model_ids: Lista de IDs de modelos a comparar
    """
    if not model_ids:
        st.info("No hay modelos seleccionados para comparar.")
        return
    
    try:
        # Recopilar datos de los modelos
        models_data = []
        for model_id in model_ids:
            metadata = model_registry.get_metadata(model_id)
            
            # Datos básicos
            model_data = {
                "ID": model_id,
                "Nombre": metadata["name"],
                "Tipo": metadata["type"],
                "Creado": metadata["created_at"]
            }
            
            # Métricas de evaluación
            for metric_name, metric_value in metadata.get("metrics", {}).items():
                model_data[f"Métrica: {metric_name}"] = metric_value
                
            # Métricas de rendimiento
            for perf_name, perf_value in metadata.get("performance_metrics", {}).items():
                model_data[f"Rendimiento: {perf_name}"] = perf_value
                
            models_data.append(model_data)
            
        # Crear DataFrame
        if models_data:
            models_df = pd.DataFrame(models_data)
            
            # Mostrar tabla completa
            st.subheader("Tabla Comparativa")
            st.dataframe(models_df, use_container_width=True)
            
            # Separar métricas de evaluación
            eval_metrics = [col for col in models_df.columns if col.startswith("Métrica:")]
            if eval_metrics:
                st.subheader("Comparación de Métricas de Evaluación")
                
                # Preparar datos para visualización
                eval_data = []
                for _, row in models_df.iterrows():
                    model_name = row["Nombre"]
                    for metric in eval_metrics:
                        metric_name = metric.replace("Métrica: ", "")
                        eval_data.append({
                            "Modelo": model_name,
                            "Métrica": metric_name,
                            "Valor": row[metric]
                        })
                
                eval_df = pd.DataFrame(eval_data)
                
                # Crear gráfico de barras
                fig = px.bar(
                    eval_df,
                    x="Modelo",
                    y="Valor",
                    color="Métrica",
                    barmode="group",
                    title="Comparación de Métricas de Evaluación",
                    labels={"Valor": "Valor de la Métrica"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de radar para una vista alternativa
                fig = go.Figure()
                
                for model_name in models_df["Nombre"].unique():
                    model_metrics = eval_df[eval_df["Modelo"] == model_name]
                    
                    # Organizar las métricas en orden
                    metrics = []
                    values = []
                    for metric_name in sorted(model_metrics["Métrica"].unique()):
                        metric_row = model_metrics[model_metrics["Métrica"] == metric_name]
                        if not metric_row.empty:
                            metrics.append(metric_name)
                            values.append(metric_row["Valor"].values[0])
                    
                    # Cerrar el polígono
                    metrics.append(metrics[0])
                    values.append(values[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=model_name
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title="Gráfico de Radar de Métricas",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Separar métricas de rendimiento
            perf_metrics = [col for col in models_df.columns if col.startswith("Rendimiento:")]
            if perf_metrics:
                st.subheader("Comparación de Rendimiento")
                
                # Preparar datos para visualización
                perf_data = []
                for _, row in models_df.iterrows():
                    model_name = row["Nombre"]
                    for metric in perf_metrics:
                        metric_name = metric.replace("Rendimiento: ", "")
                        perf_data.append({
                            "Modelo": model_name,
                            "Métrica": metric_name,
                            "Valor": row[metric]
                        })
                
                perf_df = pd.DataFrame(perf_data)
                
                # Crear gráfico de barras
                fig = px.bar(
                    perf_df,
                    x="Modelo",
                    y="Valor",
                    color="Métrica",
                    barmode="group",
                    title="Comparación de Tiempos de Ejecución",
                    labels={"Valor": "Tiempo (s)"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error al comparar modelos: {str(e)}")

def plot_benchmark_results(benchmark_file: str):
    """
    Visualiza los resultados de los benchmarks de rendimiento.
    
    Args:
        benchmark_file: Ruta al archivo JSON de resultados de benchmark
    """
    try:
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        
        # Información general del benchmark
        st.subheader("Información del Benchmark")
        st.write(f"Fecha: {benchmark_data.get('timestamp', 'Desconocida')}")
        st.write(f"Tamaños de datasets: {', '.join(map(str, benchmark_data.get('dataset_sizes', [])))}")
        st.write(f"Repeticiones: {benchmark_data.get('repeats', 'Desconocido')}")
        
        # Extraer datos para visualización
        all_results = []
        for size_result in benchmark_data.get("results", []):
            dataset_size = size_result.get("dataset_size", 0)
            
            for model_result in size_result.get("models", []):
                model_name = model_result.get("model_name", "Desconocido")
                
                if "stats" in model_result:
                    stats = model_result["stats"]
                    result = {
                        "Tamaño del Dataset": dataset_size,
                        "Modelo": model_name,
                        "Tiempo Medio (s)": stats.get("mean_time", 0),
                        "Desviación Estándar": stats.get("std_time", 0),
                        "Tiempo Mínimo (s)": stats.get("min_time", 0),
                        "Tiempo Máximo (s)": stats.get("max_time", 0),
                        "Tasa de Éxito (%)": stats.get("success_rate", 0) * 100
                    }
                    all_results.append(result)
        
        if not all_results:
            st.warning("No hay resultados de benchmark para visualizar.")
            return
        
        # Crear DataFrame con todos los resultados
        results_df = pd.DataFrame(all_results)
        
        # Visualización tabular
        st.subheader("Resultados del Benchmark")
        st.dataframe(results_df, use_container_width=True)
        
        # Gráfico interactivo de barras para tiempos de ejecución
        st.subheader("Tiempos de Ejecución por Modelo y Tamaño")
        
        # Crear figura con barras de error
        fig = px.bar(
            results_df,
            x="Modelo",
            y="Tiempo Medio (s)",
            color="Tamaño del Dataset",
            barmode="group",
            error_y="Desviación Estándar",
            labels={"Tiempo Medio (s)": "Tiempo (s)"},
            hover_data=["Tiempo Mínimo (s)", "Tiempo Máximo (s)", "Tasa de Éxito (%)"]
        )
        
        fig.update_layout(
            title="Tiempos de Ejecución por Modelo y Tamaño de Dataset",
            xaxis_title="Modelo",
            yaxis_title="Tiempo (s)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico 3D para visualizar tiempos, tamaños y tasas de éxito
        st.subheader("Visualización 3D del Rendimiento")
        
        # Normalizar tamaños para mejor visualización
        sizes = results_df["Tamaño del Dataset"].unique()
        if len(sizes) > 1:
            size_min, size_max = min(sizes), max(sizes)
            normalized_sizes = [(s - size_min) / (size_max - size_min) * 50 + 10 for s in results_df["Tamaño del Dataset"]]
        else:
            normalized_sizes = [30] * len(results_df)
        
        # Crear gráfico 3D de dispersión
        fig = go.Figure(data=[go.Scatter3d(
            x=results_df["Tamaño del Dataset"],
            y=results_df["Tiempo Medio (s)"],
            z=results_df["Tasa de Éxito (%)"],
            mode='markers',
            marker=dict(
                size=normalized_sizes,
                color=results_df["Tiempo Medio (s)"],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Tiempo (s)")
            ),
            text=results_df["Modelo"],
            hovertemplate=
            "<b>%{text}</b><br>" +
            "Tamaño: %{x}<br>" +
            "Tiempo: %{y:.2f} s<br>" +
            "Éxito: %{z:.1f}%<br>"
        )])
        
        fig.update_layout(
            title="Visualización 3D: Tamaño vs. Tiempo vs. Éxito",
            scene=dict(
                xaxis_title='Tamaño del Dataset',
                yaxis_title='Tiempo Medio (s)',
                zaxis_title='Tasa de Éxito (%)'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de líneas para escalabilidad
        if len(sizes) > 1:
            st.subheader("Análisis de Escalabilidad")
            
            # Agrupar por modelo y tamaño
            pivot_df = results_df.pivot(index="Tamaño del Dataset", columns="Modelo", values="Tiempo Medio (s)")
            
            # Crear gráfico de líneas
            fig = px.line(
                pivot_df,
                labels={"value": "Tiempo (s)", "Tamaño del Dataset": "Tamaño del Dataset"},
                title="Escalabilidad: Tiempo vs. Tamaño del Dataset"
            )
            
            fig.update_layout(
                xaxis_title="Tamaño del Dataset",
                yaxis_title="Tiempo (s)",
                legend_title="Modelo"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error al visualizar resultados de benchmark: {str(e)}")

def plot_resource_usage(monitoring_data: Dict[str, Any]):
    """
    Visualiza el uso de recursos a lo largo del tiempo.
    
    Args:
        monitoring_data: Datos de monitorización de recursos
    """
    try:
        # Extraer datos de sistema
        system_stats = monitoring_data.get("system_stats", [])
        ray_stats = monitoring_data.get("ray_stats", [])
        
        if not system_stats or not ray_stats:
            st.warning("No hay datos de monitorización disponibles para visualizar.")
            return
        
        # Preparar datos para gráficos
        timestamps = []
        cpu_usage = []
        memory_usage = []
        disk_usage = []
        ray_cpu_usage = []
        ray_memory_usage = []
        
        base_time = system_stats[0].get("timestamp", 0)
        
        for stat in system_stats:
            timestamp = stat.get("timestamp", 0) - base_time
            timestamps.append(timestamp)
            
            cpu = stat.get("cpu", {}).get("percent", None)
            cpu_usage.append(cpu)
            
            memory = stat.get("memory", {}).get("percent", None)
            memory_usage.append(memory)
            
            disk = stat.get("disk", {}).get("percent", None)
            disk_usage.append(disk)
        
        ray_timestamps = []
        for stat in ray_stats:
            timestamp = stat.get("timestamp", 0) - base_time
            ray_timestamps.append(timestamp)
            
            # Calcular porcentaje de CPU de Ray usado
            total_cpu = stat.get("total_resources", {}).get("CPU", 0)
            used_cpu = stat.get("used_resources", {}).get("CPU", 0)
            
            if total_cpu > 0:
                ray_cpu_percent = (used_cpu / total_cpu) * 100
            else:
                ray_cpu_percent = None
            
            ray_cpu_usage.append(ray_cpu_percent)
            
            # Calcular porcentaje de memoria de Ray usado
            total_memory = stat.get("total_resources", {}).get("memory", 0)
            used_memory = stat.get("used_resources", {}).get("memory", 0)
            
            if total_memory > 0:
                ray_memory_percent = (used_memory / total_memory) * 100
            else:
                ray_memory_percent = None
            
            ray_memory_usage.append(ray_memory_percent)
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Uso de Recursos del Sistema", "Uso de Recursos de Ray"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Gráfico 1: Recursos del sistema
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cpu_usage,
                mode="lines+markers",
                name="CPU (%)",
                line=dict(color="red")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode="lines+markers",
                name="Memoria (%)",
                line=dict(color="blue")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=disk_usage,
                mode="lines+markers",
                name="Disco (%)",
                line=dict(color="green")
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Recursos de Ray
        fig.add_trace(
            go.Scatter(
                x=ray_timestamps,
                y=ray_cpu_usage,
                mode="lines+markers",
                name="Ray CPU (%)",
                line=dict(color="orange")
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=ray_timestamps,
                y=ray_memory_usage,
                mode="lines+markers",
                name="Ray Memoria (%)",
                line=dict(color="purple")
            ),
            row=2, col=1
        )
        
        # Actualizar diseño
        fig.update_layout(
            title="Monitorización de Recursos a lo Largo del Tiempo",
            xaxis_title="Tiempo (s)",
            height=700,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Porcentaje (%)", range=[0, 105], row=1, col=1)
        fig.update_yaxes(title_text="Porcentaje (%)", range=[0, 105], row=2, col=1)
        fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas resumidas
        st.subheader("Resumen Estadístico")
        
        # Estadísticas de sistema
        st.markdown("**Recursos del Sistema**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "CPU Promedio", 
                f"{np.mean([x for x in cpu_usage if x is not None]):.1f}%",
                f"Max: {np.max([x for x in cpu_usage if x is not None]):.1f}%"
            )
        with col2:
            st.metric(
                "Memoria Promedio", 
                f"{np.mean([x for x in memory_usage if x is not None]):.1f}%",
                f"Max: {np.max([x for x in memory_usage if x is not None]):.1f}%"
            )
        with col3:
            st.metric(
                "Disco Promedio", 
                f"{np.mean([x for x in disk_usage if x is not None]):.1f}%",
                f"Max: {np.max([x for x in disk_usage if x is not None]):.1f}%"
            )
        
        # Estadísticas de Ray
        st.markdown("**Recursos de Ray**")
        
        col1, col2 = st.columns(2)
        with col1:
            if ray_cpu_usage and any(x is not None for x in ray_cpu_usage):
                st.metric(
                    "Ray CPU Promedio", 
                    f"{np.mean([x for x in ray_cpu_usage if x is not None]):.1f}%",
                    f"Max: {np.max([x for x in ray_cpu_usage if x is not None]):.1f}%"
                )
            else:
                st.metric("Ray CPU Promedio", "N/A")
                
        with col2:
            if ray_memory_usage and any(x is not None for x in ray_memory_usage):
                st.metric(
                    "Ray Memoria Promedio", 
                    f"{np.mean([x for x in ray_memory_usage if x is not None]):.1f}%",
                    f"Max: {np.max([x for x in ray_memory_usage if x is not None]):.1f}%"
                )
            else:
                st.metric("Ray Memoria Promedio", "N/A")
    
    except Exception as e:
        st.error(f"Error al visualizar uso de recursos: {str(e)}")

def visualize_resource_forecast(forecaster):
    """
    Visualiza el pronóstico de uso de recursos.
    
    Args:
        forecaster: Instancia de ResourceForecaster
    """
    try:
        if len(forecaster.cpu_history) < 10:
            st.warning("Se necesitan más datos para realizar un pronóstico preciso.")
            st.info(f"Puntos de datos actuales: {len(forecaster.cpu_history)}/10 requeridos")
            return
        
        # Obtener pronósticos para diferentes horizontes temporales
        forecasts = {
            "5min": forecaster.forecast(minutes_ahead=5),
            "15min": forecaster.forecast(minutes_ahead=15),
            "30min": forecaster.forecast(minutes_ahead=30),
            "60min": forecaster.forecast(minutes_ahead=60)
        }
        
        # Preparar datos históricos para visualización
        timestamps = forecaster.timestamp_history.copy()
        base_time = timestamps[0]
        relative_times = [(t - base_time) / 60 for t in timestamps]  # Convert to minutes
        
        # Obtener hora actual como referencia
        current_time = time.time()
        
        # Preparar datos de pronóstico
        forecast_times = []
        cpu_forecasts = []
        mem_forecasts = []
        
        for label, forecast in forecasts.items():
            if "error" not in forecast:
                minutes_ahead = forecast["minutes_ahead"]
                forecast_times.append(minutes_ahead)
                cpu_forecasts.append(forecast["cpu_forecast"])
                mem_forecasts.append(forecast["memory_forecast"])
        
        # Crear gráfico para CPU
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=forecaster.cpu_history,
            mode='lines+markers',
            name='CPU Histórico',
            line=dict(color='blue')
        ))
        
        # Pronóstico
        if forecast_times:
            # Añadir último punto histórico como primer punto del pronóstico
            forecast_x = [0] + forecast_times
            forecast_y = [forecaster.cpu_history[-1]] + cpu_forecasts
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines+markers',
                name='CPU Pronóstico',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title='Pronóstico de Uso de CPU',
            xaxis_title='Tiempo (minutos)',
            yaxis_title='CPU (%)',
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Crear gráfico para Memoria
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=forecaster.memory_history,
            mode='lines+markers',
            name='Memoria Histórico',
            line=dict(color='green')
        ))
        
        # Pronóstico
        if forecast_times:
            # Añadir último punto histórico como primer punto del pronóstico
            forecast_x = [0] + forecast_times
            forecast_y = [forecaster.memory_history[-1]] + mem_forecasts
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines+markers',
                name='Memoria Pronóstico',
                line=dict(color='orange', dash='dash')
            ))
        
        fig.update_layout(
            title='Pronóstico de Uso de Memoria',
            xaxis_title='Tiempo (minutos)',
            yaxis_title='Memoria (%)',
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar valores de pronóstico
        st.subheader("Valores de Pronóstico")
        
        forecast_data = []
        for label, forecast in forecasts.items():
            if "error" not in forecast:
                forecast_time = datetime.datetime.fromtimestamp(forecast["forecast_timestamp"]).strftime('%H:%M:%S')
                forecast_data.append({
                    "Horizonte": label,
                    "CPU (%)": f"{forecast['cpu_forecast']:.1f}%",
                    "Memoria (%)": f"{forecast['memory_forecast']:.1f}%",
                    "Hora Pronóstico": forecast_time
                })
        
        if forecast_data:
            st.table(pd.DataFrame(forecast_data))
        
            # Tendencias
            st.subheader("Tendencias Actuales")
            
            cpu_trend = forecasts["15min"]["cpu_trend"]
            mem_trend = forecasts["15min"]["memory_trend"]
            
            col1, col2 = st.columns(2)
            with col1:
                trend_text = "estable" if abs(cpu_trend) < 0.5 else "creciente" if cpu_trend > 0 else "decreciente"
                st.metric(
                    "Tendencia de CPU", 
                    f"{cpu_trend:+.2f}% por minuto",
                    f"{trend_text}"
                )
            with col2:
                trend_text = "estable" if abs(mem_trend) < 0.5 else "creciente" if mem_trend > 0 else "decreciente"
                st.metric(
                    "Tendencia de Memoria", 
                    f"{mem_trend:+.2f}% por minuto",
                    f"{trend_text}"
                )
    
    except Exception as e:
        st.error(f"Error al visualizar pronóstico de recursos: {str(e)}")

def visualize_task_stats(task_stats: Dict[str, Any]):
    """
    Visualiza estadísticas de tareas de Ray.
    
    Args:
        task_stats: Estadísticas de tareas de Ray
    """
    try:
        if "error" in task_stats:
            st.warning(f"No hay datos de tareas disponibles: {task_stats['error']}")
            return
        
        # Información general
        st.metric("Total de Tareas Registradas", task_stats["total_tasks"])
        
        # Tareas por estado
        states = task_stats.get("states", {})
        if states:
            st.subheader("Tareas por Estado")
            
            # Preparar datos para gráfico
            state_labels = list(states.keys())
            state_values = list(states.values())
            
            # Crear gráfico de pastel
            fig = go.Figure(data=[go.Pie(
                labels=state_labels,
                values=state_values,
                hole=.4
            )])
            
            fig.update_layout(title="Distribución de Tareas por Estado")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tareas por función
        functions = task_stats.get("functions", {})
        if functions:
            st.subheader("Tareas por Función")
            
            # Preparar datos para gráfico
            func_df = pd.DataFrame({
                "Función": list(functions.keys()),
                "Cantidad": list(functions.values())
            }).sort_values("Cantidad", ascending=False)
            
            # Crear gráfico de barras
            fig = px.bar(
                func_df,
                x="Función",
                y="Cantidad",
                color="Función",
                title="Distribución de Tareas por Función"
            )
            
            fig.update_layout(xaxis_title="Función", yaxis_title="Número de Tareas")
            st.plotly_chart(fig, use_container_width=True)
        
        # Uso promedio de recursos
        avg_resources = task_stats.get("avg_resources", {})
        if avg_resources:
            st.subheader("Uso Promedio de Recursos por Tarea")
            
            # Preparar datos para gráfico
            resource_df = pd.DataFrame({
                "Recurso": list(avg_resources.keys()),
                "Uso Promedio": list(avg_resources.values())
            })
            
            # Crear gráfico de barras
            fig = px.bar(
                resource_df,
                x="Recurso",
                y="Uso Promedio",
                color="Recurso",
                title="Uso Promedio de Recursos por Tarea"
            )
            
            fig.update_layout(xaxis_title="Recurso", yaxis_title="Cantidad Promedio")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error al visualizar estadísticas de tareas: {str(e)}")

def visualize_alerts(alerts: List[Dict[str, Any]]):
    """
    Visualiza alertas de recursos.
    
    Args:
        alerts: Lista de alertas de recursos
    """
    try:
        if not alerts:
            st.info("No hay alertas de recursos para mostrar.")
            return
        
        # Mostrar tabla de alertas
        alerts_df = pd.DataFrame([
            {
                "Recurso": alert["resource"],
                "Valor": f"{alert['value']:.1f}%",
                "Umbral": f"{alert['threshold']:.1f}%",
                "Mensaje": alert["message"],
                "Hora": datetime.datetime.fromtimestamp(alert["timestamp"]).strftime('%H:%M:%S')
            }
            for alert in alerts
        ])
        
        st.dataframe(alerts_df, use_container_width=True)
        
        # Agregar gráfico de alertas por recurso
        resource_counts = {}
        for alert in alerts:
            resource = alert["resource"]
            resource_counts[resource] = resource_counts.get(resource, 0) + 1
        
        # Preparar datos para gráfico
        resource_df = pd.DataFrame({
            "Recurso": list(resource_counts.keys()),
            "Alertas": list(resource_counts.values())
        })
        
        # Crear gráfico de barras
        fig = px.bar(
            resource_df,
            x="Recurso",
            y="Alertas",
            color="Recurso",
            title="Alertas por Recurso"
        )
        
        fig.update_layout(xaxis_title="Recurso", yaxis_title="Número de Alertas")
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error al visualizar alertas: {str(e)}")
