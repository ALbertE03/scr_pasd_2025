import requests
import streamlit as st
import pandas as pd
import os
import time
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def get_api_base_url():
    if os.environ.get('API_BASE_URL'):
        return os.environ.get('API_BASE_URL')
    elif os.path.exists('/.dockerenv') or os.environ.get('CONTAINER_NAME'):
        return "http://ml-api:8000"
    else:
        return "http://localhost:8000"

API_BASE_URL = get_api_base_url()


class APIClient:
    """Cliente para interactuar con la API de ML"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        print(f"APIClient inicializado con URL: {self.base_url}")
    
    def health_check(self) -> Dict:
        """Verifica el estado de salud de la API"""
        try:
            print(f"Verificando salud de API en: {self.base_url}/health")
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            print(f"Error en health_check: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_models(self) -> Dict:
        """Obtiene la lista de modelos disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_model_info(self, model_name: str) -> Dict:
        """Obtiene información detallada de un modelo"""
        try:
            response = self.session.get(f"{self.base_url}/models/{model_name}", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_models_by_dataset(self, dataset_name: str) -> Dict:
        """Obtiene modelos por dataset específico"""
        try:
            response = self.session.get(f"{self.base_url}/models/dataset/{dataset_name}", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def predict(self, model_name: str, features: List[List[float]], return_probabilities: bool = False) -> Dict:
        """Realiza predicciones usando un modelo"""
        try:
            payload = {
                "model_name": model_name,
                "features": features,
                "return_probabilities": return_probabilities
            }
            response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_cluster_status(self) -> Dict:
        """Obtiene el estado del cluster"""
        try:
            response = self.session.get(f"{self.base_url}/cluster/status", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_datasets(self) -> Dict:
        """Obtiene la lista de datasets disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/datasets", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_algorithms(self) -> Dict:
        """Obtiene la lista de algoritmos disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/algorithms", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def search_models(self, query: str) -> Dict:        
        """Busca modelos por query"""
        try:
            response = self.session.get(f"{self.base_url}/models/search/{query}", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}        
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def delete_model(self, model_name: str) -> Dict:
        """Elimina un modelo"""
        try:
            print(f"Intentando eliminar modelo: {model_name}")
            url = f"{self.base_url}/models/{model_name}"
            print(f"URL de solicitud: {url}")
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                
                    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
                    response = self.session.delete(url, headers=headers, timeout=30)

                    if response.status_code == 200:
                        success = True
                        print(f"Eliminación exitosa. Respuesta: {response.status_code}")
                        try:
                            result = response.json()
                            print(f"Contenido de la respuesta: {result}")
                            return {"status": "success", "data": result}
                        except ValueError:
                            print(f"Respuesta recibida (no JSON): {response.text}")
                            return {
                                "status": "success", 
                                "data": {
                                    "message": "Modelo eliminado exitosamente",
                                    "response_text": response.text
                                }
                            }
                    else:
                        print(f"Error en respuesta: {response.status_code} - {response.text}")
                        retry_count += 1
                        time.sleep(1) 
                        
                except requests.exceptions.RequestException as req_e:
                    print(f"Error de solicitud (intento {retry_count+1}): {str(req_e)}")
                    retry_count += 1
                    time.sleep(1)
            
            if not success:
                return {"status": "error", "error": f"Error eliminando modelo después de {max_retries} intentos"}
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {str(e)}")
            return {"status": "error", "error": f"Error de conexión: {str(e)}"}
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def predict_batch(self, model_name: str, file_data: bytes, filename: str, return_probabilities: bool = False) -> Dict:
        """Realiza predicciones en lote desde archivo"""
        try:
            files = {"file": (filename, file_data, "text/csv")}
            data = {
                "model_name": model_name,
                "return_probabilities": return_probabilities
            }
            response = self.session.post(f"{self.base_url}/predict/batch", files=files, data=data, timeout=60)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_inference_stats(self, model_name: str = None) -> Dict:
        """Obtiene estadísticas de inferencia en tiempo real"""
        try:
            params = {"model_name": model_name} if model_name else {}
            response = self.session.get(f"{self.base_url}/inference-stats", params=params, timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}


def render_api_tab():
    """Renderiza la pestaña de API"""
    st.header("🌐 API de Modelos ML")
    
    api_client = APIClient()
    
    health_status = api_client.health_check()
    
    if health_status["status"] == "error":
        st.error(f"❌ API no disponible: {health_status['error']}")
        st.info("Para usar la API, ejecute el contenedor o inicie la API localmente con: `python api.py`")
        return
    
    st.success("✅ API conectada correctamente")
 
    api_tabs = st.tabs([
        "🔍 Explorar Modelos",
        "🎯 Predicciones",
        "📊 Estadísticas de Inferencia"
    ])
    
    with api_tabs[0]:
        render_explore_models_tab(api_client)
    
    with api_tabs[1]:
        render_predictions_tab(api_client)
    
    with api_tabs[2]:
        render_inference_stats_tab(api_client)
    


def render_explore_models_tab(api_client: APIClient):
    """Renderiza la pestaña de exploración de modelos"""
    st.subheader("🔍 Explorar Modelos Disponibles")    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Actualizar", key="refresh_models_api"):
            st.rerun()   
    models_response = api_client.get_models()
    
    if models_response["status"] == "error":
        st.error(f"Error obteniendo modelos: {models_response['error']}")
        return
    
    models_data = models_response["data"]
    models = models_data.get("models", {})
    
    if not models:
        st.info("No hay modelos disponibles. Entrene algunos modelos primero.")
        return
    
    st.metric("Total de Modelos", models_data.get("total_models", 0))
    
    with st.expander("🔎 Filtros y Búsqueda"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            datasets = list(set([model.get("dataset", "unknown") for model in models.values()]))
            selected_datasets = st.multiselect("Filtrar por Dataset", datasets, default=datasets)
        
        with col2:
            search_query = st.text_input("Buscar modelo", placeholder="Nombre del modelo...")
        
        with col3:
            sort_by = st.selectbox("Ordenar por", ["accuracy", "training_time", "modified_time", "file_size_mb"])
    
    filtered_models = {}
    for name, model in models.items():
        if model.get("dataset", "unknown") not in selected_datasets:
            continue

        if search_query and search_query.lower() not in name.lower():
            continue
        
        filtered_models[name] = model

    if filtered_models:
        try:
            sorted_models = dict(sorted(
                filtered_models.items(),
                key=lambda x: x[1].get(sort_by, 0),
                reverse=True
            ))
        except:
            sorted_models = filtered_models
    else:
        sorted_models = {}

    if sorted_models:
        cols = st.columns(2)
        for i, (model_name, model_info) in enumerate(sorted_models.items()):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"### {model_name}")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Dataset", model_info.get("dataset", "N/A"))
                        st.metric("Accuracy", f"{model_info.get('accuracy', 0):.4f}")
                    
                    with col_info2:
                        st.metric("Tiempo Entrenamiento", f"{model_info.get('training_time', 0):.2f}s")
                        st.metric("Tamaño Archivo", f"{model_info.get('file_size_mb', 0):.2f} MB")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button(f"📋 Detalles", key=f"details_{model_name}"):
                            show_model_details(api_client, model_name)
                    
                    with col_btn2:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{model_name}", type="secondary"):
                            with st.spinner(f"Eliminando modelo {model_name}..."):
                                
                                response = api_client.delete_model(model_name)
                                if response["status"] == "success":
                                    st.success(f"✅ Modelo '{model_name}' eliminado correctamente")
                                    st.rerun()  
                                else:
                                    st.error(f"❌ Error: {response.get('error', 'Error desconocido')}")
    else:
        st.info("No hay modelos que coincidan con los filtros seleccionados.")


def show_model_details(api_client: APIClient, model_name: str):
    """Muestra detalles detallados de un modelo"""
    model_info_response = api_client.get_model_info(model_name)
    
    if model_info_response["status"] == "error":
        st.error(f"Error obteniendo detalles: {model_info_response['error']}")
        return
    
    model_info = model_info_response["data"]
    
    with st.expander(f"📋 Detalles de {model_name}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Nombre": model_info.get("model_name"),
                "Dataset": model_info.get("dataset"),
                "Tipo": model_info.get("model_type", "N/A"),
                "Archivo": model_info.get("file_path"),
                "Directorio": model_info.get("directory")
            })
        
        with col2:
            st.json({
                "Accuracy": model_info.get("accuracy"),
                "CV Mean": model_info.get("cv_mean"),
                "CV Std": model_info.get("cv_std"),
                "Tiempo Entrenamiento": model_info.get("training_time"),
                "Timestamp": model_info.get("timestamp")
            })
        
        if "model_parameters" in model_info:
            st.subheader("Parámetros del Modelo")
            st.json(model_info["model_parameters"])


def render_predictions_tab(api_client: APIClient):
    """Renderiza la pestaña de predicciones"""
    st.subheader("🎯 Realizar Predicciones")
    
    models_response = api_client.get_models()
    if models_response["status"] == "error":
        st.error(f"Error obteniendo modelos: {models_response['error']}")
        return
    
    models = models_response["data"].get("models", {})
    if not models:
        st.warning("No hay modelos disponibles para hacer predicciones.")
        return
    
    model_names = list(models.keys())
    
    prediction_tabs = st.tabs(["📝 Predicción Individual", "📁 Predicción en Lote"])
    
    with prediction_tabs[0]:
        render_individual_prediction(api_client, model_names, models)
    
    with prediction_tabs[1]:
        render_batch_prediction(api_client, model_names)



def render_individual_prediction(api_client: APIClient, model_names: List[str], models: Dict):
    """Renderiza predicción individual"""
    st.subheader("📝 Predicción Individual")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_model = st.selectbox("Seleccionar Modelo", model_names)
    with col2:
        return_probabilities = st.checkbox("Incluir Probabilidades", value=True)    
    
    if selected_model:
        model_info = models[selected_model]
        st.info(f"Dataset: {model_info.get('dataset')} | Accuracy: {model_info.get('accuracy', 0):.4f}")
        dataset = model_info.get('dataset', 'iris')
        
        if dataset == 'iris':
            feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        elif dataset == 'breast_cancer':
            feature_names = [
                'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
                'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
                'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
                'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
                'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
            ]
        elif dataset == 'wine':
            feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 
                           'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 
                           'od280/od315_of_diluted_wines', 'proline']
        elif dataset == 'digits':
            feature_names = [f'pixel_{i}' for i in range(64)]
        else:
            feature_names = [f'Feature {i+1}' for i in range(model_info.get('n_features', 4))]
        
        n_features = len(feature_names)
        
        st.subheader("Introducir Características")
        features = []
        if n_features <= 8:
            st.write(f"Características del modelo ({n_features} en total):")
            cols = st.columns(min(n_features, 4))
            for i in range(n_features):
                with cols[i % 4]:
                    value = st.number_input(
                        feature_names[i] if i < len(feature_names) else f"Feature {i+1}",
                        value=1.0,
                        step=0.1,
                        key=f"feature_{i}_{selected_model}",
                        format="%.4f"
                    )
                    features.append(value)
        
            
        else:
            if dataset == 'breast_cancer':
                
                example_features = [1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01, 
                                  3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
                                  8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
                                  3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
                                  1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01]
            elif dataset == 'digits':
                example_features = [0.0] * 64  
            else:
                example_features = [1.0] * n_features           
            st.write(f"Todas las características ({n_features} en total):")
        
            feature_tabs = st.tabs(["Editor por Grupos"])
            
            with feature_tabs[0]:
                num_tabs = (n_features + 15) // 16  
                group_tabs = st.tabs([f"Grupo {i+1}" for i in range(num_tabs)])
                
                for tab_idx, tab in enumerate(group_tabs):
                    with tab:
                        start_idx = tab_idx * 16
                        end_idx = min(start_idx + 16, n_features)
                        st.write(f"Características {start_idx+1}-{end_idx}")
                        
                        cols = st.columns(4)
                        for i in range(start_idx, end_idx):
                            with cols[(i-start_idx) % 4]:
                                value = st.number_input(
                                    f"{feature_names[i]}" if i < len(feature_names) else f"F{i+1}",
                                    value=float(example_features[i]),
                                    step=0.1,
                                    key=f"feature_{i}_{selected_model}",
                                    format="%.4f"
                                )
                                example_features[i] = value
           
            features = example_features

        if st.button("🔮 Realizar Predicción", type="primary"):
            with st.spinner("Realizando predicción..."):
                prediction_response = api_client.predict(
                    selected_model, 
                    [features], 
                    return_probabilities
                )
                
                if prediction_response["status"] == "error":
                    st.error(f"Error en predicción: {prediction_response['error']}")
                else:
                    result = prediction_response["data"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicción", result["predictions"][0])
                    
                    with col2:
                        st.metric("Tiempo", f"{result['prediction_time']:.3f}s")
                    
                    with col3:
                        st.metric("Características", result["feature_count"])
                        
                    
                    if result.get("probabilities"):
                        st.subheader("Probabilidades por Clase")
                        probs = result["probabilities"][0]
                        
                        if dataset == 'iris':
                            class_names = ['Setosa', 'Versicolor', 'Virginica']
                        elif dataset == 'wine':
                            class_names = ['Clase 1', 'Clase 2', 'Clase 3']
                        elif dataset == 'breast_cancer':
                            class_names = ['Benigno', 'Maligno']
                        else:
                            class_names = [f"Clase {i}" for i in range(len(probs))]
                        
                        
                        # Gráfico de barras con las probabilidades
                        fig = px.bar(
                            x=class_names[:len(probs)],
                            y=probs,
                            title="Distribución de Probabilidades por Clase",
                            labels={"x": "Clase", "y": "Probabilidad"}
                        )
                        st.plotly_chart(fig, use_container_width=True)


def render_batch_prediction(api_client: APIClient, model_names: List[str]):
    """Renderiza predicción en lote"""
    st.subheader("📁 Predicción en Lote")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox("Seleccionar Modelo", model_names, key="batch_model")
    
    with col2:
        return_probabilities = st.checkbox("Incluir Probabilidades", value=False, key="batch_probs")
    
    uploaded_file = st.file_uploader(
        "Subir archivo CSV con características",
        type=['csv'],
        help="El archivo debe contener las características en columnas, una fila por muestra a predecir."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa del archivo:")
            st.dataframe(df.head())
            
            st.info(f"Archivo contiene {len(df)} muestras con {len(df.columns)} características")
            
            if st.button("🚀 Ejecutar Predicción en Lote", type="primary"):
                with st.spinner("Procesando predicciones en lote..."):
                    uploaded_file.seek(0)
                    file_data = uploaded_file.read()
                    
                    prediction_response = api_client.predict_batch(
                        selected_model,
                        file_data,
                        uploaded_file.name,
                        return_probabilities
                    )
                    
                    if prediction_response["status"] == "error":
                        st.error(f"Error en predicción: {prediction_response['error']}")
                    else:
                        result = prediction_response["data"]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Muestras Procesadas", result["n_samples"])
                        with col2:
                            st.metric("Características", result["feature_count"])
                        with col3:
                            st.metric("Tiempo Total", f"{result['prediction_time']:.3f}s")
                        
                        with col4:
                            st.metric("Tiempo/Muestra", f"{result['prediction_time']/result['n_samples']:.4f}s")

                        
                            st.subheader("Resultados de Predicción")
                        
                        results_df = df.copy()
                        results_df['Predicción'] = result["predictions"]
                        
                        if result.get("probabilities"):
                            probs = result["probabilities"]
                            for i, prob_row in enumerate(probs):
                                for j, prob in enumerate(prob_row):
                                    results_df[f'Prob_Clase_{j}'] = [prob_row[j] if idx == i else None for idx in range(len(probs))]
                        
                        st.dataframe(results_df)

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name=f"predicciones_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                        st.subheader("Distribución de Predicciones")
                        pred_counts = pd.Series(result["predictions"]).value_counts().sort_index()
                        
                        fig = px.bar(
                            x=pred_counts.index,
                            y=pred_counts.values,
                            title="Distribución de Clases Predichas",
                            labels={"x": "Clase Predicha", "y": "Cantidad"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")


def render_inference_stats_tab(api_client: APIClient):
    """Renderiza estadísticas de inferencia en tiempo real"""
    st.subheader("📊 Estadísticas de Inferencia en Tiempo Real")
    
    stats_response = api_client.get_inference_stats()
    
    if stats_response["status"] == "error":
        st.error(f"Error obteniendo estadísticas de inferencia: {stats_response['error']}")
        return
    
    stats_data = stats_response["data"]
    
    if not stats_data.get("stats"):
        st.info("No hay estadísticas de inferencia disponibles. Realiza algunas predicciones para ver los datos.")
        return
    
    raw_stats = stats_data["stats"]
    aggregated_stats = stats_data.get("aggregated", {})

    model_names = list(raw_stats.keys())
    selected_model = st.selectbox("Seleccionar modelo:", ["Todos"] + model_names)

    st.subheader("Resumen de Estadísticas")
    col1, col2, col3, col4 = st.columns(4)
    
    if selected_model == "Todos":
        total_predictions = sum(len(entries) for entries in raw_stats.values())
        total_samples = sum(agg.get("total_samples", 0) for agg in aggregated_stats.values())
        avg_prediction_time = sum(agg.get("avg_prediction_time", 0) for agg in aggregated_stats.values()) / len(aggregated_stats) if aggregated_stats else 0
        active_models = len(aggregated_stats)
        
        with col1:
            st.metric("Total Predicciones", total_predictions)
        with col2:
            st.metric("Total Muestras", total_samples)
        with col3:
            st.metric("Tiempo Promedio", f"{avg_prediction_time:.3f}s")
        with col4:
            st.metric("Modelos Activos", active_models)
    else:
        if selected_model in aggregated_stats:
            agg = aggregated_stats[selected_model]
            with col1:
                st.metric("Predicciones", agg.get("total_predictions", 0))
            with col2:
                st.metric("Muestras", agg.get("total_samples", 0))
            with col3:
                st.metric("Tiempo Promedio", f"{agg.get('avg_prediction_time', 0):.3f}s")
            with col4:
                accuracy = agg.get("accuracy")
                st.metric("Accuracy", f"{accuracy:.4f}" if accuracy else "N/A")
    

    st.subheader("Evolución Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:

        if selected_model == "Todos":
            fig = go.Figure()
            for model_name, entries in raw_stats.items():
                if entries:
                    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in entries]
                    prediction_times = [entry["prediction_time"] for entry in entries]
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=prediction_times,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
            fig.update_layout(
                title="Tiempo de Predicción por Modelo",
                xaxis_title="Tiempo",
                yaxis_title="Tiempo de Predicción (s)",
                hovermode='x unified'
            )
        else:
            if selected_model in raw_stats and raw_stats[selected_model]:
                entries = raw_stats[selected_model]
                timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in entries]
                prediction_times = [entry["prediction_time"] for entry in entries]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prediction_times,
                    mode='lines+markers',
                    name=f"Tiempo - {selected_model}",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title=f"Tiempo de Predicción - {selected_model}",
                    xaxis_title="Tiempo",
                    yaxis_title="Tiempo de Predicción (s)"
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="No hay datos disponibles", x=0.5, y=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if selected_model == "Todos":
            fig = go.Figure()
            for model_name, entries in raw_stats.items():
                if entries:
                    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in entries]
                    n_samples = [entry["n_samples"] for entry in entries]
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=n_samples,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
            fig.update_layout(
                title="Muestras Procesadas por Modelo",
                xaxis_title="Tiempo",
                yaxis_title="Número de Muestras",
                hovermode='x unified'
            )
        else:
            if selected_model in raw_stats and raw_stats[selected_model]:
                entries = raw_stats[selected_model]
                timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in entries]
                n_samples = [entry["n_samples"] for entry in entries]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=n_samples,
                    mode='lines+markers',
                    name=f"Muestras - {selected_model}",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title=f"Muestras Procesadas - {selected_model}",
                    xaxis_title="Tiempo",
                    yaxis_title="Número de Muestras"
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="No hay datos disponibles", x=0.5, y=0.5)
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadísticas Detalladas por Modelo")
    if aggregated_stats:
        table_data = []
        for model_name, agg in aggregated_stats.items():
            table_data.append({
                "Modelo": model_name,
                "Predicciones": agg.get("total_predictions", 0),
                "Muestras Totales": agg.get("total_samples", 0),
                "Tiempo Promedio": f"{agg.get('avg_prediction_time', 0):.3f}s",
                "Tiempo Min": f"{agg.get('min_prediction_time', 0):.3f}s",
                "Tiempo Max": f"{agg.get('max_prediction_time', 0):.3f}s",
                "Tiempo/Muestra": f"{agg.get('avg_time_per_sample', 0):.4f}s",
                "Última Predicción": agg.get("last_prediction", "N/A"),
                "Accuracy": f"{agg.get('accuracy', 0):.4f}" if agg.get('accuracy') else "N/A"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    if selected_model != "Todos" and selected_model in raw_stats:
        st.subheader(f"Detalle de Inferencias - {selected_model}")
        entries = raw_stats[selected_model]
        
        if entries:
            recent_entries = entries[-20:] if len(entries) > 20 else entries
            recent_data = []
            
            for entry in reversed(recent_entries): 
                recent_data.append({
                    "Timestamp": entry["timestamp"],
                    "Tiempo Predicción": f"{entry['prediction_time']:.3f}s",
                    "Muestras": entry["n_samples"],
                    "Tiempo/Muestra": f"{entry['avg_time_per_sample']:.4f}s",
                    "Accuracy": f"{entry['accuracy']:.4f}" if entry.get('accuracy') else "N/A"
                })
            st.dataframe(pd.DataFrame(recent_data), use_container_width=True)
        else:
            st.info("No hay inferencias registradas para este modelo.")
    st.subheader("Información del Sistema")

    models_response = api_client.get_models()
    models = models_response["data"].get("models", {}) if models_response["status"] == "success" else {}
    
    health_status = api_client.health_check()
    cluster_status = api_client.get_cluster_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_models = len(models)
        st.metric("Modelos Disponibles", total_models)
    
    with col2:
        if health_status["status"] == "success":
            ray_status = health_status["data"].get("ray_initialized", False)
            st.metric("Ray Status", "Conectado" if ray_status else "Desconectado")
        else:
            st.metric("API Status", "Error")
    
    with col3:
        if cluster_status["status"] == "success":
            alive_nodes = cluster_status["data"].get("alive_nodes", 0)
            total_nodes = cluster_status["data"].get("total_nodes", 0)
            st.metric("Nodos Cluster", f"{alive_nodes}/{total_nodes}")
        else:
            st.metric("Nodos Cluster", "N/A")
    
    with col4:
        total_size = sum(model.get("file_size_mb", 0) for model in models.values())
        st.metric("Tamaño Total Modelos", f"{total_size:.1f} MB")


def delete_model_confirm(api_client: APIClient, model_name: str):
    """Elimina un modelo directamente"""
    st.warning(f"¿Estás seguro de que deseas eliminar el modelo '{model_name}'?")
    st.write("Esta acción no se puede deshacer.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Sí, eliminar", key=f"yes_{model_name}", type="primary"):
            with st.spinner(f"Eliminando modelo {model_name}..."):
                
                delete_response = api_client.delete_model(model_name)
                
                if delete_response["status"] == "success":
                    st.success(f"✅ Modelo {model_name} eliminado correctamente")
                    st.experimental_rerun()
                else:
                    st.error(f"❌ Error eliminando el modelo: {delete_response.get('error', 'Error desconocido')}")
                    with st.expander("Detalles del error"):
                        st.json(delete_response)
    
    with col2:
        if st.button("❌ Cancelar", key=f"no_{model_name}"):
            st.info("Eliminación cancelada")

