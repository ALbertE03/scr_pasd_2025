import requests
import streamlit as st
import pandas as pd
import os
import time
import random
from typing import Dict, List, Union
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np 
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

    def start(self,training_params:dict):
        """
        Envía la solicitud de entrenamiento a la API de FastAPI.
        """
        try:
            df = training_params.get("df")
            if isinstance(df, pd.DataFrame):
                payload = training_params.copy()
                payload["dataset"] = df.to_dict(orient='records')
                del payload["df"] 
            else:
                raise ValueError("El parámetro 'df' debe ser un DataFrame de pandas.")

            response = self.session.post(
                f"{self.base_url}/train/oneDataset",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Error de conexión con la API: {str(e)}"}
        except Exception as e:
            return {"error": f"Error inesperado: {str(e)}"}

    def read(self, uploaded_file) -> pd.DataFrame:
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            
            response = self.session.post(  
                f"{self.base_url}/read/csv",
                files=files
            )
            response.raise_for_status()
            return pd.DataFrame(response.json()["data"])
        except Exception as e:
            print(f"Error en read: {str(e)}")
            return pd.DataFrame()

            
    def health_check(self) -> Dict:
        """Verifica el estado de salud de la API"""
        try:
            print(f"Verificando salud de API en: {self.base_url}/health")
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            print(f"Error en health_check: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_models(self) -> Dict:
        """Obtiene la lista de modelos disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=30)
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
    
    def predict(self, model_name: str, features: List[List[Union[int, float]]], features_name, return_probabilities: bool = False,) -> Dict:
        """Realiza predicciones usando un modelo"""
        try:
            
            payload = {
                "model_name": model_name,
                "features": features,
                "feature_names": features_name,
                "return_probabilities": return_probabilities
            }
            response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        

    def add_nodo(self, worker_name: str, add_cpu: int) -> Dict:
        """Agrega un nodo al cluster Ray llamando al endpoint /add/node"""
        try:
            payload = {"worker_name": worker_name, "add_cpu": add_cpu}
            response = self.session.post(f"{self.base_url}/add/node", params=payload, timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    def delete_nodo(self, worker_name: str) -> Dict:
        """Elimina un nodo del cluster Ray llamando al endpoint /remove/node"""
        try:
            params = {"node_name": worker_name}
            response = self.session.delete(f"{self.base_url}/remove/node", params=params, timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_all_ray_nodes(self) -> Dict:
        """Obtiene la lista de todos los nodos Ray ejecutándose actualmente"""
        try:
            params = {"command": "docker ps --filter 'name=ray' --format '{{.Names}}'"}
            response = self.session.get(f"{self.base_url}/cluster/nodes", params=params, timeout=30)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}


    def get_cluster_status(self) -> Dict:
        """Obtiene el estado del cluster"""
        try:
            response = self.session.get(f"{self.base_url}/cluster/status", timeout=30)
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

    def get_system_metrics(self):
        """Obtiene métricas del sistema desde el endpoint /system/status"""
        try:
            response = self.session.get(f"{self.base_url}/system/status", timeout=10)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    
def render_api_tab(api_client: APIClient):
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
    
def convert_str_to_dtype(dtype_str):
    """Convierte strings de tipos de datos de vuelta a dtypes de pandas/numpy"""
    dtype_mapping = {
        'float64': np.float64,
        'float32': np.float32,
        'int64': np.int64,
        'int32': np.int32,
        'bool': bool,
        'object': object,
        'category': 'category',
        'datetime64[ns]': 'datetime64[ns]',
        'timedelta64[ns]': 'timedelta64[ns]'
    }
    
    if dtype_str.startswith('Float64'):
        return pd.Float64Dtype()
    elif dtype_str.startswith('Int64'):
        return pd.Int64Dtype()
    elif dtype_str.startswith('boolean'):
        return pd.BooleanDtype()

    return dtype_mapping.get(dtype_str, object)

def render_explore_models_tab(api_client: APIClient):
    """Renderiza la pestaña de exploración de modelos"""
    st.subheader("🔍 Explorar Modelos Disponibles")    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Actualizar", key="refresh_models_api"):
            st.rerun()
    try:
        models_response = api_client.get_models()
    except:

        st.info("No hay modelos disponibles. Entrene algunos modelos primero.")
        return
    #st.write(models_response)
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
        
        
    filtered_models = {}
    for name, model in models.items():
        if model.get("dataset", "unknown") not in selected_datasets:
            continue

        if search_query and search_query.lower() not in name.lower():
            continue
        
        filtered_models[name] = model


    if filtered_models:
        cols = st.columns(2)
        for i, (model_name, model_info) in enumerate(filtered_models.items()):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"### {model_name}")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Dataset", model_info.get("dataset_name", "N/A"))

                        scores = model_info.get('scores', {})
                        if isinstance(scores, dict):
                            accuracy = scores.get('Accuracy', 0)
                        else:
                            accuracy = 0
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    with col_info2:
                        
                        scores = model_info.get('scores', {})
                        if isinstance(scores, dict):
                            training_time = scores.get('Training Time (s)', 0)
                        else:
                            training_time = 0
                        st.metric("Tiempo Entrenamiento", f"{training_time:.2f}s")
                        st.metric("Tamaño Archivo", f"{model_info.get('object_size_kb', 0):.2f} KB")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
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
    
    prediction_tabs = st.tabs(["📝 Predicción Individual"])#🚀 Predicciones Concurrentes"])#, "📁 Predicción en Lote"])
    
    with prediction_tabs[0]:
        render_individual_prediction(api_client, model_names, models)
    
    #with prediction_tabs[1]:
        #render_concurrent_predictions(api_client, model_names, models)
    
    #with prediction_tabs[2]:
        #render_batch_prediction(api_client, model_names)



def render_individual_prediction(api_client: APIClient, model_names: List[str], models: Dict):
    """Renderiza predicción individual dentro de un formulario"""
    st.subheader("📝 Predicción Individual")
    
    selected_model = st.selectbox("Seleccionar Modelo", model_names)    
    
    if not selected_model:
        return
    
    model_info = models[selected_model]
    dataset_name = model_info.get('dataset_name', '') or model_info.get('dataset', '')
    columns_to_exclude = model_info.get("columns_to_exclude", [])
    feature_names = model_info.get("columns", [])
    target = model_info.get('target')

    if feature_names:
        feature_names = [x for x in feature_names if x not in columns_to_exclude and x != target]
    
    data_types = [convert_str_to_dtype(dt) for dt in model_info.get('data_type', [])]
    n_features = max(0, len(feature_names))
    
    if n_features == 0:
        st.error("⚠️ No se encontraron características válidas para este modelo.")
        return
    
    scores = model_info.get('scores', {})
    accuracy = scores.get('Accuracy', 0) if isinstance(scores, dict) else 0
    
    st.info(f"""
    **Dataset:** {dataset_name}  
    **Características:** {n_features}
    """)


    prediction_mode = st.radio(
        "Modo de Predicción",
        ["Manual", "Archivo CSV"],
        help="Selecciona 'Manual' para introducir valores individuales o 'Archivo CSV' para cargar múltiples filas"
    )

    if prediction_mode == "Archivo CSV":
        render_batch_prediction_mode(api_client, selected_model, model_info, feature_names, data_types, n_features)
    else:
        render_manual_prediction_mode(api_client, selected_model, model_info, feature_names, data_types, n_features)


def render_batch_prediction_mode(api_client: APIClient, selected_model: str, model_info: Dict, feature_names: List[str], data_types: List, n_features: int):
    """Renderiza el modo de predicción por lotes desde archivo CSV"""
    st.subheader("📁 Predicción desde Archivo CSV")
    
    st.markdown("""
    **Instrucciones:**
    1. Sube un archivo CSV con las características para predicción
    2. El archivo debe tener exactamente **{n_features}** columnas
    3. Las columnas deben estar en el mismo orden que las características del modelo
    4. Se realizará una predicción por cada fila del archivo
    """.format(n_features=n_features))
    
    uploaded_file = st.file_uploader(
        "Seleccionar archivo CSV",
        type=['csv'],
        help=f"El archivo debe tener exactamente {n_features} columnas con las características del modelo"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            
            # Validar número de columnas
            if df.shape[1] != n_features:
                st.error(f"❌ Error: El archivo tiene {df.shape[1]} columnas, pero se esperan {n_features} columnas.")
                st.info("Por favor, asegúrate de que el archivo tenga la estructura correcta.")
                return
            st.success(f"✅ Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            # Mostrar preview del archivo
            st.subheader("Vista previa del archivo")
            st.dataframe(df.head(10), use_container_width=True)
            
            if df.shape[0] > 10:
                st.info(f"Mostrando las primeras 10 filas de {df.shape[0]} total.")
            
            validation_errors = []
            for i, (col_name, expected_dtype) in enumerate(zip(feature_names, data_types)):
                col_data = df.iloc[:, i]
                
                if pd.api.types.is_numeric_dtype(expected_dtype):
                    if not pd.api.types.is_numeric_dtype(col_data):
                        try:
                            pd.to_numeric(col_data, errors='raise')
                        except:
                            validation_errors.append(f"Columna ({col_name}): se esperan valores numéricos")
                
                elif pd.api.types.is_bool_dtype(expected_dtype):
                    unique_vals = col_data.unique()
                    if not all(val in [True, False, 'True', 'False', 'true', 'false', 1, 0] for val in unique_vals if pd.notna(val)):
                        validation_errors.append(f"Columna {i+1} ({col_name}): se esperan valores booleanos")
            
            #if validation_errors:
                #st.warning("⚠️ Se encontraron posibles problemas de validación:")
                #for error in validation_errors:
                    #st.write(f"- {error}")
                #st.info("Se intentará convertir automáticamente los tipos de datos durante la predicción.")
            
            col1, col2 = st.columns(2)
            return_probabilities = False
            with col2:
                max_rows = st.number_input("Máximo de filas a procesar", min_value=1, max_value=min(1000, df.shape[0]), value=min(100, df.shape[0]))
            
            if st.button("🔮 Realizar Predicciones", type="primary"):
                df_to_process = df.head(max_rows)
                
                with st.spinner(f"Realizando predicciones para {len(df_to_process)} filas..."):
                    try:
                        features_list = df_to_process.values.tolist()

                        prediction_response = api_client.predict(
                            selected_model, 
                            features_list,
                            feature_names,
                            return_probabilities
                        )
                        
                        if prediction_response["status"] == "error":
                            st.error(f"Error en predicción: {prediction_response['error']}")
                            st.info("Intenta verificar que los tipos de datos sean compatibles con el modelo.")
                        else:
                            result = prediction_response["data"]
                            # Validar que tengamos predicciones
                            if not result.get("predictions"):
                                st.error("No se obtuvieron predicciones del modelo")
                            elif len(result["predictions"]) == 0:
                                st.error("El modelo devolvió una lista vacía de predicciones")
                            else:
                                display_batch_prediction_results(result, df_to_process, return_probabilities)
                            
                    except Exception as e:
                        st.error(f"Error al procesar el archivo: {str(e)}")
                        st.info("Verifica que los datos del archivo sean compatibles con el modelo.")
        
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
            st.info("Asegúrate de que el archivo sea un CSV válido.")


def render_manual_prediction_mode(api_client: APIClient, selected_model: str, model_info: Dict, feature_names: List[str], data_types: List, n_features: int):
    """Renderiza el modo de predicción manual"""
    example_features = []
    for i, dt in enumerate(data_types):
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i+1}"
        
        if pd.api.types.is_integer_dtype(dt):
            example = random.randint(0, 100)
        elif pd.api.types.is_float_dtype(dt):
            example = round(random.uniform(0, 1), 2)
        elif pd.api.types.is_bool_dtype(dt):
            example = random.choice([True, False])
        elif pd.api.types.is_datetime64_any_dtype(dt):
            example = pd.Timestamp.now() + pd.Timedelta(days=random.randint(-30, 30))
        elif pd.api.types.is_categorical_dtype(dt):
            example = random.choice(dt.categories) if hasattr(dt, 'categories') else f"Categoria_{random.randint(1, 5)}"
        else:
            example = f"Valor_{random.randint(1, 100)}"
        
        example_features.append(example)

    
    with st.form(key='prediction_form'):
        st.subheader("Introducir Características")
        features = []
        
        if n_features <= 8:
            num_cols = min(max(n_features, 1), 4)
            cols = st.columns(num_cols)
            for i in range(n_features):
                with cols[i % num_cols]:
                    feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
                    key = f"feature_{i}_{selected_model}"
                    
                    if pd.api.types.is_numeric_dtype(data_types[i]):
                        step = 1 if pd.api.types.is_integer_dtype(data_types[i]) else 0.1
                        value = st.number_input(
                            feature_name,
                            value=float(example_features[i]),
                            step=float(step),
                            key=key,
                            format="%.4f"
                        )
                        if pd.api.types.is_integer_dtype(data_types[i]):
                            value = int(value)

                    elif pd.api.types.is_bool_dtype(data_types[i]):
                        value = st.selectbox(
                            feature_name,
                            options=[True, False],
                            index=int(example_features[i]),
                            format_func=lambda x: "Verdadero" if x else "Falso",
                            key=key
                        )
                    
                    elif pd.api.types.is_datetime64_any_dtype(data_types[i]):
                        value = st.date_input(
                            feature_name,
                            value=pd.to_datetime(example_features[i]),
                            key=key
                        )
                        value = pd.to_datetime(value)

                    elif pd.api.types.is_categorical_dtype(data_types[i]):
                        categories = data_types[i].categories if hasattr(data_types[i], 'categories') else []
                        value = st.selectbox(
                            feature_name,
                            options=categories if categories else [str(example_features[i])],
                            index=0,
                            key=key
                        )
                    else:
                        value = st.text_input(
                            feature_name,
                            value=str(example_features[i]),
                            key=key
                        )
                    
                    features.append(value)
        else:
            tab_groups = [(f"Grupo {i+1}", i*8, min((i+1)*8, n_features)) for i in range((n_features + 7) // 8)]
            tabs = st.tabs([group[0] for group in tab_groups])
            
            for tab, (_, start, end) in zip(tabs, tab_groups):
                with tab:
                    cols = st.columns(2)
                    for i in range(start, end):
                        with cols[(i - start) % 2]:
                            feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
                            key = f"feature_{i}_{selected_model}"
                            
                            if pd.api.types.is_numeric_dtype(data_types[i]):
                                step = 1 if pd.api.types.is_integer_dtype(data_types[i]) else 0.1
                                value = st.number_input(
                                    feature_name,
                                    value=float(example_features[i]),
                                    step=step,
                                    key=key
                                )
                                if pd.api.types.is_integer_dtype(data_types[i]):
                                    value = int(value)
                            
                            elif pd.api.types.is_bool_dtype(data_types[i]):
                                value = st.selectbox(
                                    feature_name,
                                    options=[True, False],
                                    index=int(example_features[i]),
                                    key=key
                                )
                            
                            elif pd.api.types.is_datetime64_any_dtype(data_types[i]):
                                value = st.date_input(
                                    feature_name,
                                    value=pd.to_datetime(example_features[i]),
                                    key=key
                                )
                                value = pd.to_datetime(value)
                            
                            elif pd.api.types.is_categorical_dtype(data_types[i]):
                                categories = data_types[i].categories if hasattr(data_types[i], 'categories') else []
                                value = st.selectbox(
                                    feature_name,
                                    options=categories if categories else [str(example_features[i])],
                                    index=0,
                                    key=key
                                )
                            else:
                                value = st.text_input(
                                    feature_name,
                                    value=str(example_features[i]),
                                    key=key
                                )
                            
                            if i >= len(features):
                                features.append(value)
                            else:
                                features[i] = value
        submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary")
        
        if submitted:
            with st.spinner("Realizando predicción..."):
                try:
                    prediction_response = api_client.predict(
                        selected_model, 
                        [features],
                        feature_names,
                        False
                    )
                    
                    if prediction_response["status"] == "error":
                        st.error(f"Error en predicción: {prediction_response['error']}")
                    else:
                        result = prediction_response["data"]
                        display_prediction_results(result, False) 
                        
                except Exception as e:
                    st.error(f"Error al conectar con el servidor: {str(e)}")


def display_batch_prediction_results(result, input_df, show_probabilities):
    """Muestra los resultados de predicciones en lote"""
    try:
        # Validar que tenemos los datos necesarios
        if not result or not result.get("predictions"):
            st.error("❌ No se recibieron predicciones válidas del modelo")
            return
        
        predictions = result["predictions"]
        if not predictions or len(predictions) == 0:
            st.error("❌ El modelo devolvió una lista vacía de predicciones")
            return
        
        st.success("✅ Predicciones completadas exitosamente")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predicciones", len(predictions))
        
        with col2:
            st.metric("Tiempo Total", f"{result.get('prediction_time', 0):.3f}s")
        
        with col3:
            avg_time = result.get('prediction_time', 0) / len(predictions) if predictions else 0
            st.metric("Tiempo Promedio", f"{avg_time:.4f}s/predicción")
        
        with col4:
            st.metric("Características", result.get("feature_count", 0))
        
        # Crear DataFrame con resultados
        results_df = input_df.copy()
        
        # Verificar que el número de predicciones coincida con el número de filas de entrada
        num_predictions = len(predictions)
        num_input_rows = len(input_df)
        
        if num_predictions != num_input_rows:
            st.warning(f"⚠️ Número de predicciones ({num_predictions}) no coincide con número de filas de entrada ({num_input_rows})")
            
            # Si hay menos predicciones que filas de entrada, ajustar el DataFrame
            if num_predictions < num_input_rows:
                results_df = input_df.iloc[:num_predictions].copy()
                st.info(f"Mostrando solo las primeras {num_predictions} filas con predicciones exitosas")
            # Si hay más predicciones que filas de entrada (poco probable), truncar predicciones
            else:
                predictions_to_use = predictions[:num_input_rows]
                result["predictions"] = predictions_to_use
                if "probabilities" in result and result["probabilities"]:
                    result["probabilities"] = result["probabilities"][:num_input_rows]
        
        results_df['Predicción'] = predictions
        
        if show_probabilities and "probabilities" in result and result["probabilities"]:
            # Verificar que las probabilidades también coincidan
            if len(result["probabilities"]) == len(results_df):
                # Si hay probabilidades, agregar la probabilidad máxima
                max_probs = [max(prob) for prob in result["probabilities"]]
                results_df['Confianza'] = [f"{prob:.4f}" for prob in max_probs]
            else:
                st.warning("⚠️ No se pueden mostrar probabilidades debido a discrepancia en dimensiones")
        
        st.subheader("Resultados de Predicciones")
        st.dataframe(results_df, use_container_width=True)
        
        # Opción para descargar resultados
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar resultados",
            data=csv,
            file_name=f"predicciones_{result.get('model_name', 'modelo')}.csv",
            mime="text/csv"
        )
        
        # Mostrar estadísticas de las predicciones
        if predictions:
            st.subheader("Estadísticas de Predicciones")
            predictions_series = pd.Series(predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribución de predicciones:**")
                value_counts = predictions_series.value_counts().sort_index()
                st.bar_chart(value_counts)
            
            with col2:
                st.write("**Resumen estadístico:**")
                if predictions_series.dtype in ['int64', 'float64']:
                    st.write(predictions_series.describe())
                else:
                    st.write(f"Valores únicos: {predictions_series.nunique()}")
                    st.write(f"Valor más frecuente: {predictions_series.mode().iloc[0] if not predictions_series.mode().empty else 'N/A'}")
                    st.write(f"Total de predicciones: {len(predictions_series)}")
    
    except Exception as e:
        st.error(f"❌ Error mostrando resultados de predicción: {str(e)}")
        st.info("Error técnico: Por favor, verifica los datos de entrada y vuelve a intentar")
        # Mostrar información de debug si hay datos disponibles
        if result:
            st.write("**Información de debug:**")
            st.write(f"- Predicciones recibidas: {len(result.get('predictions', []))}")
            st.write(f"- Filas de entrada: {len(input_df)}")
            st.write(f"- Tipo de predicciones: {type(result.get('predictions', []))}")
            if result.get('predictions'):
                st.write(f"- Primer elemento: {result['predictions'][0] if result['predictions'] else 'N/A'}")
                st.write(f"- Tipo del primer elemento: {type(result['predictions'][0]) if result['predictions'] else 'N/A'}")

def display_prediction_results(result, show_probabilities):
    """Muestra los resultados de la predicción"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicción", result["predictions"][0])
    
    with col2:
        st.metric("Tiempo", f"{result['prediction_time']:.3f}s")
    
    with col3:
        st.metric("Características", result["feature_count"])
         

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
    # Crear un diccionario de nombres truncados para el selectbox
    truncated_model_names = ["Todos"] + [truncate_model_name(name, 30) for name in model_names]
    model_name_mapping = {"Todos": "Todos"}
    for i, name in enumerate(model_names):
        model_name_mapping[truncated_model_names[i+1]] = name
    
    selected_display_name = st.selectbox("Seleccionar modelo:", truncated_model_names)
    selected_model = model_name_mapping[selected_display_name]

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
                    truncated_name = truncate_model_name(model_name, 15)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=prediction_times,
                        mode='lines+markers',
                        name=truncated_name,
                        hovertemplate=f'<b>{model_name}</b><br>' +
                                    'Tiempo: %{x}<br>' +
                                    'Predicción: %{y:.3f}s<extra></extra>',
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
            fig.update_layout(
                title="Tiempo de Predicción por Modelo",
                xaxis_title="Tiempo",
                yaxis_title="Tiempo de Predicción (s)",
                hovermode='closest',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10)
                ),
                margin=dict(r=150)  # Espacio adicional para la leyenda
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
            # Create a cumulative predictions chart
            fig = go.Figure()
            for model_name, entries in raw_stats.items():
                if entries:
                    # Sort entries by timestamp
                    sorted_entries = sorted(entries, key=lambda x: datetime.fromisoformat(x["timestamp"]))
                    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in sorted_entries]
                    
                    # Calculate cumulative predictions
                    cumulative_predictions = []
                    running_total = 0
                    for entry in sorted_entries:
                        running_total += 1  # Count each prediction event
                        cumulative_predictions.append(running_total)
                    
                    truncated_name = truncate_model_name(model_name, 15)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=cumulative_predictions,
                        mode='lines',
                        name=truncated_name,
                        hovertemplate=f'<b>{model_name}</b><br>' +
                                    'Tiempo: %{x}<br>' +
                                    'Predicciones: %{y}<extra></extra>',
                        line=dict(width=2),
                    ))
            fig.update_layout(
                title="Predicciones Acumuladas por Modelo",
                xaxis_title="Tiempo",
                yaxis_title="Total Predicciones",
                hovermode='closest',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10)
                ),
                margin=dict(r=150)  # Espacio adicional para la leyenda
            )
        else:
            if selected_model in raw_stats and raw_stats[selected_model]:
                entries = raw_stats[selected_model]
                # Sort entries by timestamp
                sorted_entries = sorted(entries, key=lambda x: datetime.fromisoformat(x["timestamp"]))
                timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in sorted_entries]
                
                # Calculate cumulative predictions and samples
                cumulative_predictions = []
                cumulative_samples = []
                running_pred_total = 0
                running_sample_total = 0
                
                for entry in sorted_entries:
                    running_pred_total += 1  # Count each prediction event
                    running_sample_total += entry["n_samples"]
                    cumulative_predictions.append(running_pred_total)
                    cumulative_samples.append(running_sample_total)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cumulative_predictions,
                    mode='lines',
                    name="Predicciones",
                    line=dict(width=2, color='blue'),
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cumulative_samples,
                    mode='lines',
                    name="Muestras",
                    line=dict(width=2, color='red', dash='dash'),
                ))
                
                fig.update_layout(
                    title=f"Predicciones y Muestras Acumuladas - {selected_model}",
                    xaxis_title="Tiempo",
                    yaxis_title="Total",
                    legend=dict(orientation="h")
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="No hay datos disponibles", x=0.5, y=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    if selected_model == "Todos" and aggregated_stats:
        st.subheader("Comparativa de Modelos")
        
        bubble_data = []
        for model_name, agg in aggregated_stats.items():
            total_predictions = agg.get("total_predictions", 0)
            avg_time = agg.get("avg_prediction_time", 0)
            total_samples = agg.get("total_samples", 0)
            accuracy = agg.get("accuracy", 0) or 0
            
            if total_predictions > 0:
                bubble_data.append({
                    "Modelo": model_name,
                    "Tiempo Promedio (s)": avg_time,
                    "Total Predicciones": total_predictions,
                    "Accuracy": accuracy,
                    "Muestras": total_samples
                })
        
        if bubble_data:
            df_bubble = pd.DataFrame(bubble_data)
            # Agregar columna con nombres truncados para visualización
            df_bubble['Modelo_Truncado'] = df_bubble['Modelo'].apply(lambda x: truncate_model_name(x, 12))
            
            fig = px.scatter(
                df_bubble,
                x="Tiempo Promedio (s)",
                y="Accuracy", 
                size="Total Predicciones",
                color="Modelo_Truncado",
                hover_name="Modelo",  # Nombre completo en el hover
                text="Modelo_Truncado",
                size_max=60,
                title="Comparativa de Eficiencia y Precisión entre Modelos",
                hover_data={
                    "Modelo_Truncado": False,  # No mostrar en hover
                    "Total Predicciones": True,
                    "Muestras": True
                }
            )
            
            fig.update_layout(
                xaxis_title="Tiempo Promedio de Predicción (s)",
                yaxis_title="Accuracy",
                height=500,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10),
                    title="Modelos"
                ),
                margin=dict(r=180)  # Más espacio para la leyenda
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=2, color='DarkSlateGrey')),
                textfont=dict(size=9)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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

def render_concurrent_predictions(api_client: APIClient, model_names: List[str], models: Dict):
    """Renderiza la interfaz para predicciones concurrentes"""
    st.subheader("🚀 Predicciones Concurrentes")
    
    st.markdown("""
    <div style="background-color: rgba(0, 180, 216, 0.1); padding: 15px; border-radius: 5px; border-left: 4px solid #00b4d8;">
        <h4 style="margin-top:0">⚡ Realizar múltiples predicciones en paralelo</h4>
        <p>Envía varias solicitudes de predicción simultáneamente para evaluar el rendimiento y capacidad de la API.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_model = st.selectbox("Seleccionar Modelo", model_names, key="concurrent_model")
    with col2:
        return_probabilities = st.checkbox("Incluir Probabilidades", value=False, key="concurrent_probs")
    
    if selected_model:
        model_info = models[selected_model]
        st.info(f"Dataset: {model_info.get('dataset', 'desconocido')} | Accuracy: {model_info.get('accuracy', 0):.4f}")
        col1, col2 = st.columns([1, 1])
        with col1:
            num_requests = st.slider("Número de Solicitudes Concurrentes", 2, 50, 10, key="concurrent_requests")
        with col2:
            mode = st.radio(
                "Modo de concurrencia", 
                ["Una muestra por solicitud", "Múltiples muestras por solicitud"],
                index=0,
                key="concurrency_mode",
                help="'Una muestra por solicitud' envía N solicitudes independientes cada una con un solo vector (máxima paralelización). 'Múltiples muestras por solicitud' envía N solicitudes cada una con varios vectores."
            )
    
        dataset = model_info.get('dataset', 'iris')
        if dataset == 'iris':
            sample_vector = [5.84, 3.05, 3.76, 1.20]
        elif dataset == 'breast_cancer':
            
            sample_vector = [14.0, 14.0, 91.0, 654.0, 0.1, 0.1, 0.1, 0.05, 0.2, 0.07] + [0.5] * 20
        elif dataset == 'digits':
            sample_vector = [0.0] * 64
        else:
            n_features = model_info.get('n_features', 4)
            sample_vector = [1.0] * n_features
        
        display_features = st.slider(
            "Número de Características",1,65,1,key='asdww')
        sample_vector = [1.0] * display_features
        with st.expander("Personalizar muestra base (opcional)"):
            st.caption("Personaliza algunos valores de la muestra base que serán ligeramente variados para las predicciones concurrentes")
            
            cols = st.columns(2)
            
            for i in range(display_features):
                with cols[i % 2]:
                    sample_vector[i] = st.number_input(
                        f"Característica {i+1}",
                        value=float(sample_vector[i]),
                        format="%.2f",
                        step=0.1,
                        key=f"concurrent_feature_{i}"
                    )
        if mode == "Una muestra por solicitud":
            samples_per_request = 1
            st.info(f"Se realizarán {num_requests} solicitudes simultáneas, cada una con un solo vector de características ({num_requests} muestras en total)")
        else:
            samples_per_request = st.slider("Muestras por Solicitud", 2, 100, 10, key="samples_per_request")
            st.info(f"Se realizarán {num_requests} solicitudes simultáneas con {samples_per_request} muestras cada una ({num_requests * samples_per_request} muestras en total)")
        
        if st.button("🚀 Iniciar Predicciones Concurrentes", type="primary", key="start_concurrent"):
            
            features_batches = []
            for i in range(num_requests):
                batch = []
                for j in range(samples_per_request):
                    variation = np.random.uniform(0, 15, len(sample_vector))
                    sample = [max(0, a * b) for a, b in zip(sample_vector, variation)]
                    batch.append(sample)
                features_batches.append(batch)
            
            with st.spinner(f"Procesando {num_requests * samples_per_request} predicciones concurrentes..."):
                start_time = time.time()
                
                result = api_client.predict_concurrent(selected_model, features_batches, return_probabilities)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if result["status"] == "error":
                    st.error(f"Error en predicciones concurrentes: {result['error']}")
                else:
                    data = result["data"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Muestras", data.get("n_samples", num_requests * samples_per_request))
                    
                    with col2:
                        st.metric("Tiempo Total", f"{total_time:.3f}s")
                    
                    with col3:
                        throughput = data.get("n_samples", 0) / max(0.001, total_time)
                        st.metric("Rendimiento", f"{throughput:.1f} muestras/s")
                    
                    st.subheader("Resultados de Predicciones Concurrentes")

                    total_predictions = len(data.get("predictions", []))
                    
                    if total_predictions > 0:
                        from collections import Counter
                        class_counts = Counter(data["predictions"])

                        df_counts = pd.DataFrame({
                            "Clase": list(class_counts.keys()),
                            "Cantidad": list(class_counts.values())
                        })

                        fig = px.bar(
                            df_counts,
                            x="Clase",
                            y="Cantidad",
                            title="Distribución de Clases Predichas",
                            color="Clase",
                            text="Cantidad"
                        )
                        fig.update_layout(xaxis_title="Clase Predicha", yaxis_title="Cantidad")
                        st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Estadísticas de Rendimiento")
                    
                    if mode == "Una muestra por solicitud":
                        df_performance = pd.DataFrame({
                            "Métrica": [
                                "Tiempo Total de Procesamiento",
                                "Tiempo Promedio por Predicción",
                                "Predicciones por Segundo (Throughput)",
                                "Paralelismo Efectivo",
                                "Latencia Promedio"
                            ],
                            "Valor": [
                                f"{total_time:.3f} s",
                                f"{data.get('avg_prediction_time', total_time/num_requests):.3f} s",
                                f"{num_requests/total_time:.2f}",
                                f"{data.get('avg_prediction_time', total_time/num_requests)*num_requests/max(0.001, total_time):.2f}",
                                f"{data.get('avg_prediction_time', total_time/num_requests):.3f} s"
                            ]
                        })
                    else:
                        df_performance = pd.DataFrame({
                            "Métrica": [
                                "Tiempo Total de Procesamiento",
                                "Tiempo Promedio por Solicitud",
                                "Tiempo Promedio por Muestra",
                                "Solicitudes por Segundo",
                                "Muestras por Segundo"
                            ],
                            "Valor": [
                                f"{total_time:.3f} s",
                                f"{data.get('avg_prediction_time', total_time/num_requests):.3f} s",
                                f"{data.get('avg_time_per_sample', total_time/(num_requests*samples_per_request)):.4f} s",
                                f"{num_requests/total_time:.2f}",
                                f"{(num_requests*samples_per_request)/total_time:.2f}"
                            ]
                        })
                    
                    st.dataframe(df_performance, use_container_width=True)

                    with st.expander("Ver muestra de predicciones"):
                        sample_size = min(20, len(data.get("predictions", [])))
                        sample_data = []
                        
                        for i in range(sample_size):
                            row = {"#": i+1, "Predicción": data["predictions"][i]}
                            
                            if "probabilities" in data and i < len(data["probabilities"]):
                                probs = data["probabilities"][i]
                                for j, prob in enumerate(probs):
                                    row[f"Prob. Clase {j}"] = f"{prob:.4f}"
                            
                            sample_data.append(row)
                        
                        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

def truncate_model_name(model_name: str, max_length: int = 20) -> str:
    """Trunca nombres de modelos largos para mejorar la visualización en gráficos"""
    if len(model_name) <= max_length:
        return model_name
    

    if "_" in model_name:
        parts = model_name.split("_")

        model_type = parts[0]
        if len(model_type) > max_length:
            return model_type[:max_length-3] + "..."
        
        remaining_length = max_length - len(model_type) - 4 
        if remaining_length > 0:
            dataset_part = "_".join(parts[1:])
            if len(dataset_part) > remaining_length:
                return model_type + "_" + dataset_part[:remaining_length] + "..."

