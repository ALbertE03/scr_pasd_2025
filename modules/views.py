import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import random

def run_training_in_thread(training_params, metrics, api_client):
    """Ejecuta el entrenamiento en un hilo separado """
    try:
        st.session_state.train = True
        response = api_client.start(training_params)
        
        print(f"API Response: {response}")
        
        if 'error' in response:
            st.session_state['train_result'] = None
            st.session_state['train_error'] = f"Error con la API: {response['error']}"
        elif 'data' in response:
            data = response['data']
            
            # Aceptar tanto listas como diccionarios
            valid_structure = False
            
            if isinstance(data, list):
                # Verificar que sea una lista de modelos válidos
                valid_structure = all('model_name' in item and 'scores' in item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                # Verificar que sea un diccionario de modelos válidos
                valid_structure = all('model_name' in item and 'scores' in item for item in data.values() if isinstance(item, dict))
            
            if valid_structure:
                st.session_state['train_result'] = data
                st.session_state['train_metrics'] = metrics
                st.session_state['train_error'] = None
            else:
                st.session_state['train_result'] = None
                st.session_state['train_error'] = f"Estructura de datos inesperada en la respuesta: {type(data)}"
                print(f"Invalid data structure: {data}")
        else:
            st.session_state['train_result'] = None
            st.session_state['train_error'] = "Respuesta inesperada de la API - no hay 'data' ni 'error'"
            print(f"Unexpected response structure: {response}")
            
    except Exception as e:
        st.session_state['train_result'] = None
        st.session_state['train_error'] = f"❌ Error al enviar el trabajo al clúster: {str(e)}"
        print(f"Exception in run_training_in_thread: {e}")
    finally:
        st.session_state['training_loading'] = False

classification_only_models = [

            # Modelos basados en árboles
            "RandomForest", 
            "GradientBoosting", 
            "AdaBoost",
            "ExtraTrees",
            "DecisionTree", 
            
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

regression_only_models=['Regresion lineal']
def render_training_tab(cluster_status,api_client):
    """Renderiza la pestaña de entrenamiento con capacidades avanzadas"""
    st.header("🧠 Entrenamiento Distribuido")
    
    if not cluster_status['connected']:
        st.warning("Debes conectarte al cluster para ejecutar entrenamientos")
        return
    
    training_tabs = st.tabs([
        "🚀 Entrenamiento",
        "🚀Entrenamiento Avanzado"
    ])
    
    with training_tabs[0]:
        render_advanced_training(cluster_status,api_client)
    
    with training_tabs[1]:
        render(cluster_status,api_client)



def render(cluster_status, api_client):
    st.subheader("🚀 Entrenamiento Distribuido Avanzado")
    st.markdown("""
    <div class="success-card">
        <h4>✅ Procesamiento de múltiples modelos y datasets en paralelo</h4>
        <p>Entrene varios modelos y datasets simultáneamente aprovechando la potencia del cluster distribuido</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:

        ## cambiar todo esto y poner un que pueda cargar cualquier csv y mandarlo a la api con un endpind que hay que crear 
        datasets = ['iris', 'wine', 'breast_cancer', 'digits']
        selected_dataset = st.multiselect(
            "Dataset",
            options=datasets,
            default=datasets[:2],
            key="advanced_dataset_select_pro"
        )
        st.session_state.current_dataset = selected_dataset
        for  i in selected_dataset:
            st.session_state.data_test_size[i]=0.3
            
    
        selected_models = st.multiselect(
            "Modelos a entrenar en paralelo",
            options=classification_only_models,
            default=st.session_state.selected_models[:4] if hasattr(st.session_state, 'selected_models') else classification_only_models[:4],
            key="advanced_models_multiselect_pro"
        )
        
        st.session_state.selected_models = selected_models
    with col2:
        r = st.selectbox('',options=selected_dataset,key='SCR',help='por defecto todos son 0.3')
        d = datasets[datasets.index(r)]
        st.session_state.data_test_size[d]= st.slider(
            "% Datos de prueba",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.test_size,
            step=0.05,
            key="advanced_test_size_slider_pro"
        )
    with st.expander("⚙️ Configuración de Hiperparámetros"):
        st.caption("Configure hiperparámetros específicos para cada modelo seleccionado")
        
        hyperparams = {}
        
        for model in selected_models:
            st.subheader(f"{model}")
            cols = st.columns(3)
            
            if model == "RandomForest":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de árboles ({model})", 10, 200, 100, 10,key='1'),
                    "max_depth": cols[1].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1,key='2'),
                    "min_samples_split": cols[2].slider(f"Min muestras para split ({model})", 2, 10, 2, 1,key='3')
                }
            elif model == "GradientBoosting":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 10, 200, 100, 10,key='4'),
                    "learning_rate": cols[1].slider(f"Tasa de aprendizaje ({model})", 0.01, 0.3, 0.1, 0.01,key='5'),
                    "max_depth": cols[2].slider(f"Profundidad máxima ({model})", 2, 10, 3, 1,key='6')
                }
            elif model == "SVM":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1,key='7'),
                    "kernel": cols[1].selectbox(f"Kernel ({model})", ["linear", "rbf", "poly"], 1,key='8'),
                    "gamma": cols[2].selectbox(f"Gamma ({model})", ["scale", "auto"], 0,key='9')
                }
            elif model == "LogisticRegression":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1,key='10'),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 1000, 100, 100,key='11'),
                    "solver": cols[2].selectbox(f"Solver ({model})", ["lbfgs", "liblinear", "newton-cg"], 0,key='12')
                }            
            elif model == "AdaBoost":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 50, 200, 100, 10,key='13'),
                    "learning_rate": cols[1].slider(f"Tasa de aprendizaje ({model})", 0.01, 1.0, 0.1, 0.01,key='14'),
                    "algorithm": cols[2].selectbox(f"Algoritmo ({model})", ["SAMME"], 0,key='15')
                }
            elif model == "ExtraTrees":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de árboles ({model})", 10, 200, 100, 10,key='16'),
                    "max_depth": cols[1].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1,key='17'),
                    "min_samples_split": cols[2].slider(f"Min muestras para split ({model})", 2, 10, 2, 1,key='18')
                }
            elif model == "KNN":
                hyperparams[model] = {
                    "n_neighbors": cols[0].slider(f"Número de vecinos ({model})", 1, 20, 5, 1,key='19'),
                    "weights": cols[1].selectbox(f"Pesos ({model})", ["uniform", "distance"], 0,key='20'),
                    "algorithm": cols[2].selectbox(f"Algoritmo ({model})", ["auto", "ball_tree", "kd_tree", "brute"], 0,key='21')
                }
            elif model == "DecisionTree":
                hyperparams[model] = {
                    "max_depth": cols[0].slider(f"Profundidad máxima ({model})", 2, 20, 10, 1,key='22'),
                    "min_samples_split": cols[1].slider(f"Min muestras para split ({model})", 2, 10, 2, 1,key='23'),
                    "criterion": cols[2].selectbox(f"Criterio ({model})", ["gini", "entropy"], 0,key='24')
                }
            elif model == "SGD":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.0001, 0.1, 0.0001, 0.0001, format="%.4f",key='25'),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 2000, 1000, 100,key='26'),
                    "tol": cols[2].slider(f"Tolerancia ({model})", 1e-5, 1e-2, 1e-3, format="%.5f",key='29')
                }
            elif model == "LinearSVM":
                hyperparams[model] = {
                    "C": cols[0].slider(f"Parámetro C ({model})", 0.1, 10.0, 1.0, 0.1,key='30'),
                    "max_iter": cols[1].slider(f"Max iteraciones ({model})", 100, 1000, 1000, 100,key='31'),
                    "tol": cols[2].slider(f"Tolerancia ({model})", 1e-5, 1e-2, 1e-3, format="%.5f",key='32')
                }
            elif model == "GaussianNB":
                hyperparams[model] = {
                    "var_smoothing": cols[0].slider(f"Suavizado de varianza ({model})", 1e-12, 1e-6, 1e-9, format="%.2e",key='33')
                }
            elif model == "BernoulliNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01,key='34'),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True,key='35')
                }
            elif model == "MultinomialNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01,key='36'),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True,key='37')
                }
            elif model == "ComplementNB":
                hyperparams[model] = {
                    "alpha": cols[0].slider(f"Alpha ({model})", 0.01, 2.0, 1.0, 0.01,key='38'),
                    "fit_prior": cols[1].checkbox(f"Ajustar prior ({model})", True,key='39')
                }
            elif model == "LDA":
                hyperparams[model] = {
                    "solver": cols[0].selectbox(f"Solver ({model})", ["svd", "lsqr", "eigen"], 0,key='40'),
                    "shrinkage": cols[1].slider(f"Shrinkage ({model})", None, 1.0, None,key='41') if cols[1].checkbox(f"Usar shrinkage ({model})", False,key='42') else None
                }
            elif model == "QDA":
                hyperparams[model] = {
                    "reg_param": cols[0].slider(f"Parámetro de regularización ({model})", 0.0, 1.0, 0.0, 0.01,key='43')
                }
            elif model == "MLP":
                hyperparams[model] = {
                    "hidden_layer_sizes": (cols[0].slider(f"Neuronas capa oculta ({model})", 10, 200, 100, 10,key='44')),
                    "activation": cols[1].selectbox(f"Activación ({model})", ["relu", "tanh", "logistic"], 0,key='45'),
                    "max_iter": cols[2].slider(f"Max iteraciones ({model})", 100, 500, 200, 50,key='46')
                }
            elif model == "Bagging":
                hyperparams[model] = {
                    "n_estimators": cols[0].slider(f"Número de estimadores ({model})", 5, 50, 10, 5,key='47'),
                    "max_samples": cols[1].slider(f"Max muestras ({model})", 0.1, 1.0, 1.0, 0.1,key='48'),
                    "bootstrap": cols[2].checkbox(f"Bootstrap ({model})", True,key='49')
                }
            elif model == "Voting":
                hyperparams[model] = {
                    "voting": cols[0].selectbox(f"Tipo de votación ({model})", ["hard", "soft"], 1,key='50')
                }
                
    co1, co2, co3 = st.columns([1,2,1])
    with co2:
        start_training = st.button(
            "🚀 Iniciar Entrenamiento Distribuido", 
            type="primary",
            key="advanced_start_training_button_pro",
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
           
def render_advanced_training(cluster_status, api_client):
    """Renderiza la interfaz de entrenamiento con selección manual del tipo de problema"""
    st.subheader("🚀 Entrenamiento Distribuido")
    
    st.markdown("""
    <div class="success-card">
        <h4>✅ Procesamiento de múltiples modelos en paralelo</h4>
        <p>Entrene varios modelos simultáneamente aprovechando la potencia del cluster distribuido</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'training_loading' not in st.session_state:
        st.session_state.training_loading = False
    if 'train_result' not in st.session_state:
        st.session_state.train_result = None
    if 'train_error' not in st.session_state:
        st.session_state.train_error = None
    

    col1, col2 = st.columns([3, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Subir archivo CSV", 
            type=["csv"],
            key="advanced_dataset_upload"
        )
   
    if uploaded_file is not None:
        try:
            df = api_client.read(uploaded_file)
            
            if df.empty:
                st.error("❌ El archivo CSV está vacío o no se pudo leer correctamente.")
                st.stop()
            
            
            with col1:
                st.markdown(f"""
                **📊 Estadísticas del Dataset:**
                - 📝 Registros: {df.shape[0]:,}
                - 🔠 Características: {df.shape[1]}
                - 🕵 Valores faltantes: {df.isna().sum().sum()}
                """)
                
                with st.expander("🔍 Vista previa del dataset (primeras 10 filas)"):
                    st.dataframe(df.head(10))
            
            
            with st.form("training_form", clear_on_submit=False):
                col_form1, col_form2 = st.columns([3, 2])
                
                with col_form1:
                   
                    with st.expander("🧹 Manejo de Valores Faltantes", expanded=True):
                        missing_strategy = st.selectbox(
                            "Estrategia para valores faltantes:",
                            options=[
                                "ninguna",
                                "Eliminar filas con valores faltantes",
                                "Rellenar con la media/moda",
                                "Rellenar con valor específico"
                            ],
                            key="missing_strategy"
                        )
                        
                        fill_value = "0"
                        if missing_strategy == "Rellenar con valor específico":
                            fill_value = st.text_input("Valor de relleno:", "0")
                    
                    
                    target_column = st.selectbox(
                        "🎯 Seleccione la columna target (variable objetivo)",
                        options=df.columns,
                        index=len(df.columns)-1,
                        key="target_column_select"
                    )
                    
                    auto_problem_type = "Clasificación"
                    classification_subtype = None
                    
                    if target_column:
                        target_series = df[target_column]
                        unique_values = target_series.nunique()
        
                        if pd.api.types.is_numeric_dtype(target_series):
                            auto_problem_type = "Regresión" if unique_values > 10 else "Clasificación"
                        else:
                            auto_problem_type = "Clasificación"
                        
                        if auto_problem_type == "Clasificación":
                            classification_subtype = "Binaria" if unique_values == 2 else "Multiclase"
                        
                        detection_msg = f"""
                        🔍 **Tipo de problema detectado:** {auto_problem_type}
                        """
                        if classification_subtype:
                            detection_msg += f" - Subtipo: {classification_subtype} ({unique_values} clases)"
                        
                        st.markdown(detection_msg)
                        
                        with st.expander("⚙️ Opciones avanzadas de preprocesamiento", expanded=True):
                            features_to_exclude = st.multiselect(
                                "🚫 Excluir columnas del modelo:",
                                options=[col for col in df.columns if col != target_column],
                                key="features_to_exclude"
                            )
                            
                            transform_target = False
                            if auto_problem_type == "Regresión":
                                transform_target = st.checkbox(
                                    "Transformar target (logarítmico)",
                                    help="Útil para distribuciones sesgadas"
                                )
                
                with col_form2:
                    if target_column:
                        st.markdown("## ⚙️ Configuración del Modelado")
                        
                    
                        problem_type = st.radio(
                            "Seleccione el tipo de problema:",
                            options=["Clasificación", "Regresión"],
                            index=0 if auto_problem_type == "Clasificación" else 1,
                            key="problem_type_selection"
                        )
                    
                        if problem_type == "Clasificación" and classification_subtype:
                            st.info(f"🔮 Problema de clasificación {classification_subtype.lower()} detectado")
                        
                    
                        test_size = st.slider(
                            "📊 % para datos de prueba",
                            min_value=0.1,
                            max_value=0.5,
                            value=0.2,
                            step=0.05,
                            key="test_size_slider"
                        )
                        
                        random_state = st.number_input(
                            "🔢 Semilla aleatoria (random state)",
                            min_value=0,
                            value=42,
                            key="random_state_input"
                        )
                        
                        if problem_type == "Clasificación":   
                            available_models = classification_only_models
                        else:
                            available_models = regression_only_models
                
                        selected_models = st.multiselect(
                            "🤖 Modelos a entrenar",
                            options=available_models,
                            default=available_models[:1],
                            key="advanced_models_multiselect"
                        )
                     
                        estrategia = []
                        if classification_subtype == "Multiclase":
                            estrategia = st.multiselect(
                                "Estrategia multiclase",
                                options=["One-vs-Rest", "One-vs-One"],
                                default=["One-vs-Rest"],    
                                key='epep'
                            )
                        
                        st.markdown("### 📏 Métricas de Evaluación")
                        if problem_type == "Clasificación":
                            if classification_subtype == "Binaria":
                                default_metrics = ["Accuracy", "ROC-AUC", "F1","Precision"]
                            else:
                                default_metrics = ["Accuracy", "F1","Precision"]
                            
                            metrics = st.multiselect(
                                "Seleccione métricas:",
                                options=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",'matriz de confusion'],
                                default=default_metrics,
                                key="classification_metrics"
                            )
                        else:
                            metrics = st.multiselect(
                                "Seleccione métricas:",
                                options=["MAE", "MSE", "RMSE", "R2", "MAPE"],
                                default=["RMSE", "R2"],
                                key="regression_metrics"
                            )

                st.markdown("---")
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    start_training = st.form_submit_button(
                        "🚀 Iniciar Entrenamiento Distribuido", 
                        type="primary",
                        disabled=st.session_state.training_loading
                    )

            if start_training and target_column:

                df_processed = df.copy()
                
                if missing_strategy == "Eliminar filas con valores faltantes":
                    df_processed = df_processed.dropna()
                elif missing_strategy == "Rellenar con la media/moda":
                    for col in df_processed.columns:
                        if pd.api.types.is_numeric_dtype(df_processed[col]):
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                        else:
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                elif missing_strategy == "Rellenar con valor específico":
                    try:
                        fill_val = float(fill_value) if '.' in fill_value else int(fill_value)
                    except ValueError:
                        fill_val = fill_value
                    df_processed = df_processed.fillna(fill_val)
                
                has_missing = df_processed.isna().sum().sum() > 0
                target_in_excluded = target_column in features_to_exclude
                
                if has_missing:
                    st.error("⚠️ Aún hay valores faltantes en el dataset. Por favor aplique una estrategia de manejo.")
                elif target_in_excluded:
                    st.error("❌ La columna target no puede estar en las características a excluir.")
                else:
                    training_params = {
                        "df": df_processed,
                        "target_column": target_column,
                        "problem_type": problem_type,
                        "metrics": metrics,
                        "test_size": test_size,
                        "random_state": random_state,
                        "features_to_exclude": features_to_exclude,
                        "transform_target": transform_target,
                        "selected_models": selected_models,
                        'estrategia': estrategia,
                        "dataset_name": uploaded_file.name
                    }
                    
                    
                    with st.spinner("🔄 Entrenamiento en progreso... (no cierre esta ventana)"):
                        run_training_in_thread(training_params, metrics, api_client)

                        try:
                            if st.session_state.get('train_result'):
                                st.success("✅ ¡Entrenamiento completado!")
                                plot_results(st.session_state['train_result'], st.session_state.get('train_metrics', []))
                            elif st.session_state.get('train_error'):
                                st.error(st.session_state['train_error'])
                        except Exception as result_error:
                            st.error(f"❌ Error mostrando resultados: {result_error}")
                           
                            st.session_state['train_result'] = None
                            st.session_state['train_error'] = None
                            
                            st.write("**Debug Info:**")
                            if st.session_state.get('train_result'):
                                st.write("Estructura de train_result:")
                                st.json(st.session_state['train_result'])
                            st.exception(result_error)
                            
        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
    else:
        with col2:
            st.info("📁 Suba un archivo CSV para comenzar la configuración del entrenamiento")


def plot_results(data, metrics):
    """Genera gráficos de resultados de entrenamiento con matrices de confusión y comparativas."""
    
    if not data:
        st.warning("No hay datos para mostrar")
        return
    if isinstance(data, dict):

        data = list(data.values())
        st.info(f"✅ Convertido diccionario de {len(data)} modelos a lista")
    elif not isinstance(data, list):
        st.error(f"Error: Se esperaba una lista o diccionario de resultados, pero se recibió: {type(data)}")
        st.json(data)  
        return
    
    # Filtrar solo los modelos exitosos
    successful_models = [model for model in data if model.get('status') == 'Success']
    
    if not successful_models:
        st.warning("No hay modelos entrenados exitosamente para mostrar")
        if data:
            st.write("Datos recibidos:")
            st.json(data)
        return
    
    st.markdown("## 📊 Resultados del Entrenamiento")
    
    st.markdown("### 📈 Comparación entre Modelos")
    
    comparison_data = []
    for model in successful_models:
        try:

            if 'model_name' not in model:
                st.error(f"Error: Modelo sin 'model_name': {model}")
                continue
                
            if 'scores' not in model:
                st.error(f"Error: Modelo '{model.get('model_name', 'unknown')}' sin 'scores': {model}")
                continue
            
            model_metrics = {
                'Modelo': model['model_name'],  
                'Status': model.get('status', 'Unknown')
            }

            for metric, value in model['scores'].items():
                try:
                    if isinstance(value, (int, float)):
                        model_metrics[metric] = value
                    elif isinstance(value, dict) and 'mean' in value:
                        model_metrics[metric] = value['mean']
                    elif metric == 'Confusion Matrix':
                        continue
                    else:

                        model_metrics[metric] = str(value)
                except Exception as metric_error:
                    st.warning(f"Error procesando métrica '{metric}' para modelo '{model['model_name']}': {metric_error}")
                    continue
            
            comparison_data.append(model_metrics)
            
        except KeyError as e:
            st.error(f"Error procesando modelo: falta la clave {e}")
            st.json(model)
            continue
        except Exception as e:
            st.error(f"Error inesperado procesando modelo: {e}")
            st.json(model)
            continue
    
    df_comparison = pd.DataFrame(comparison_data)
    
    if len(df_comparison) > 1:
        metrics_to_plot = [m for m in df_comparison.columns if m not in ['Modelo', 'Status'] and 
                          isinstance(df_comparison[m].iloc[0], (int, float))]
        
        if metrics_to_plot:
            st.markdown("#### Comparación de Métricas Clave")
            fig = px.bar(df_comparison.melt(id_vars=['Modelo'], value_vars=metrics_to_plot),
                         x='Modelo', y='value', color='variable', barmode='group',
                         labels={'value': 'Valor', 'variable': 'Métrica'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔍 Detalle por Modelo")
    tabs = st.tabs([f"{x['model_name']}" for x in successful_models])
    
    for i, model_data in enumerate(successful_models):
        with tabs[i]:
            st.subheader(f"Modelo: {model_data['model_name']}")
 
            st.markdown("#### 📏 Métricas de Rendimiento")
            cols_metrics = st.columns(3)
            metric_count = 0
            
            # Verificar que existan scores
            if 'scores' not in model_data:
                st.error(f"Error: El modelo '{model_data.get('model_name', 'unknown')}' no tiene scores")
                continue
            
            try:
                for metric, value in model_data['scores'].items():
                    if metric == 'Confusion Matrix':
                        continue
                        
                    with cols_metrics[metric_count % 3]:
                        try:
                            if isinstance(value, (int, float)):
                                st.metric(label=metric, value=f"{value:.4f}")
                            elif isinstance(value, dict) and 'mean' in value:
                                st.metric(label=metric, 
                                         value=f"{value['mean']:.4f}",
                                         delta=f"±{value['std']:.4f}" if 'std' in value else None)
                            else:
                                st.metric(label=metric, value=str(value))
                            metric_count += 1
                        except Exception as metric_error:
                            st.warning(f"Error mostrando métrica '{metric}': {metric_error}")
                            continue
                            
            except Exception as scores_error:
                st.error(f"Error procesando scores del modelo '{model_data.get('model_name', 'unknown')}': {scores_error}")
                st.json(model_data.get('scores', 'No scores available'))
            try:
                if 'Confusion Matrix' in model_data.get('scores', {}):
                    st.markdown("#### 🧮 Matriz de Confusión")
                    
                    cm_data = model_data['scores']['Confusion Matrix']
                    
                    if isinstance(cm_data, dict) and 'matrix' in cm_data and 'labels' in cm_data:
                        matrix = cm_data['matrix']
                        labels = cm_data['labels']
                        matrix_np = np.array(matrix)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=matrix_np,
                            x=[str(l) for l in labels], 
                            y=[str(l) for l in labels],
                            hoverongaps=False,
                            colorscale='Blues',
                            showscale=True,
                            colorbar=dict(title="Cantidad")
                        ))

                        annotations = []
                        for r_idx, row in enumerate(matrix_np):
                            for c_idx, value in enumerate(row):
                                annotations.append(
                                    go.layout.Annotation(
                                        text=str(int(value)),
                                        x=str(labels[c_idx]), 
                                        y=str(labels[r_idx]),
                                        xref='x1', 
                                        yref='y1', 
                                        showarrow=False,
                                        font=dict(
                                            color='white' if value > matrix_np.max() / 2 else 'black',
                                            size=14
                                        )
                                    )
                                )
                        
                        fig.update_layout(
                            title_text='Matriz de Confusión',
                            xaxis_title="Predicción", 
                            yaxis_title="Real",
                            annotations=annotations,
                            xaxis=dict(side='bottom'), 
                            yaxis=dict(autorange='reversed'),
                            width=500,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f'confusion_matrix_{model_data["model_name"]}_{i}')
                        
                    else:
                        st.warning("⚠️ Formato de matriz de confusión no reconocido")
                        st.write("Formato esperado: dict con 'matrix' y 'labels'")
                        st.write(f"Formato recibido: {type(cm_data)}")
                        if isinstance(cm_data, dict):
                            st.write(f"Claves disponibles: {list(cm_data.keys())}")
                            
            except Exception as cm_error:
                st.error(f"Error procesando matriz de confusión para modelo '{model_data.get('model_name', 'unknown')}': {cm_error}")
                st.write("Datos del modelo para debug:")
                st.json(model_data)