import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import random
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
    df = None
    target_column=[]
    features_to_exclude = []
    transform_target=False
    classification_subtype=""
    problem_type=""
    estrategia = []
    if 'upload' not in st.session_state:
        st.session_state.upload = False
    if 'faltantes' not in st.session_state:
        st.session_state.faltantes= False
    if 'veri' not in st.session_state:
        st.session_state.veri = False
    col1, col2 = st.columns([3, 2])
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = pd.DataFrame()
    with col1:
        uploaded_file = st.file_uploader(
            "Subir archivo CSV", 
            type=["csv"],
            key="advanced_dataset_upload"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.upload= True
                df = api_client.read(uploaded_file)
                st.session_state.current_dataset = df
                if  df.empty:
                    st.error("❌ El archivo CSV está vacío o no se pudo leer correctamente.")
                    return
                st.markdown(f"""
                **📊 Estadísticas del Dataset:**
                - 📝 Registros: {df.shape[0]:,}
                - 🔠 Características: {df.shape[1]}
                - 🕵 Valores faltantes: {df.isna().sum().sum()}
                """)
                
                with st.expander("🔍 Vista previa del dataset (primeras 10 filas)"):
                    st.dataframe(st.session_state.current_dataset.head(10))
                
                with st.expander("🧹 Manejo de Valores Faltantes", expanded=True):
                    missing_strategy = st.radio(
                        "Estrategia para valores faltantes:",
                        options=[
                            "Eliminar filas con valores faltantes",
                            "Rellenar con la media/moda",
                            "Rellenar con valor específico"
                        ],
                        
                        key="missing_strategy"
                    )
                    
                    if missing_strategy == "Rellenar con valor específico":
                        fill_value = st.text_input("Valor de relleno:", "0")
                        try:
                            fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
                        except ValueError:
                            pass  
                
                if missing_strategy == "Eliminar filas con valores faltantes":
                    _df = df.copy()
                    _df = _df.dropna()
                    st.session_state.current_dataset = _df

                    st.success(f"✅ Filas después de eliminar valores faltantes: {_df.shape[0]}")
                elif missing_strategy == "Rellenar con la media/moda":
                    _df = df.copy()
                    for col in _df.columns:
                        if pd.api.types.is_numeric_dtype(_df[col]):
                            _df[col] = _df[col].fillna(_df[col].mean())
                        else:
                            _df[col] = _df[col].fillna(_df[col].mode()[0])
                    st.session_state.current_dataset = _df
                    st.success("✅ Valores faltantes rellenados con media/moda")
                elif missing_strategy == "Rellenar con valor específico":
                    _df = _df.fillna(fill_value)
                    st.session_state.current_dataset = _df
                    st.success(f"✅ Valores faltantes rellenados con: {fill_value}")
                
                target_column = st.selectbox(
                    "🎯 Seleccione la columna target (variable objetivo)",
                    options=df.columns,
                    index=len(df.columns)-1,
                    key="target_column_select"
                )
                
                if target_column:
                    target_series = st.session_state.current_dataset[target_column]
                    unique_values = target_series.nunique()
    
                    if pd.api.types.is_numeric_dtype(target_series):
                        auto_problem_type = "Regresión" if unique_values > 10 else "Clasificación"
                    else:
                        auto_problem_type = "Clasificación"
                    
                    classification_subtype = None
                    if auto_problem_type == "Clasificación":
                        classification_subtype = "Binaria" if unique_values == 2 else "Multiclase"
                    
                    detection_msg = f"""
                    🔍 Tipo de problema: {auto_problem_type}
                    """
                    if classification_subtype:
                        detection_msg += f"- Subtipo: {classification_subtype} ({unique_values} clases)"
                    
                    st.markdown(detection_msg)
                    
                    with st.expander("⚙️ Opciones avanzadas de preprocesamiento", expanded=True):
                        features_to_exclude = st.multiselect(
                            "🚫 Excluir columnas del modelo:",
                            options=[col for col in df.columns if col != target_column],
                            key="features_to_exclude"
                        )
                        
                        if auto_problem_type == "Regresión":
                            transform_target = st.checkbox(
                                "Transformar target (logarítmico)",
                                help="Útil para distribuciones sesgadas"
                            )
                        
                        
                
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")

    with col2:
        if uploaded_file is not None and target_column:

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
            
            if classification_subtype=="Multiclase":
                estrategia = st.multiselect(
                    "estrategia multiclase",
                    options=["One-vs-Rest", "One-vs-One"],
                    default=["One-vs-Rest"],    
                    key='epep'
                )
            st.markdown("### 🔄 Validación Cruzada")
            cv_folds = st.selectbox(
                "Número de folds para CV",
                options=[3, 5, 10],
                index=1,
                key="cv_folds_select"
            )
            
            st.markdown("### 📏 Métricas de Evaluación")
            if problem_type == "Clasificación":
                if classification_subtype == "Binaria":
                    default_metrics = ["Accuracy", "ROC-AUC", "F1","Precision"]
                
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

        results_container = st.container()
        st.markdown("---")
        c, col2, col3 = st.columns([1,2,1])
        start_training = False
        with col2:
            if st.session_state.upload:
                if st.button("🔍 Validar Configuración"):
                    st.session_state.veri = True
                    if st.session_state.current_dataset.isna().sum().sum() > 0:
                        st.session_state.faltantes = True
                        
                        st.warning("⚠️ Aún hay valores faltantes en el dataset. Por favor aplique una estrategia de manejo.")
                    elif target_column in features_to_exclude:
                        st.error("❌ La columna target no puede estar en las características a excluir.")
                    else:
                        st.success("✅ Configuración válida. Puede proceder con el entrenamiento.")

            if st.session_state.upload:
                if st.session_state.veri:
                    if not st.session_state.faltantes:
                        start_training = st.button(
                            "🚀 Iniciar Entrenamiento Distribuido", 
                            type="primary",
                            key="advanced_start_training_button",
                            disabled=False  
                        )
                    else:
                        start_training = st.button(
                            "🚀 Iniciar Entrenamiento Distribuido", 
                            type="primary",
                            key="advanced_start_training_button",
                            disabled=True  
                        )
                else:
                    st.info("verifica el dataset primero ")

    if start_training:
        st.session_state.training_in_progress = True
        with st.container():  
            st.markdown("""
                <div class="dashboard-container">
                    <h3>🔄 Entrenando Modelos en Paralelo</h3>
                </div>
                """, unsafe_allow_html=True)
            
            training_params = {
                "df": st.session_state.current_dataset,
                "target_column": target_column,
                "problem_type": problem_type,
                "metrics": metrics,
                "test_size": test_size,
                'cv_folds': cv_folds,
                "random_state": random_state,
                "features_to_exclude": features_to_exclude,
                "transform_target": transform_target,
                "selected_models": selected_models,
                'estrategia': estrategia
            }
            
            with st.spinner("Enviando trabajo al clúster de Ray..."):
                try:
                    response = api_client.start(training_params)
                    if 'data' in response:
                        plot_results(response['data'],metrics)  
                    else:
                        st.error("ocurrio un error")
                except Exception as e:

                    st.error(f"❌ Error al enviar el trabajo al clúster: {str(e)}")
                    return


def plot_results(data, metrics):
    """Genera gráficos de resultados de entrenamiento con matrices de confusión y comparativas."""
    
    if not data:
        st.warning("No hay datos para mostrar")
        return
    
    st.markdown("## 📊 Resultados del Entrenamiento")
    
    # --- Comparación entre modelos (fuera de las tabs) ---
    st.markdown("### 📈 Comparación entre Modelos")
    
    # Crear DataFrame para comparación
    comparison_data = []
    for model in data:
        model_metrics = {
            'Modelo': model['model'],
            'Folds Completados': f"{model.get('completed_folds', '?')}/{model.get('total_folds', '?')}"
        }
        
        # Agregar métricas numéricas
        for metric, value in model['scores'].items():
            if isinstance(value, (int, float)):
                model_metrics[metric] = value
            elif isinstance(value, dict) and 'mean' in value:
                model_metrics[metric] = value['mean']
        
        comparison_data.append(model_metrics)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    
    # Gráfico de comparación de métricas clave
    if len(df_comparison) > 1:
        metrics_to_plot = [m for m in df_comparison.columns if m not in ['Modelo', 'Folds Completados'] and 
                          isinstance(df_comparison[m].iloc[0], (int, float))]
        
        if metrics_to_plot:
            st.markdown("#### Comparación de Métricas Clave")
            fig = px.bar(df_comparison.melt(id_vars=['Modelo'], value_vars=metrics_to_plot),
                         x='Modelo', y='value', color='variable', barmode='group',
                         labels={'value': 'Valor', 'variable': 'Métrica'})
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Tabs por modelo ---
    st.markdown("### 🔍 Detalle por Modelo")
    tabs = st.tabs([f"{x['model']}" for x in data])
    
    for i, model_data in enumerate(data):
        with tabs[i]:
            st.subheader(f"Modelo: {model_data['model']}")
            
            # Mostrar métricas principales
            st.markdown("#### 📏 Métricas de Rendimiento")
            cols_metrics = st.columns(3)
            metric_count = 0
            
            for metric, value in model_data['scores'].items():
                if metric == 'Confusion Matrix':
                    continue
                    
                with cols_metrics[metric_count % 3]:
                    if isinstance(value, (int, float)):
                        st.metric(label=metric, value=f"{value:.4f}")
                    elif isinstance(value, dict) and 'mean' in value:
                        st.metric(label=metric, 
                                 value=f"{value['mean']:.4f}",
                                 delta=f"±{value['std']:.4f}" if 'std' in value else None)
                    else:
                        st.metric(label=metric, value=str(value))
                    metric_count += 1
            
            # Mostrar matriz de confusión si está disponible
            if 'Confusion Matrix' in model_data['scores']:
                cm_data = model_data['scores']['Confusion Matrix']
                
                st.markdown("#### 🧮 Matriz de Confusión")
                
                if 'Confusion Matrix' in model_data['scores']:
                            cm_data = model_data['scores']['Confusion Matrix']
                            if isinstance(cm_data, dict) and 'matrix' in cm_data and 'labels' in cm_data:
                                st.markdown("---")
                                st.subheader("Matriz de Confusión")
                                
                                matrix = cm_data['matrix']
                                labels = cm_data['labels']
                                matrix_np = np.array(matrix)
                                
                                fig = go.Figure(data=go.Heatmap(
                                    z=matrix_np,
                                    x=[str(l) for l in labels], 
                                    y=[str(l) for l in labels],
                                    hoverongaps=False,
                                    colorscale='Blues',
                                    showscale=False
                                ))
                                
                                annotations = []
                                for r_idx, row in enumerate(matrix_np):
                                    for c_idx, value in enumerate(row):
                                        annotations.append(
                                            go.layout.Annotation(
                                                text=str(value),
                                                x=str(labels[c_idx]), y=str(labels[r_idx]),
                                                xref='x1', yref='y1', showarrow=False,
                                                font=dict(color='white' if value > matrix_np.max() / 2 else 'black')
                                            )
                                        )
                                
                                fig.update_layout(
                                    title_text='Matriz de Confusión',
                                    xaxis_title="Predicción", yaxis_title="Real",
                                    annotations=annotations,
                                    xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed')
                                )
                                st.plotly_chart(fig, use_container_width=True,key=f'{random.random()}')
    
                else:
                    st.warning("Formato de matriz de confusión no reconocido")