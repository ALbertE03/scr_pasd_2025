import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import random
import json

# Configuración de modelos
CLASSIFICATION_MODELS = [
    "RandomForest", "GradientBoosting", "AdaBoost", "ExtraTrees", "DecisionTree", 
    "LogisticRegression", "SGD", "PassiveAggressive", "KNN", "SVM", "LinearSVM",
    "GaussianNB", "BernoulliNB", "MultinomialNB", "ComplementNB", "LDA", "QDA", 
    "MLP", "Bagging", "Voting"
]

REGRESSION_MODELS = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor", 
    "RANSACRegressor", "TheilSenRegressor", "ARDRegression", "BayesianRidge",
    "PassiveAggressiveRegressor", "SGDRegressor", "DecisionTreeRegressor",
    "ExtraTreesRegressor", "RandomForestRegressor", "GradientBoostingRegressor",
    "HistGradientBoostingRegressor", "AdaBoostRegressor", "SVR", "LinearSVR",
    "KNeighborsRegressor", "MLPRegressor", "BaggingRegressor", "VotingRegressor"
]

HYPERPARAMETER_CONFIG = {
    "RandomForest": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 10, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 10, "default": 2, "step": 1}
    },
    "GradientBoosting": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
        "max_depth": {"type": "slider", "min": 2, "max": 10, "default": 3, "step": 1}
    },
    "SVM": {
        "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        "kernel": {"type": "selectbox", "options": ["linear", "rbf", "poly"], "default": 1},
        "gamma": {"type": "selectbox", "options": ["scale", "auto"], "default": 0}
    },
    "LogisticRegression": {
        "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        "max_iter": {"type": "slider", "min": 100, "max": 1000, "default": 500, "step": 100},
        "solver": {"type": "selectbox", "options": ["lbfgs", "liblinear", "newton-cg"], "default": 0}
    },
    "AdaBoost": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 50, "step": 10},
        "learning_rate": {"type": "slider", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.1}
    },
    "ExtraTrees": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 10, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 10, "default": 2, "step": 1}
    },
    "DecisionTree": {
        "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 10, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 10, "default": 2, "step": 1},
        "criterion": {"type": "selectbox", "options": ["gini", "entropy"], "default": 0}
    },
    "KNN": {
        "n_neighbors": {"type": "slider", "min": 1, "max": 20, "default": 5, "step": 1},
        "weights": {"type": "selectbox", "options": ["uniform", "distance"], "default": 0},
        "algorithm": {"type": "selectbox", "options": ["auto", "ball_tree", "kd_tree", "brute"], "default": 0}
    },
    "LinearSVM": {
        "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    },
    "SGD": {
        "alpha": {"type": "slider", "min": 0.0001, "max": 0.1, "default": 0.0001, "step": 0.0001, "format": "%.4f"},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    },
    "GaussianNB": {
        "var_smoothing": {"type": "slider", "min": 1e-12, "max": 1e-6, "default": 1e-9, "step": 1e-12, "format": "%.2e"}
    },
    "BernoulliNB": {
        "alpha": {"type": "slider", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.01},
        "fit_prior": {"type": "checkbox", "default": True}
    },
    "MultinomialNB": {
        "alpha": {"type": "slider", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.01},
        "fit_prior": {"type": "checkbox", "default": True}
    },
    "MLP": {
        "hidden_layer_sizes": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "activation": {"type": "selectbox", "options": ["relu", "tanh", "logistic"], "default": 0},
        "max_iter": {"type": "slider", "min": 200, "max": 1000, "default": 500, "step": 50}
    },
    # Modelos de regresión
    "LinearRegression": {
        "fit_intercept": {"type": "checkbox", "default": True}
    },
    "Ridge": {
        "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    },
    "Lasso": {
        "alpha": {"type": "slider", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.01},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    },
    "ElasticNet": {
        "alpha": {"type": "slider", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.01},
        "l1_ratio": {"type": "slider", "min": 0.1, "max": 0.9, "default": 0.5, "step": 0.1},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    },
    "HuberRegressor": {
        "epsilon": {"type": "slider", "min": 1.1, "max": 2.0, "default": 1.35, "step": 0.05},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100},
        "alpha": {"type": "slider", "min": 0.0001, "max": 0.01, "default": 0.0001, "step": 0.0001, "format": "%.4f"}
    },
    "RANSACRegressor": {
        "max_trials": {"type": "slider", "min": 50, "max": 500, "default": 100, "step": 50},
        "min_samples": {"type": "slider", "min": 2, "max": 50, "default": 10, "step": 1},
        "residual_threshold": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1}
    },
    "SVR": {
        "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        "kernel": {"type": "selectbox", "options": ["linear", "rbf", "poly"], "default": 1},
        "gamma": {"type": "selectbox", "options": ["scale", "auto"], "default": 0}
    },
    "RandomForestRegressor": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 10, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 10, "default": 2, "step": 1}
    },
    "GradientBoostingRegressor": {
        "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
        "max_depth": {"type": "slider", "min": 2, "max": 10, "default": 3, "step": 1}
    },
    "MLPRegressor": {
        "hidden_layer_sizes": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
        "activation": {"type": "selectbox", "options": ["relu", "tanh", "logistic"], "default": 0},
        "max_iter": {"type": "slider", "min": 500, "max": 2000, "default": 1000, "step": 100}
    }
}

def init_session_state():
    """Inicializa variables del session state"""
    defaults = {
        'training_loading': False,
        'train_result': None,
        'train_error': None,
        'current_target': None,
        'target_distribution': None,
        'problem_type_detected': None,
        'classification_subtype': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
def get_model_manager_stats(api_client):
    """Obtiene estadísticas del ModelManager"""
    try:
        response = api_client.get("/models/stats")
        return response.get('model_manager_stats', {"total_models": 0, "model_ids": []})
    except Exception as e:
        st.error(f"Error obteniendo estadísticas del ModelManager: {e}")
        return {"total_models": 0, "model_ids": []}

def search_models_in_manager(api_client, model_name_pattern=None, dataset_name_pattern=None):
    """Busca modelos en el ModelManager"""
    try:
        params = {}
        if model_name_pattern:
            params['model_name'] = model_name_pattern
        if dataset_name_pattern:
            params['dataset_name'] = dataset_name_pattern
        
        response = api_client.get("/models/search", params=params)
        return response.get('matching_models', [])
    except Exception as e:
        st.error(f"Error buscando modelos: {e}")
        return []

def detect_problem_type(target_series):
    """Detecta automáticamente el tipo de problema y subtipo"""
    unique_values = target_series.nunique()
    
    if pd.api.types.is_numeric_dtype(target_series):
        problem_type = "Regresión" if unique_values > 10 else "Clasificación"
    else:
        problem_type = "Clasificación"
    
    classification_subtype = None
    if problem_type == "Clasificación":
        classification_subtype = "Binaria" if unique_values == 2 else "Multiclase"
    
    return problem_type, classification_subtype, unique_values

def update_target_distribution(df, target_column):
    """Actualiza la distribución del target cuando cambia"""
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        problem_type, classification_subtype, unique_values = detect_problem_type(target_series)
 
        st.session_state.current_target = target_column
        st.session_state.problem_type_detected = problem_type
        st.session_state.classification_subtype = classification_subtype
        
        if problem_type == "Clasificación":
            st.session_state.target_distribution = target_series.value_counts().sort_index()
        else:
            st.session_state.target_distribution = {
                'mean': target_series.mean(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max()
            }
        
        return problem_type, classification_subtype, unique_values
    return None, None, 0

def show_target_info(df, target_column):
    """Muestra información del target seleccionado"""
    if not target_column:
        return None, None, 0
    
    if st.session_state.current_target != target_column:
        problem_type, classification_subtype, unique_values = update_target_distribution(df, target_column)
    else:
        problem_type = st.session_state.problem_type_detected
        classification_subtype = st.session_state.classification_subtype
        unique_values = len(st.session_state.target_distribution) if isinstance(st.session_state.target_distribution, pd.Series) else df[target_column].nunique()
    
    detection_msg = f"🔍 **Tipo de problema detectado:** {problem_type}"
    if classification_subtype:
        detection_msg += f" - Subtipo: {classification_subtype} ({unique_values} clases)"
    st.markdown(detection_msg)
    

    if problem_type == "Clasificación":
        class_counts = st.session_state.target_distribution
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            st.warning("⚠️ **Problema detectado:** Algunas clases tienen muy pocas muestras:")
            #st.info("💡 **Recomendaciones:** Agregar más datos, combinar clases, usar test_size menor")
        
        with st.expander("📊 Distribución de clases"):
            st.bar_chart(class_counts)
    else:
        stats = st.session_state.target_distribution
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Media", f"{stats['mean']:.2f}")
        with col2:
            st.metric("Desv. Std", f"{stats['std']:.2f}")
        with col3:
            st.metric("Mínimo", f"{stats['min']:.2f}")
        with col4:
            st.metric("Máximo", f"{stats['max']:.2f}")
    
    return problem_type, classification_subtype, unique_values

def create_hyperparameter_controls(selected_models, context_prefix=""):
    """Crea controles de hiperparámetros de forma dinámica"""
    if not selected_models:
        return {}
        
    hyperparams = {}
    
    for model in selected_models:
        if model in HYPERPARAMETER_CONFIG:
            st.subheader(f"⚙️ {model}")
            config = HYPERPARAMETER_CONFIG[model]
            cols = st.columns(min(3, len(config)))
            hyperparams[model] = {}
            
            for i, (param, param_config) in enumerate(config.items()):
                with cols[i % len(cols)]:

                    key = f"{context_prefix}_{model}_{param}_{hash(str(param_config))}"
                    
                    if param_config["type"] == "slider":
                        if "format" in param_config:
                            hyperparams[model][param] = st.slider(
                                f"{param.replace('_', ' ').title()}", 
                                param_config["min"], param_config["max"], 
                                param_config["default"], 
                                param_config.get("step", 1), 
                                format=param_config["format"], 
                                key=key,
                                help=f"Rango: {param_config['min']} - {param_config['max']}"
                            )
                        else:
                            hyperparams[model][param] = st.slider(
                                f"{param.replace('_', ' ').title()}", 
                                param_config["min"], param_config["max"], 
                                param_config["default"], 
                                param_config.get("step", 1), 
                                key=key,
                                help=f"Rango: {param_config['min']} - {param_config['max']}"
                            )
                    elif param_config["type"] == "selectbox":
                        hyperparams[model][param] = st.selectbox(
                            f"{param.replace('_', ' ').title()}", 
                            param_config["options"], 
                            param_config["default"], 
                            key=key,
                            help=f"Opciones: {', '.join(param_config['options'])}"
                        )
                    elif param_config["type"] == "checkbox":
                        hyperparams[model][param] = st.checkbox(
                            f"{param.replace('_', ' ').title()}", 
                            param_config["default"], 
                            key=key
                        )
        else:
            st.info(f"ℹ️ {model}: Usando parámetros por defecto (no hay configuración personalizada)")
            hyperparams[model] = {}
    
    return hyperparams

def process_training_response(response, metrics):
    """Procesa la respuesta del entrenamiento de forma simplificada"""
    if 'error' in response:
        st.session_state['train_result'] = None
        st.session_state['train_error'] = f"Error con la API: {response['error']}"
        return
    
    if 'data' not in response:
        st.session_state['train_result'] = None
        st.session_state['train_error'] = "Respuesta inesperada de la API - no hay 'data'"
        return
    
    data = response['data']

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):

        if all(isinstance(v, dict) and 'model_name' in v and 'scores' in v 
               for v in data[0].values() if isinstance(v, dict)):
            data = data[0]
    
    if isinstance(data, dict):
        if not data or all(isinstance(item, dict) and 'model_name' in item and 'scores' in item 
                          for item in data.values() if isinstance(item, dict)):
            st.session_state['train_result'] = data
            st.session_state['train_metrics'] = metrics
            st.session_state['train_error'] = None
            
            for key in ['model_manager_stats', 'models_trained']:
                if key in response:
                    st.session_state[f'{key}_info'] = response[key]
        else:
            st.session_state['train_result'] = None
            st.session_state['train_error'] = f"Estructura de datos inválida: {str(data)[:200]}..."
    else:
        st.session_state['train_result'] = None
        st.session_state['train_error'] = f"Tipo de datos inesperado: {type(data)}"

def render_training_tab(cluster_status, api_client):
    """Renderiza la pestaña de entrenamiento con capacidades avanzadas"""
    st.header("🧠 Entrenamiento Distribuido")
    
    if not cluster_status['connected']:
        st.warning("Debes conectarte al cluster para ejecutar entrenamientos")
        return

    training_tabs = st.tabs([
        "🚀 Entrenamiento"
    ])
       # "🚀 Entrenamiento Avanzado"
    # ])
    
    with training_tabs[0]:
        render_advanced_training(api_client)
    
    #with training_tabs[1]:
        #render_legacy_training(cluster_status, api_client)

def render_legacy_training(cluster_status, api_client):
    """Renderiza la interfaz de entrenamiento avanzada legacy"""
    st.subheader("🚀 Entrenamiento Distribuido Avanzado")
    st.markdown("""
    <div class="success-card">
        <h4>✅ Procesamiento de múltiples modelos y datasets en paralelo</h4>
        <p>Entrene varios modelos y datasets simultáneamente aprovechando la potencia del cluster distribuido</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        datasets = ['iris', 'wine', 'breast_cancer', 'digits']
        selected_dataset = st.multiselect(
            "Dataset",
            options=datasets,
            default=datasets[:2],
            key="advanced_dataset_select_pro"
        )
        st.session_state.current_dataset = selected_dataset
        for i in selected_dataset:
            st.session_state.data_test_size[i] = 0.3
            
        selected_models = st.multiselect(
            "Modelos a entrenar en paralelo",
            options=CLASSIFICATION_MODELS,
            default=st.session_state.selected_models[:4] if hasattr(st.session_state, 'selected_models') else CLASSIFICATION_MODELS[:4],
            key="advanced_models_multiselect_pro"
        )
        
        st.session_state.selected_models = selected_models
    
    with col2:
        r = st.selectbox('', options=selected_dataset, key='SCR', help='por defecto todos son 0.3')
        d = datasets[datasets.index(r)]
        st.session_state.data_test_size[d] = st.slider(
            "% Datos de prueba",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.test_size,
            step=0.05,
            key="advanced_test_size_slider_pro"
        )
    
    with st.expander("⚙️ Configuración de Hiperparámetros"):
        st.caption("Configure hiperparámetros específicos para cada modelo seleccionado")
        hyperparams = create_hyperparameter_controls(selected_models, "legacy_training")
                
    co1, co2, co3 = st.columns([1, 2, 1])
    with co2:
        start_training = st.button(
            "🚀 Iniciar Entrenamiento Distribuido", 
            type="primary",
            key="advanced_start_training_button_pro",
        )
    
    if start_training:
        st.markdown("""
        <div class="dashboard-container">                
                    <h3>🔄 Entrenando Modelos en Paralelo</h3>
        </div>
        """, unsafe_allow_html=True)
def render_advanced_training(api_client):
    """Renderiza la interfaz de entrenamiento simplificada"""
    init_session_state()
    
    st.subheader("🚀 Entrenamiento Distribuido")
    st.markdown("""
    <div class="success-card">
        <h4>✅ Procesamiento de múltiples modelos en paralelo</h4>
        <p>Entrene varios modelos simultáneamente aprovechando la potencia del cluster distribuido</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                return
            
            with col1:
                st.markdown(f"""
                **📊 Estadísticas del Dataset:**
                - 📝 Registros: {df.shape[0]:,}
                - 🔠 Características: {df.shape[1]}
                - 🕵 Valores faltantes: {df.isna().sum().sum()}
                """)
                
                with st.expander("🔍 Vista previa del dataset (primeras 10 filas)"):
                    st.dataframe(df.head(10))
            
      
            st.markdown("---")
            st.markdown("### 🎯 Configuración del Target")
            
            target_column = st.selectbox(
                "Seleccione la columna target (variable objetivo)",
                options=df.columns,
                index=len(df.columns)-1,
                key="target_column_select"
            )
            
            problem_type = None
            classification_subtype = None
            unique_values = 0
            
            if target_column:
                problem_type, classification_subtype, unique_values = show_target_info(df, target_column)

            hyperparams = {}
            if target_column:
                st.markdown("---")
                st.markdown("### 🔧 Configuración de Hiperparámetros")
                
                if st.session_state.get('problem_type_detected') == "Clasificación":
                    available_models_for_hyperparams = CLASSIFICATION_MODELS
                else:
                    available_models_for_hyperparams = REGRESSION_MODELS
                

                selected_models_for_hyperparams = st.multiselect(
                    "🤖 Seleccione modelos para configurar hiperparámetros:",
                    options=available_models_for_hyperparams,
                    default=[],
                    key="models_for_hyperparams",
                    help="Los modelos seleccionados aquí tendrán configuración avanzada de hiperparámetros"
                )
                
                if selected_models_for_hyperparams:
                    with st.expander("⚙️ Configuración Avanzada de Hiperparámetros", expanded=True):
                        st.info("💡 **Tip:** Los valores por defecto son adecuados para la mayoría de casos. Modifica solo si conoces el impacto de cada parámetro.")
                        hyperparams = create_hyperparameter_controls(selected_models_for_hyperparams, "advanced_training")
                else:
                    st.info("👆 Seleccione modelos arriba para configurar sus hiperparámetros")

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
                    
                    if target_column:
                        with st.expander("⚙️ Opciones avanzadas de preprocesamiento", expanded=True):
                            features_to_exclude = st.multiselect(
                                "🚫 Excluir columnas del modelo:",
                                options=[col for col in df.columns if col != target_column],
                                key="features_to_exclude"
                            )
                            
                            transform_target = False
                            if problem_type == "Regresión":
                                transform_target = st.checkbox(
                                    "Transformar target (logarítmico)",
                                    help="Útil para distribuciones sesgadas"
                                )
                    else:
                        features_to_exclude = []
                        transform_target = False
                
                with col_form2:
                    if target_column:
                        st.markdown("## ⚙️ Configuración del Modelado")

                        default_index = 0 if st.session_state.get('problem_type_detected') == "Clasificación" else 1
                        problem_type = st.radio(
                            "Seleccione el tipo de problema:",
                            options=["Clasificación", "Regresión"],
                            index=default_index,
                            key="problem_type_selection"
                        )

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
                            available_models = CLASSIFICATION_MODELS
                        else:
                            available_models = REGRESSION_MODELS
                
                        selected_models = st.multiselect(
                            "🤖 Modelos a entrenar",
                            options=available_models,
                            default=available_models[:1],
                            key="advanced_models_multiselect",
                            help="Seleccione uno o más modelos para entrenar en paralelo"
                        )
                     

                        estrategia = []
                        if problem_type == "Clasificación":
                            estrategia = st.multiselect(
                                "Estrategia multiclase",
                                options=["One-vs-Rest", "One-vs-One"],
                                default=["One-vs-Rest"],    
                                key='estrategia_multiclase',
                                help="Estrategia para problemas de clasificación multiclase"
                            )
                        
                        st.markdown("### 📏 Métricas de Evaluación")
                        if problem_type == "Clasificación":
                            if classification_subtype == "Binaria":
                                default_metrics = ["Accuracy", "ROC-AUC", "F1"]
                                available_metrics = ["Accuracy", "Recall", "F1", "ROC-AUC", 'matriz de confusion']
                            else:
                                default_metrics = ["Accuracy", "F1"]
                                available_metrics = ["Accuracy", "Recall", "F1", 'matriz de confusion']
                            
                            metrics = st.multiselect(
                                "Seleccione métricas:",
                                options=available_metrics,
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
                    else:
                        st.info("👆 Primero seleccione una columna target")
                        problem_type = None
                        selected_models = []
                        estrategia = []
                        metrics = []
                        test_size = 0.2
                        random_state = 42

                st.markdown("---")
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    start_training = st.form_submit_button(
                        "🚀 Iniciar Entrenamiento Distribuido", 
                        type="primary",
                        disabled=st.session_state.training_loading or not target_column or not selected_models
                    )
            if start_training and target_column and selected_models:
                handle_training_execution(
                    df, target_column, problem_type, metrics, test_size, 
                    random_state, features_to_exclude, transform_target, 
                    selected_models, estrategia, uploaded_file.name, 
                    missing_strategy, fill_value, api_client, hyperparams
                )
                        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
    else:
        with col2:
            st.info("📁 Suba un archivo CSV para comenzar la configuración del entrenamiento")

def handle_training_execution(df, target_column, problem_type, metrics, test_size, 
                             random_state, features_to_exclude, transform_target, 
                             selected_models, estrategia, dataset_name, 
                             missing_strategy, fill_value, api_client, hyperparams=None):
    """Maneja la ejecución del entrenamiento de forma simplificada"""
    
    # Mostrar reporte de calidad de datos antes del procesamiento
    st.markdown("### 📊 Análisis de Calidad de Datos")
    show_data_quality_report(df)
    
    # Preprocesar los datos
    df_processed = preprocess_data(df, missing_strategy, fill_value)
    
    # Mostrar reporte después del procesamiento si hubo cambios
    if not df.equals(df_processed):
        st.markdown("#### 🔄 Después del preprocesamiento:")
        show_data_quality_report(df_processed)

    # Validar los datos para entrenamiento
    validation_result = validate_training_data(
        df_processed, target_column, features_to_exclude, 
        problem_type, test_size
    )
    
    if not validation_result['is_valid']:
        st.error(validation_result['error_message'])
        return

    # Crear parámetros de entrenamiento
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
        "dataset_name": dataset_name
    }
    
    # Agregar hiperparámetros si están disponibles
    if hyperparams:
        training_params["hyperparams"] = hyperparams
    
    # Limpiar y validar parámetros para JSON
    st.markdown("### 🧹 Validación de Compatibilidad JSON")
    cleaned_params = clean_training_params(training_params)
    
    if cleaned_params is None:
        st.error("❌ Error: No se pudieron preparar los datos para el entrenamiento")
        return
    
    # Verificación final de serialización
    try:
        # Test de serialización del DataFrame (solo una muestra pequeña)
        test_sample = cleaned_params['df'].head(5).to_dict('records')
        json.dumps(test_sample)
        st.success("✅ Datos validados correctamente para JSON")
    except Exception as e:
        st.error(f"❌ Error en validación JSON final: {e}")
        return
   
    with st.spinner("🔄 Entrenamiento en progreso... (no cierre esta ventana)"):
        try:
            if hyperparams:
                configured_models = [model for model in hyperparams.keys() if hyperparams[model]]
                if configured_models:
                    st.info(f"🔧 Aplicando hiperparámetros personalizados a: {', '.join(configured_models)}")
            
            response = api_client.start(cleaned_params)
            process_training_response(response, metrics)
            
            if st.session_state.get('train_result'):
                st.success("✅ ¡Entrenamiento completado!")

                if hyperparams:
                    configured_models = [model for model in hyperparams.keys() if hyperparams[model]]
                    if configured_models:
                        st.info(f"🔧 Modelos con hiperparámetros personalizados: {', '.join(configured_models)}")
                
                plot_results(st.session_state['train_result'], st.session_state.get('train_metrics', []))
            elif st.session_state.get('train_error'):
                st.error(st.session_state['train_error'])
                
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {e}")
            st.info("💡 **Sugerencia:** Revisa los hiperparámetros y el formato de los datos")
            
            # Mostrar información adicional sobre el error si es de serialización
            if "JSON" in str(e) or "serializ" in str(e).lower():
                st.error("🔧 **Error de serialización detectado.** Intenta con una estrategia diferente de manejo de valores faltantes.")
                
                # Mostrar muestra de datos problemáticos si es posible
                with st.expander("🔍 Diagnóstico de Datos"):
                    st.write("Muestra de los primeros 5 registros:")
                    st.dataframe(df_processed.head())
                    
                    st.write("Tipos de datos:")
                    st.write(df_processed.dtypes)

def preprocess_data(df, missing_strategy, fill_value):
    """Preprocesa los datos según la estrategia seleccionada"""
    df_processed = df.copy()
    
    if missing_strategy == "Eliminar filas con valores faltantes":
        df_processed = df_processed.dropna()
    elif missing_strategy == "Rellenar con la media/moda":
        for col in df_processed.columns:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                # Verificar si hay valores para calcular la media
                if not df_processed[col].dropna().empty:
                    mean_val = df_processed[col].mean()
                    # Verificar que la media no sea NaN o infinita
                    if pd.isna(mean_val) or np.isinf(mean_val):
                        st.warning(f"⚠️ Media problemática en columna '{col}', usando 0 como fallback")
                        df_processed[col] = df_processed[col].fillna(0.0)
                    else:
                        df_processed[col] = df_processed[col].fillna(mean_val)
                else:
                    st.warning(f"⚠️ Columna '{col}' sin valores válidos, rellenando con 0")
                    df_processed[col] = df_processed[col].fillna(0.0)
            else:
                # Para columnas categóricas
                if not df_processed[col].dropna().empty:
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_val[0])
                    else:
                        df_processed[col] = df_processed[col].fillna('Unknown')
                else:
                    df_processed[col] = df_processed[col].fillna('Unknown')
    elif missing_strategy == "Rellenar con valor específico":
        try:
            fill_val = float(fill_value) if '.' in fill_value else int(fill_value)
        except ValueError:
            fill_val = fill_value
        df_processed = df_processed.fillna(fill_val)
    
    # Limpiar el DataFrame para asegurar compatibilidad JSON
    df_processed = clean_dataframe_for_json(df_processed)
    
    return df_processed

def validate_training_data(df_processed, target_column, features_to_exclude, problem_type, test_size):
    """Valida los datos para el entrenamiento"""

    # Verificar valores faltantes
    if df_processed.isna().sum().sum() > 0:
        return {"is_valid": False, "error_message": "⚠️ Aún hay valores faltantes en el dataset. Por favor aplique una estrategia de manejo."}

    # Verificar valores infinitos
    inf_count = 0
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        inf_count += np.isinf(df_processed[col]).sum()
    
    if inf_count > 0:
        return {"is_valid": False, "error_message": f"⚠️ Se encontraron {inf_count} valores infinitos. Por favor aplique preprocesamiento."}

    # Verificar que el target no esté en las características a excluir
    if target_column in features_to_exclude:
        return {"is_valid": False, "error_message": "❌ La columna target no puede estar en las características a excluir."}

    # Verificar tamaño mínimo del dataset
    n_samples = len(df_processed)
    if n_samples < 10:
        return {"is_valid": False, "error_message": "❌ El dataset es demasiado pequeño. Se necesitan al menos 10 muestras."}
  
    # Validaciones específicas para clasificación
    if problem_type == "Clasificación":
        target_processed = df_processed[target_column]
        class_counts = target_processed.value_counts()
        min_class_count = class_counts.min()
        total_test_samples = int(len(df_processed) * test_size)
        min_test_samples_needed = len(class_counts)
        
        if min_class_count < 2:
            return {
                "is_valid": False, 
                "error_message": f"❌ **Error crítico:** La clase '{class_counts.idxmin()}' tiene solo {min_class_count} muestra(s). Todas las clases necesitan al menos 2 muestras."
            }
        
        if total_test_samples < min_test_samples_needed:
            max_test_size = (len(df_processed) - min_test_samples_needed) / len(df_processed)
            return {
                "is_valid": False,
                "error_message": f"❌ **Error:** Con test_size={test_size:.1%}, solo hay {total_test_samples} muestras para test, pero se necesitan al menos {min_test_samples_needed}. Reduce el test_size a máximo {max_test_size:.1%}"
            }
    
    # Verificar que hay características numéricas
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns
    if len(numeric_features) == 0 and target_column not in df_processed.select_dtypes(include=[np.number]).columns:
        return {
            "is_valid": False,
            "error_message": "⚠️ **Advertencia:** No se detectaron características numéricas. Asegúrate de que el dataset sea apropiado para machine learning."
        }
    
    # Verificación de compatibilidad JSON (muestra pequeña)
    try:
        sample_data = df_processed.to_dict('records')
        json.dumps(sample_data)
    except Exception as e:
        return {
            "is_valid": False,
            "error_message": f"❌ **Error de compatibilidad JSON:** {str(e)}. Los datos contienen valores no serializables."
        }
    
    return {"is_valid": True, "error_message": None}


def plot_results(data, metrics):
    """Genera gráficos de resultados de entrenamiento simplificado"""
    
    if not data:
        st.warning("No hay datos para mostrar")
        return

    try:

        successful_models = normalize_results_data(data)
        
        if not successful_models:
            st.warning("No hay modelos entrenados exitosamente para mostrar")
            return
        
        st.markdown("## 📊 Resultados del Entrenamiento")

        create_model_comparison(successful_models)

        show_model_details(successful_models)
        
    except Exception as e:
        st.error(f"Error mostrando resultados: {e}")
        st.json(data)

def normalize_results_data(data):
    """Normaliza el formato de los datos de resultados"""
    if isinstance(data, dict):
        data = list(data.values())
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if all(isinstance(v, dict) and 'model_name' in v for v in data[0].values() if isinstance(v, dict)):
            data = list(data[0].values())
    
    successful_models = []
    for model in data:
        if isinstance(model, dict) and ('status' not in model or model.get('status') == 'Success'):
            if 'model_name' in model and 'scores' in model:
                successful_models.append(model)
    
    return successful_models

def create_model_comparison(successful_models):
    """Crea gráficos de comparación entre modelos"""
    if len(successful_models) <= 1:
        return
        
    st.markdown("### 📈 Comparación entre Modelos")
    
    comparison_data = []
    for model in successful_models:
        model_metrics = {'Modelo': model['model_name']}
        
        for metric, value in model['scores'].items():
            if metric != 'Confusion Matrix':
                if isinstance(value, (int, float)):
                    model_metrics[metric] = value
                elif isinstance(value, dict) and 'mean' in value:
                    model_metrics[metric] = value['mean']
        
        comparison_data.append(model_metrics)
    
    df_comparison = pd.DataFrame(comparison_data)
    metrics_to_plot = [m for m in df_comparison.columns if m != 'Modelo' and 
                      isinstance(df_comparison[m].iloc[0], (int, float))]
    
    if metrics_to_plot:
        fig = px.bar(
            df_comparison.melt(id_vars=['Modelo'], value_vars=metrics_to_plot),
            x='Modelo', y='value', color='variable', barmode='group',
            labels={'value': 'Valor', 'variable': 'Métrica'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_details(successful_models):
    """Muestra detalles de cada modelo"""
    st.markdown("### 🔍 Detalle por Modelo")
    tabs = st.tabs([f"{model['model_name']}" for model in successful_models])
    
    for i, model_data in enumerate(successful_models):
        with tabs[i]:
            st.subheader(f"Modelo: {model_data['model_name']}")
            
            show_model_metrics(model_data)
            
            show_confusion_matrix(model_data, i)

def show_model_metrics(model_data):
    """Muestra las métricas de un modelo"""
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
                delta_str = f"±{value['std']:.4f}" if 'std' in value else None
                st.metric(label=metric, value=f"{value['mean']:.4f}", delta=delta_str)
            else:
                st.metric(label=metric, value=str(value))
            metric_count += 1

def show_confusion_matrix(model_data, model_index):
    """Muestra la matriz de confusión si está disponible"""
    if 'Confusion Matrix' not in model_data.get('scores', {}):
        return
        
    st.markdown("#### 🧮 Matriz de Confusión")
    
    cm_data = model_data['scores']['Confusion Matrix']
    
    if not (isinstance(cm_data, dict) and 'matrix' in cm_data and 'labels' in cm_data):
        st.warning("⚠️ Formato de matriz de confusión no reconocido")
        return
    
    matrix = np.array(cm_data['matrix'])
    labels = [str(l) for l in cm_data['labels']]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels, 
        y=labels,
        hoverongaps=False,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Cantidad")
    ))

    annotations = []
    for r_idx, row in enumerate(matrix):
        for c_idx, value in enumerate(row):
            text_color = 'white' if value > matrix.max() / 2 else 'black'
            annotations.append(
                go.layout.Annotation(
                    text=str(int(value)),
                    x=labels[c_idx], 
                    y=labels[r_idx],
                    xref='x1', 
                    yref='y1', 
                    showarrow=False,
                    font=dict(color=text_color, size=14)
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
    
    st.plotly_chart(fig, use_container_width=True, key=f'confusion_matrix_{model_data["model_name"]}_{model_index}')

def clean_dataframe_for_json(df):
    """
    Limpia un DataFrame para asegurar que sea JSON-serializable,
    manejando valores NaN, infinitos y tipos de datos problemáticos.
    """
    df_clean = df.copy()
    
    # Reemplazar valores infinitos y NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Para cada columna, manejar según su tipo
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Para columnas de texto, convertir NaN a string vacío
            df_clean[col] = df_clean[col].fillna('')
            # Asegurar que todos los valores sean strings
            df_clean[col] = df_clean[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            # Para columnas numéricas
            if df_clean[col].isna().any():
                # Si hay NaN, rellenar con 0 como fallback seguro
                df_clean[col] = df_clean[col].fillna(0.0)
            
            # Convertir a float64 estándar para evitar problemas con tipos NumPy
            if df_clean[col].dtype in ['float16', 'float32']:
                df_clean[col] = df_clean[col].astype('float64')
            elif df_clean[col].dtype in ['int8', 'int16', 'int32']:
                df_clean[col] = df_clean[col].astype('int64')
        elif pd.api.types.is_bool_dtype(df_clean[col]):
            # Para columnas booleanas, convertir NaN a False
            df_clean[col] = df_clean[col].fillna(False)
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            # Para fechas, convertir a string
            df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
        else:
            # Para otros tipos, convertir a string
            df_clean[col] = df_clean[col].astype(str).fillna('')
    
    return df_clean

def validate_json_serializable(obj, max_depth=3, current_depth=0):
    """
    Valida que un objeto sea JSON-serializable de forma recursiva.
    Retorna True si es serializable, False si no.
    """
    if current_depth > max_depth:
        return True  # Evitar recursión infinita
    
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            # Verificar valores problemáticos en números
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return False
            return True
        elif isinstance(obj, (list, tuple)):
            return all(validate_json_serializable(item, max_depth, current_depth + 1) for item in obj)
        elif isinstance(obj, dict):
            return all(
                isinstance(k, str) and validate_json_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()
            )
        elif isinstance(obj, pd.DataFrame):
            # Para DataFrames, verificar una muestra pequeña
            sample_size = min(10, len(obj))
            sample_df = obj.head(sample_size)
            return validate_json_serializable(sample_df.to_dict('records'), max_depth, current_depth + 1)
        elif hasattr(obj, 'tolist'):  # Arrays de NumPy
            return validate_json_serializable(obj.tolist(), max_depth, current_depth + 1)
        else:
            # Intentar serialización directa como test
            json.dumps(obj)
            return True
    except (TypeError, ValueError, OverflowError):
        return False

def clean_training_params(training_params):
    """
    Limpia los parámetros de entrenamiento para asegurar compatibilidad JSON.
    """
    cleaned_params = training_params.copy()
    
    # Limpiar el DataFrame si existe
    if 'df' in cleaned_params and isinstance(cleaned_params['df'], pd.DataFrame):
        st.info("🧹 Limpiando datos para garantizar compatibilidad JSON...")
        
        original_shape = cleaned_params['df'].shape
        cleaned_df = clean_dataframe_for_json(cleaned_params['df'])
        
        # Verificar que el DataFrame limpio sea serializable
        if not validate_json_serializable(cleaned_df):
            st.error("❌ Error: No se pudo limpiar completamente el DataFrame para JSON")
            return None
        
        cleaned_params['df'] = cleaned_df
        
        # Mostrar información sobre la limpieza
        nan_count_before = training_params['df'].isna().sum().sum()
        nan_count_after = cleaned_df.isna().sum().sum()
        
        if nan_count_before > 0:
            st.success(f"✅ Limpieza completada: {nan_count_before} valores NaN/infinitos procesados")
        
        if original_shape != cleaned_df.shape:
            st.warning(f"⚠️ El tamaño del dataset cambió: {original_shape} → {cleaned_df.shape}")
    
    # Limpiar otros parámetros que puedan tener valores problemáticos
    for key, value in cleaned_params.items():
        if key != 'df':
            if isinstance(value, (np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    cleaned_params[key] = 0 if np.isnan(value) else (999999 if np.isposinf(value) else -999999)
                else:
                    cleaned_params[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, dict):
                # Limpiar diccionarios recursivamente (como hyperparams)
                cleaned_params[key] = clean_dict_for_json(value)
    
    return cleaned_params

def clean_dict_for_json(d):
    """
    Limpia un diccionario de forma recursiva para JSON.
    """
    if not isinstance(d, dict):
        return d
    
    cleaned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            cleaned[key] = clean_dict_for_json(value)
        elif isinstance(value, (np.integer, np.floating)):
            if np.isnan(value) or np.isinf(value):
                cleaned[key] = None
            else:
                cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                float(item) if isinstance(item, np.floating) else 
                int(item) if isinstance(item, np.integer) else 
                None if (isinstance(item, float) and (np.isnan(item) or np.isinf(item))) else 
                item
                for item in value
            ]
        else:
            cleaned[key] = value
    
    return cleaned

def show_data_quality_report(df):
    """
    Muestra un reporte de calidad de los datos.
    """
    with st.expander("📋 Reporte de Calidad de Datos", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de filas", f"{len(df):,}")
            st.metric("Total de columnas", len(df.columns))
        
        with col2:
            nan_count = df.isna().sum().sum()
            st.metric("Valores faltantes", f"{nan_count:,}")
            
            inf_count = 0
            for col in df.select_dtypes(include=[np.number]).columns:
                inf_count += np.isinf(df[col]).sum()
            st.metric("Valores infinitos", f"{inf_count:,}")
        
        with col3:
            duplicate_count = df.duplicated().sum()
            st.metric("Filas duplicadas", f"{duplicate_count:,}")
        
        # Mostrar detalles por columna si hay problemas
        if nan_count > 0 or inf_count > 0:
            st.markdown("#### 🔍 Detalles por columna:")
            problem_cols = []
            
            for col in df.columns:
                col_nans = df[col].isna().sum()
                col_infs = 0
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_infs = np.isinf(df[col]).sum()
                
                if col_nans > 0 or col_infs > 0:
                    problem_cols.append({
                        'Columna': col,
                        'Tipo': str(df[col].dtype),
                        'NaN': col_nans,
                        'Infinitos': col_infs,
                        '% Problemático': f"{((col_nans + col_infs) / len(df) * 100):.1f}%"
                    })
            
            if problem_cols:
                st.dataframe(pd.DataFrame(problem_cols), use_container_width=True)
        else:
            st.success("✅ No se detectaron problemas de calidad en los datos")