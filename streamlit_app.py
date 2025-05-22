"""
Interfaz Streamlit para la plataforma de aprendizaje supervisado distribuido.
Esta interfaz permite entrenar modelos, visualizar métricas y hacer predicciones.
"""
import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import sys
import ray
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_registry import ModelRegistry
from training.distributed_pipeline import DistributedTrainingPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Inicializar registro de modelos
model_registry = ModelRegistry()

# Configuración de la página
st.set_page_config(
    page_title="Plataforma de Aprendizaje Supervisado Distribuido",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Plataforma de Aprendizaje Supervisado Distribuido")

# Barra lateral para navegación
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Seleccione una página:",
    ["Inicio", "Entrenamiento", "Modelos", "Predicciones", "Monitorización", "Configuración"]
)

# Página principal
if pagina == "Inicio":
    st.header("🏠 Bienvenido a la Plataforma de Aprendizaje Supervisado Distribuido")
    st.markdown("""
    Esta plataforma proporciona herramientas para el entrenamiento distribuido de modelos de aprendizaje supervisado,
    gestión de modelos, inferencia, y monitorización del sistema.
    
    ### Funcionalidades principales:
    
    - **Entrenamiento**: Entrene modelos de aprendizaje supervisado utilizando Ray para distribuir el proceso.
    - **Modelos**: Gestione y examine los modelos entrenados en el registro.
    - **Predicciones**: Utilice los modelos entrenados para hacer predicciones con nuevos datos.
    - **Métricas**: Visualice y compare las métricas de rendimiento de diferentes modelos.
    - **Monitorización**: Supervise los recursos del sistema y el rendimiento del cluster Ray.
    - **Configuración**: Configure los parámetros del sistema y la integración con servicios externos.
    """)

# Página de Entrenamiento
elif pagina == "Entrenamiento":
    st.header("🚀 Entrenamiento de Modelos")
    st.markdown("""
    Esta sección permite entrenar modelos de aprendizaje supervisado utilizando Ray para distribuir el proceso.
    Elija un conjunto de datos, configure los parámetros del modelo y comience el entrenamiento.
    """)
    
    # Carga de datos
    st.subheader("1️⃣ Carga de datos")
    
    # Opciones para cargar datos
    data_option = st.radio(
        "Seleccione la fuente de datos:",
        ["Conjuntos de datos de ejemplo", "Subir archivo CSV", "Ingresar ruta a un archivo"]
    )
    
    df = None
    
    if data_option == "Conjuntos de datos de ejemplo":
        dataset_choice = st.selectbox(
            "Seleccione un conjunto de datos:",
            ["Iris", "Wine", "Breast Cancer"]
        )
        
        try:
            if dataset_choice == "Iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                target_column = 'target'
            elif dataset_choice == "Wine":
                from sklearn.datasets import load_wine
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                target_column = 'target'
            elif dataset_choice == "Breast Cancer":
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                target_column = 'target'
                
            st.success(f"Conjunto de datos '{dataset_choice}' cargado correctamente.")
            st.write(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
            st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al cargar el conjunto de datos: {str(e)}")
    
    elif data_option == "Subir archivo CSV":
        uploaded_file = st.file_uploader("Suba un archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Archivo CSV cargado correctamente.")
                st.write(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
                st.dataframe(df.head(), use_container_width=True)
                
                # Seleccionar columna objetivo
                target_column = st.selectbox(
                    "Seleccione la columna objetivo:",
                    df.columns.tolist()
                )
                
            except Exception as e:
                st.error(f"Error al cargar el archivo CSV: {str(e)}")
    
    elif data_option == "Ingresar ruta a un archivo":
        file_path = st.text_input("Introduzca la ruta completa al archivo CSV:")
        
        if file_path:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    st.success(f"Archivo CSV cargado desde '{file_path}'.")
                    st.write(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Seleccionar columna objetivo
                    target_column = st.selectbox(
                        "Seleccione la columna objetivo:",
                        df.columns.tolist()
                    )
                else:
                    st.error(f"El archivo '{file_path}' no existe.")
            except Exception as e:
                st.error(f"Error al cargar el archivo CSV: {str(e)}")
    
    # Continuar con el proceso de entrenamiento si se ha cargado un dataset
    if df is not None:
        st.subheader("2️⃣ Configuración del modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Seleccione el tipo de modelo:",
                [
                    "RandomForestClassifier", 
                    "LogisticRegression", 
                    "GradientBoostingClassifier",
                    "SVC",
                    "KNeighborsClassifier",
                    "DecisionTreeClassifier",
                    "AdaBoostClassifier",
                    "ExtraTreesClassifier",
                    "SGDClassifier",
                    "GaussianNB"
                ]
            )
            
            test_size = st.slider(
                "Tamaño del conjunto de prueba:",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
            
            random_state = st.number_input(
                "Estado aleatorio (para reproducibilidad):",
                min_value=0,
                max_value=100,
                value=42
            )
        
        with col2:
            # Parámetros específicos del modelo
            if model_type == "RandomForestClassifier":
                # Parámetros básicos
                n_estimators = st.slider(
                    "Número de árboles:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                max_depth = st.slider(
                    "Profundidad máxima:",
                    min_value=1,
                    max_value=30,
                    value=10
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    min_samples_split = st.slider(
                        "Muestras mínimas para dividir:",
                        min_value=2,
                        max_value=20,
                        value=2
                    )
                    min_samples_leaf = st.slider(
                        "Muestras mínimas por hoja:",
                        min_value=1,
                        max_value=20,
                        value=1
                    )
                    max_features = st.select_slider(
                        "Características máximas:",
                        options=["sqrt", "log2", None],
                        value="sqrt"
                    )
                    criterion = st.select_slider(
                        "Criterio de división:",
                        options=["gini", "entropy", "log_loss"],
                        value="gini"
                    )
                
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "criterion": criterion,
                    "random_state": random_state
                }
            
            elif model_type == "LogisticRegression":
                # Parámetros básicos
                C = st.slider(
                    "Parámetro de regularización (C):",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01
                )
                max_iter = st.slider(
                    "Número máximo de iteraciones:",
                    min_value=100,
                    max_value=1000,
                    value=100,
                    step=100
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    solver = st.select_slider(
                        "Algoritmo de optimización:",
                        options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                        value="lbfgs"
                    )
                    penalty = st.select_slider(
                        "Tipo de penalización:",
                        options=["l2", "l1", "elasticnet", "none"],
                        value="l2"
                    )
                    l1_ratio = st.slider(
                        "Ratio ElasticNet (solo para elasticnet):",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        disabled=(penalty != "elasticnet")
                    )
                
                model_params = {
                    "C": C,
                    "max_iter": max_iter,
                    "solver": solver,
                    "penalty": penalty,
                    "random_state": random_state
                }
                
                # Solo añadir l1_ratio si se está usando elasticnet
                if penalty == "elasticnet":
                    model_params["l1_ratio"] = l1_ratio
            
            elif model_type == "GradientBoostingClassifier":
                # Parámetros básicos
                n_estimators = st.slider(
                    "Número de etapas:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                learning_rate = st.slider(
                    "Tasa de aprendizaje:",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    max_depth = st.slider(
                        "Profundidad máxima:",
                        min_value=1,
                        max_value=20,
                        value=3
                    )
                    min_samples_split = st.slider(
                        "Muestras mínimas para dividir:",
                        min_value=2,
                        max_value=20,
                        value=2
                    )
                    subsample = st.slider(
                        "Proporción de muestras para entrenar:",
                        min_value=0.1,
                        max_value=1.0,
                        value=1.0,
                        step=0.1
                    )
                    loss = st.select_slider(
                        "Función de pérdida:",
                        options=["log_loss", "exponential"],
                        value="log_loss"
                    )
                
                model_params = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "subsample": subsample,
                    "loss": loss,
                    "random_state": random_state
                }
                
            elif model_type == "SVC":
                # Parámetros básicos
                C = st.slider(
                    "Parámetro de regularización (C):",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                kernel = st.select_slider(
                    "Tipo de kernel:",
                    options=["rbf", "linear", "poly", "sigmoid"],
                    value="rbf"
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    gamma = st.select_slider(
                        "Coeficiente gamma:",
                        options=["scale", "auto"] + [0.001, 0.01, 0.1, 1.0, 10.0],
                        value="scale"
                    )
                    degree = st.slider(
                        "Grado del polinomio (solo para kernel poly):",
                        min_value=2,
                        max_value=10,
                        value=3,
                        disabled=(kernel != "poly")
                    )
                    probability = st.checkbox("Habilitar estimación de probabilidad", value=True)
                    
                model_params = {
                    "C": C,
                    "kernel": kernel,
                    "gamma": gamma,
                    "probability": probability,
                    "random_state": random_state
                }
                
                # Solo añadir degree si se usa kernel polinómico
                if kernel == "poly":
                    model_params["degree"] = degree
            
            elif model_type == "KNeighborsClassifier":
                # Parámetros básicos
                n_neighbors = st.slider(
                    "Número de vecinos:",
                    min_value=1,
                    max_value=20,
                    value=5
                )
                weights = st.select_slider(
                    "Tipo de peso:",
                    options=["uniform", "distance"],
                    value="uniform"
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    algorithm = st.select_slider(
                        "Algoritmo:",
                        options=["auto", "ball_tree", "kd_tree", "brute"],
                        value="auto"
                    )
                    p = st.slider(
                        "Parámetro p para la distancia Minkowski:",
                        min_value=1,
                        max_value=2,
                        value=2
                    )
                
                model_params = {
                    "n_neighbors": n_neighbors,
                    "weights": weights,
                    "algorithm": algorithm,
                    "p": p
                }
                
            elif model_type == "DecisionTreeClassifier":
                # Parámetros básicos
                max_depth = st.slider(
                    "Profundidad máxima:",
                    min_value=1,
                    max_value=30,
                    value=10
                )
                criterion = st.select_slider(
                    "Criterio de división:",
                    options=["gini", "entropy", "log_loss"],
                    value="gini"
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    min_samples_split = st.slider(
                        "Muestras mínimas para dividir:",
                        min_value=2,
                        max_value=20,
                        value=2
                    )
                    min_samples_leaf = st.slider(
                        "Muestras mínimas por hoja:",
                        min_value=1,
                        max_value=20,
                        value=1
                    )
                    max_features = st.select_slider(
                        "Características máximas:",
                        options=["sqrt", "log2", None],
                        value=None
                    )
                
                model_params = {
                    "max_depth": max_depth,
                    "criterion": criterion,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "random_state": random_state
                }
                
            elif model_type == "AdaBoostClassifier":
                # Parámetros básicos
                n_estimators = st.slider(
                    "Número de estimadores:",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10
                )
                learning_rate = st.slider(
                    "Tasa de aprendizaje:",
                    min_value=0.01,
                    max_value=2.0,
                    value=1.0,
                    step=0.01
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    algorithm = st.select_slider(
                        "Algoritmo:",
                        options=["SAMME", "SAMME.R"],
                        value="SAMME.R"
                    )
                
                model_params = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "algorithm": algorithm,
                    "random_state": random_state
                }
                
            elif model_type == "ExtraTreesClassifier":
                # Parámetros básicos
                n_estimators = st.slider(
                    "Número de árboles:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                max_depth = st.slider(
                    "Profundidad máxima:",
                    min_value=1,
                    max_value=30,
                    value=10
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    min_samples_split = st.slider(
                        "Muestras mínimas para dividir:",
                        min_value=2,
                        max_value=20,
                        value=2
                    )
                    min_samples_leaf = st.slider(
                        "Muestras mínimas por hoja:",
                        min_value=1,
                        max_value=20,
                        value=1
                    )
                    max_features = st.select_slider(
                        "Características máximas:",
                        options=["sqrt", "log2", None],
                        value="sqrt"
                    )
                    criterion = st.select_slider(
                        "Criterio de división:",
                        options=["gini", "entropy", "log_loss"],
                        value="gini"
                    )
                
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "criterion": criterion,
                    "random_state": random_state
                }
                
            elif model_type == "SGDClassifier":
                # Parámetros básicos
                alpha = st.slider(
                    "Término de regularización (alpha):",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.0001,
                    step=0.0001,
                    format="%.4f"
                )
                max_iter = st.slider(
                    "Número máximo de iteraciones:",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100
                )
                
                # Sección de parámetros avanzados
                with st.expander("Parámetros avanzados"):
                    loss = st.select_slider(
                        "Función de pérdida:",
                        options=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
                        value="hinge"
                    )
                    penalty = st.select_slider(
                        "Tipo de penalización:",
                        options=["l2", "l1", "elasticnet"],
                        value="l2"
                    )
                    l1_ratio = st.slider(
                        "Ratio ElasticNet (solo para elasticnet):",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.15,
                        step=0.05,
                        disabled=(penalty != "elasticnet")
                    )
                    learning_rate = st.select_slider(
                        "Tipo de tasa de aprendizaje:",
                        options=["constant", "optimal", "invscaling", "adaptive"],
                        value="optimal"
                    )
                
                model_params = {
                    "alpha": alpha,
                    "max_iter": max_iter,
                    "loss": loss,
                    "penalty": penalty,
                    "learning_rate": learning_rate,
                    "random_state": random_state
                }
                
                if penalty == "elasticnet":
                    model_params["l1_ratio"] = l1_ratio
                
            elif model_type == "GaussianNB":
                # Parámetros básicos
                var_smoothing = st.slider(
                    "Suavizado de varianza:",
                    min_value=1e-10,
                    max_value=1e-5,
                    value=1e-9,
                    format="%.2e"
                )
                
                model_params = {
                    "var_smoothing": var_smoothing
                }
        
        # Opciones de distribución
        st.subheader("3️⃣ Configuración de distribución")
        
        use_ray = st.checkbox("Utilizar Ray para entrenamiento distribuido", value=True)
        
        if use_ray:
            num_workers = st.slider(
                "Número de trabajadores para Ray:",
                min_value=1,
                max_value=psutil.cpu_count(),
                value=min(4, psutil.cpu_count())
            )
            
            # Opción para entrenar múltiples modelos en paralelo
            train_multiple = st.checkbox("Entrenar múltiples modelos en paralelo", value=False)
            
            if train_multiple:
                st.subheader("Entrenamiento Distribuido Personalizado")
                
                # Explicación mejorada con columnas
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(r"""
                    $$\textbf{Fundamentos del Entrenamiento Distribuido}$$
                    
                    Un paradigma de computación paralela que divide el procesamiento entre múltiples unidades de cómputo utilizando un modelo formal:
                    
                    $$\mathcal{D} = (G, \mathcal{M}, \mathcal{P}, \mathcal{C})$$
                    
                    Donde:
                    - $$G = (V, E)$$ es el grafo de computación distribuida
                    - $$\mathcal{M}$$ es el conjunto de modelos $$\{f_{\theta_1}, f_{\theta_2}, ..., f_{\theta_m}\}$$
                    - $$\mathcal{P}$$ es el conjunto de trabajadores $$\{P_1, P_2, ..., P_n\}$$
                    - $$\mathcal{C}$$ son los canales de comunicación
                    
                    El factor de aceleración teórico está dado por:
                    
                    $$\text{SpeedUp} = \frac{T_{secuencial}}{T_{paralelo}} \leq \min(|\mathcal{M}|, |\mathcal{P}|)$$
                    """)
                
                with col2:
                    st.markdown(r"""
                    $$\textbf{Ventajas Cuantificables:}$$
                    
                    $$\checkmark$$ $$\text{Reducción tiempo: }$$ 
                    $$\Delta T = T_{secuencial} \cdot \left(1 - \frac{1}{\min(|\mathcal{M}|, |\mathcal{P}|)}\right)$$
                    
                    $$\checkmark$$ $$\text{Paralelismo efectivo: }$$ 
                    $$PE = \frac{\sum_{i=1}^{|\mathcal{P}|} \text{tiempo\_activo}(P_i)}{|\mathcal{P}| \cdot \max_i \text{tiempo\_total}(P_i)}$$
                    
                    $$\checkmark$$ $$\text{Utilización recursos: }$$ 
                    $$\eta = \frac{\sum_{i=1}^{|\mathcal{P}|} \text{carga}(P_i)}{|\mathcal{P}| \cdot \text{capacidad\_máxima}} \approx 100\%$$
                    
                    $$\checkmark$$ $$\text{Tolerancia a fallos: }$$ 
                    $$R_{system} = 1 - \prod_{i=1}^{|\mathcal{P}|} (1 - R_i)$$
                    """)
                
                # Visualización de modelos disponibles
                st.subheader("Selección personalizada de modelos")
                st.info(f"El modelo **{model_type}** ya está configurado como modelo principal")
                
                # Mostramos los modelos disponibles en una forma más visual
                st.write("Selecciona qué modelos adicionales quieres entrenar en paralelo:")
                
                # Selección de modelos adicionales
                additional_models = []
                
                # Usamos tabs para organizar los modelos disponibles
                tab1, tab2 = st.tabs(["Modelos básicos", "Modelos avanzados"])
                
                with tab1:
                    # Presentamos las opciones básicas en formato de tarjetas con columnas
                    col1, col2, col3 = st.columns(3)
                    
                    # Opción RandomForest
                    with col1:
                        if model_type != "RandomForestClassifier":
                            use_rf = st.checkbox("Random Forest", value=True)
                            if use_rf:
                                additional_models.append("RandomForestClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    rf_estimators = st.slider(
                                        "Número de árboles:",
                                        min_value=10,
                                        max_value=500,
                                        value=100,
                                        step=10,
                                        key="rf_estimators"
                                    )
                                    rf_max_depth = st.slider(
                                        "Profundidad máxima:",
                                        min_value=1,
                                        max_value=30,
                                        value=10,
                                        key="rf_max_depth"
                                    )
                                    rf_criterion = st.select_slider(
                                        "Criterio:",
                                        options=["gini", "entropy", "log_loss"],
                                        value="gini",
                                        key="rf_criterion"
                                    )
                    
                    # Opción LogisticRegression
                    with col2:
                        if model_type != "LogisticRegression":
                            use_lr = st.checkbox("Regresión Logística", value=True)
                            if use_lr:
                                additional_models.append("LogisticRegression")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    lr_C = st.slider(
                                        "Parámetro C:",
                                        min_value=0.01,
                                        max_value=10.0,
                                        value=1.0,
                                        step=0.01,
                                        key="lr_C"
                                    )
                                    lr_max_iter = st.slider(
                                        "Iteraciones máximas:",
                                        min_value=100,
                                        max_value=1000,
                                        value=100,
                                        step=100,
                                        key="lr_max_iter"
                                    )
                                    lr_solver = st.select_slider(
                                        "Solver:",
                                        options=["lbfgs", "liblinear", "saga"],
                                        value="lbfgs",
                                        key="lr_solver"
                                    )
                    
                    # Opción GradientBoosting
                    with col3:
                        if model_type != "GradientBoostingClassifier":
                            use_gb = st.checkbox("Gradient Boosting", value=True)
                            if use_gb:
                                additional_models.append("GradientBoostingClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    gb_estimators = st.slider(
                                        "Número de estimadores:",
                                        min_value=10,
                                        max_value=500,
                                        value=100,
                                        step=10,
                                        key="gb_estimators"
                                    )
                                    gb_learning_rate = st.slider(
                                        "Tasa de aprendizaje:",
                                        min_value=0.01,
                                        max_value=1.0,
                                        value=0.1,
                                        step=0.01,
                                        key="gb_learning_rate"
                                    )
                                    gb_max_depth = st.slider(
                                        "Profundidad máxima:",
                                        min_value=1,
                                        max_value=10,
                                        value=3,
                                        key="gb_max_depth"
                                    )
                
                with tab2:
                    # Presentamos opciones avanzadas en formato de tarjetas con columnas
                    col1, col2, col3 = st.columns(3)
                    
                    # Opción SVC
                    with col1:
                        if model_type != "SVC":
                            use_svc = st.checkbox("Support Vector Classifier", value=False)
                            if use_svc:
                                additional_models.append("SVC")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    svc_C = st.slider(
                                        "Parámetro C:",
                                        min_value=0.1,
                                        max_value=10.0,
                                        value=1.0,
                                        step=0.1,
                                        key="svc_C"
                                    )
                                    svc_kernel = st.select_slider(
                                        "Kernel:",
                                        options=["rbf", "linear", "poly", "sigmoid"],
                                        value="rbf",
                                        key="svc_kernel"
                                    )
                                    svc_gamma = st.select_slider(
                                        "Gamma:",
                                        options=["scale", "auto"],
                                        value="scale",
                                        key="svc_gamma"
                                    )
                    
                    # Opción KNN
                    with col2:
                        if model_type != "KNeighborsClassifier":
                            use_knn = st.checkbox("K-Nearest Neighbors", value=False)
                            if use_knn:
                                additional_models.append("KNeighborsClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    knn_n_neighbors = st.slider(
                                        "Número de vecinos:",
                                        min_value=1,
                                        max_value=20,
                                        value=5,
                                        key="knn_n_neighbors"
                                    )
                                    knn_weights = st.select_slider(
                                        "Pesos:",
                                        options=["uniform", "distance"],
                                        value="uniform",
                                        key="knn_weights"
                                    )
                                    knn_algorithm = st.select_slider(
                                        "Algoritmo:",
                                        options=["auto", "ball_tree", "kd_tree", "brute"],
                                        value="auto",
                                        key="knn_algorithm"
                                    )
                    
                    # Opción AdaBoost
                    with col3:
                        if model_type != "AdaBoostClassifier":
                            use_ada = st.checkbox("AdaBoost", value=False)
                            if use_ada:
                                additional_models.append("AdaBoostClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    ada_estimators = st.slider(
                                        "Número de estimadores:",
                                        min_value=10,
                                        max_value=200,
                                        value=50,
                                        step=10,
                                        key="ada_estimators"
                                    )
                                    ada_learning_rate = st.slider(
                                        "Tasa de aprendizaje:",
                                        min_value=0.01,
                                        max_value=2.0,
                                        value=1.0,
                                        step=0.01,
                                        key="ada_learning_rate"
                                    )
                    
                    # Segunda fila para modelos avanzados
                    col1, col2, col3 = st.columns(3)
                    
                    # Opción ExtraTrees
                    with col1:
                        if model_type != "ExtraTreesClassifier":
                            use_et = st.checkbox("Extra Trees", value=False)
                            if use_et:
                                additional_models.append("ExtraTreesClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    et_estimators = st.slider(
                                        "Número de árboles:",
                                        min_value=10,
                                        max_value=500,
                                        value=100,
                                        step=10,
                                        key="et_estimators"
                                    )
                                    et_max_depth = st.slider(
                                        "Profundidad máxima:",
                                        min_value=1,
                                        max_value=30,
                                        value=None,
                                        key="et_max_depth"
                                    )
                    
                    # Opción SGDClassifier
                    with col2:
                        if model_type != "SGDClassifier":
                            use_sgd = st.checkbox("SGD Classifier", value=False)
                            if use_sgd:
                                additional_models.append("SGDClassifier")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    sgd_loss = st.select_slider(
                                        "Función de pérdida:",
                                        options=["hinge", "log_loss", "modified_huber"],
                                        value="hinge",
                                        key="sgd_loss"
                                    )
                                    sgd_alpha = st.slider(
                                        "Alpha:",
                                        min_value=0.0001,
                                        max_value=0.01,
                                        value=0.0001,
                                        step=0.0001,
                                        format="%.4f",
                                        key="sgd_alpha"
                                    )
                    
                    # Opción GaussianNB
                    with col3:
                        if model_type != "GaussianNB":
                            use_gnb = st.checkbox("Naive Bayes", value=False)
                            if use_gnb:
                                additional_models.append("GaussianNB")
                                st.markdown("---")
                                # Expandir para mostrar parámetros
                                with st.expander("Configurar parámetros"):
                                    gnb_var_smoothing = st.slider(
                                        "Suavizado de varianza:",
                                        min_value=1e-10,
                                        max_value=1e-5,
                                        value=1e-9,
                                        format="%.2e",
                                        key="gnb_var_smoothing"
                                    )
                
                # Resumen visual de los modelos seleccionados
                st.subheader("Resumen de entrenamiento")
                if len(additional_models) > 0:
                    total_models = 1 + len(additional_models)
                    
                    # Crear un indicador visual del número de modelos
                    st.write(f"**Total de modelos a entrenar: {total_models}**")
                    
                    # Usar una barra de progreso más visual
                    progress_value = min(total_models / 10, 1.0)  # Normalizar a máximo 1.0
                    st.progress(progress_value)
                    
                    # Mostrar modelos en formato de tarjetas
                    st.write("**Modelos seleccionados:**")
                    
                    # Crear una cuadrícula visual para los modelos
                    cols = st.columns(3)
                    
                    # Añadir el modelo principal primero
                    model_cards = [{"name": model_type, "type": "principal", "color": "#0068c9"}]
                    
                    # Añadir los modelos adicionales
                    for model in additional_models:
                        model_cards.append({"name": model, "type": "adicional", "color": "#3a9e4f"})
                    
                    # Mostrar tarjetas para cada modelo
                    for i, model_card in enumerate(model_cards):
                        col_idx = i % 3
                        with cols[col_idx]:
                            st.markdown(
                                f"<div style='background-color: {model_card['color']}; padding: 10px; "
                                f"border-radius: 5px; text-align: center; color: white; margin-bottom: 10px;'>"
                                f"<h5>{model_card['name']}</h5>"
                                f"<p>Modelo {model_card['type']}</p>"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                    
                    # Añadir información sobre la capacidad de procesamiento
                    st.info(
                        f"📊 **Análisis de capacidad:** Con {num_workers} trabajadores en Ray, se podrán procesar "
                        f"hasta {min(num_workers, total_models)} modelos simultáneamente."
                    )
                    
                    # Mostrar estimación de tiempo
                    st.write("**Estimación de tiempo:**")
                    st.markdown(
                        "⏱️ El entrenamiento distribuido puede reducir el tiempo total hasta un "
                        f"{min(num_workers, total_models) * 100 // total_models}% comparado con el entrenamiento secuencial."
                    )
                    
                    # Resumen final con los nombres de los modelos
                    st.success(
                        f"Se entrenarán {total_models} modelos en paralelo: **{model_type}** (principal) y "
                        f"**{', '.join(additional_models)}**."
                    )
                else:
                    st.warning("No has seleccionado modelos adicionales. Solo se entrenará el modelo principal.")
        else:
            num_workers = 1
            train_multiple = False
        
        # Botón para iniciar el entrenamiento
        if st.button("Entrenar modelo"):
            try:
                with st.spinner("Preparando entrenamiento..."):
                    # Preparar los datos
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]
                    
                    # Crear pipeline de entrenamiento
                    pipeline = DistributedTrainingPipeline(registry=model_registry, init_ray=use_ray)
                    
                    # Configurar modelos según la opción seleccionada
                    if train_multiple and use_ray:
                        # Configurar múltiples modelos
                        models_config = [
                            # El modelo configurado por el usuario
                            {
                                'name': f"{model_type}_primary",
                                'estimator': eval(model_type),
                                'params': model_params
                            }
                        ]
                        
                        # Añadir modelos adicionales según la selección del usuario con los parámetros personalizados
                        if model_type != "RandomForestClassifier" and "RandomForestClassifier" in additional_models:
                            models_config.append({
                                'name': "RandomForest_custom",
                                'estimator': RandomForestClassifier,
                                'params': {
                                    'n_estimators': rf_estimators,
                                    'max_depth': rf_max_depth,
                                    'criterion': rf_criterion,
                                    'random_state': random_state
                                }
                            })
                        
                        if model_type != "LogisticRegression" and "LogisticRegression" in additional_models:
                            models_config.append({
                                'name': "LogisticRegression_custom",
                                'estimator': LogisticRegression,
                                'params': {
                                    'C': lr_C,
                                    'max_iter': lr_max_iter,
                                    'solver': lr_solver,
                                    'random_state': random_state
                                }
                            })
                        
                        if model_type != "GradientBoostingClassifier" and "GradientBoostingClassifier" in additional_models:
                            models_config.append({
                                'name': "GradientBoosting_custom",
                                'estimator': GradientBoostingClassifier,
                                'params': {
                                    'n_estimators': gb_estimators,
                                    'learning_rate': gb_learning_rate,
                                    'max_depth': gb_max_depth,
                                    'random_state': random_state
                                }
                            })
                            
                        # Añadir los nuevos modelos avanzados si están seleccionados
                        if model_type != "SVC" and "SVC" in additional_models:
                            models_config.append({
                                'name': "SVC_custom",
                                'estimator': SVC,
                                'params': {
                                    'C': svc_C,
                                    'kernel': svc_kernel,
                                    'gamma': svc_gamma,
                                    'random_state': random_state,
                                    'probability': True
                                }
                            })
                            
                        if model_type != "KNeighborsClassifier" and "KNeighborsClassifier" in additional_models:
                            models_config.append({
                                'name': "KNN_custom",
                                'estimator': KNeighborsClassifier,
                                'params': {
                                    'n_neighbors': knn_n_neighbors,
                                    'weights': knn_weights,
                                    'algorithm': knn_algorithm
                                }
                            })
                            
                        if model_type != "AdaBoostClassifier" and "AdaBoostClassifier" in additional_models:
                            models_config.append({
                                'name': "AdaBoost_custom",
                                'estimator': AdaBoostClassifier,
                                'params': {
                                    'n_estimators': ada_estimators,
                                    'learning_rate': ada_learning_rate,
                                    'random_state': random_state
                                }
                            })
                            
                        if model_type != "ExtraTreesClassifier" and "ExtraTreesClassifier" in additional_models:
                            models_config.append({
                                'name': "ExtraTrees_custom",
                                'estimator': ExtraTreesClassifier,
                                'params': {
                                    'n_estimators': et_estimators,
                                    'max_depth': et_max_depth,
                                    'random_state': random_state
                                }
                            })
                            
                        if model_type != "SGDClassifier" and "SGDClassifier" in additional_models:
                            models_config.append({
                                'name': "SGD_custom",
                                'estimator': SGDClassifier,
                                'params': {
                                    'loss': sgd_loss,
                                    'alpha': sgd_alpha,
                                    'random_state': random_state,
                                    'max_iter': 1000
                                }
                            })
                            
                        if model_type != "GaussianNB" and "GaussianNB" in additional_models:
                            models_config.append({
                                'name': "NaiveBayes_custom",
                                'estimator': GaussianNB,
                                'params': {
                                    'var_smoothing': gnb_var_smoothing
                                }
                            })
                        
                        # Mostrar detalle de los modelos que se van a entrenar
                        st.subheader("Configuración de modelos para entrenamiento paralelo")
                        for i, model_config in enumerate(models_config):
                            model_name = model_config['name']
                            model_params = model_config['params']
                            
                            # Crear una representación visual de cada modelo
                            with st.expander(f"Modelo {i+1}: {model_name}"):
                                # Mostrar los parámetros en una tabla
                                params_df = pd.DataFrame(
                                    [(k, v) for k, v in model_params.items()],
                                    columns=["Parámetro", "Valor"]
                                )
                                st.table(params_df)
                        
                        st.info(f"Se entrenarán {len(models_config)} modelos en paralelo usando Ray.")
                    else:
                        # Configurar un solo modelo
                        models_config = [{
                            'name': model_type,
                            'estimator': eval(model_type),  # Convertir el nombre a la clase real
                            'params': model_params
                        }]
                    
                    # Entrenar el modelo
                    st.info("Iniciando entrenamiento distribuido...")
                    
                    if use_ray:
                        if train_multiple:
                            st.success(f"Se entrenarán {len(models_config)} modelos en paralelo utilizando Ray con {num_workers} trabajadores.")
                        else:
                            st.success(f"Se entrenará {model_type} de forma distribuida utilizando Ray con {num_workers} trabajadores.")
                        
                        # Explicación mejorada del entrenamiento distribuido
                        with st.expander("¿Cómo funciona el entrenamiento distribuido?"):
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown(r"""
                                $$\textbf{El entrenamiento distribuido con Ray: Fundamentos Matemáticos}$$
                                
                                1. $$\textbf{Topología del clúster:}$$ Ray crea una estructura computacional distribuida definida formalmente como un dígrafo ponderado $$\mathcal{G} = (V, E, \omega)$$, donde:
                                   - $$V = \{v_{\text{head}}, v_1, v_2, ..., v_n\}$$ representa el conjunto de nodos computacionales
                                   - $$E \subseteq V \times V$$ representa las conexiones de comunicación entre nodos
                                   - $$\omega: E \rightarrow \mathbb{R}^+$$ asigna a cada arista un peso que representa la latencia de comunicación
                                   
                                   La topología forma una estructura de estrella con matriz de adyacencia $$A_{n+1 \times n+1}$$ donde:
                                   $$A_{ij} = \begin{cases}
                                   \omega(v_i, v_j) & \text{si } (v_i, v_j) \in E \\
                                   \infty & \text{en otro caso}
                                   \end{cases}$$
                                
                                2. $$\textbf{Modelo formal de memoria distribuida:}$$ El Object Store implementa un sistema de memoria distribuida compartida (DSM) definido como una tupla $$(S, \mathcal{O}, \Phi, \mathcal{A})$$, donde:
                                   - $$S = \{s_{\text{head}}, s_1, ..., s_n\}$$ es el conjunto de espacios de almacenamiento local
                                   - $$\mathcal{O}$$ es el conjunto de objetos serializables (matrices, tensores, modelos)
                                   - $$\Phi: \mathcal{O} \rightarrow \{0,1\}^d$$ es una función de hash consistente que mapea objetos a identificadores únicos de $$d$$ bits
                                   - $$\mathcal{A}: \Phi(\mathcal{O}) \times S \rightarrow \{0,1\}$$ es la función de ubicación que determina la presencia de un objeto en un almacenamiento
                                   
                                   Así, para los datos de entrenamiento $$X \in \mathbb{R}^{m \times p}$$ e $$y \in \mathbb{R}^{m}$$:
                                   $$\text{ref}_X = \Phi(X) \in \{0,1\}^d \quad \text{y} \quad \text{ref}_y = \Phi(y) \in \{0,1\}^d$$
                                   
                                   La recuperación de datos sigue la ecuación:
                                   $$X = \Phi^{-1}(\text{ref}_X) \quad \text{si} \quad \exists s_i \in S: \mathcal{A}(\text{ref}_X, s_i) = 1$$
                                
                                3. $$\textbf{Cálculo distribuido mediante actores:}$$ El sistema utiliza un modelo formal de actores $$\mathcal{A} = (\Sigma, \gamma, \Pi, \mu)$$ donde:
                                   - $$\Sigma$$ es el conjunto de todos los posibles estados de los actores
                                   - $$\gamma: V \rightarrow 2^{\Sigma}$$ asigna a cada nodo el conjunto de actores que ejecuta
                                   - $$\Pi$$ es el conjunto de funciones de procesamiento paralelizables (entrenamiento de modelos)
                                   - $$\mu: \Sigma \times \Pi \times \mathcal{O}^* \rightarrow \Sigma \times \mathcal{O}$$ es la función de transición de estados
                                   
                                   Para cada modelo $$f_{\theta_i}$$ con hiperparámetros $$\theta_i$$, el proceso de entrenamiento se formaliza como:
                                   $$\text{train}_i : \mathcal{O}^* \rightarrow \mathcal{O}, \quad \text{train}_i(\text{ref}_X, \text{ref}_y, \theta_i) = \Phi(f_{\theta_i})$$
                                   
                                   Que resuelve el problema de optimización:
                                   $$f_{\theta_i} = \underset{\theta}{\text{argmin}} \mathcal{L}(f_{\theta}(X), y, \lambda\|\theta\|_p)$$
                                   
                                   donde $$\mathcal{L}$$ es la función de pérdida, $$\lambda$$ es el parámetro de regularización, y $$\|\theta\|_p$$ es la norma-$$p$$ de los parámetros.
                                
                                4. $$\textbf{Planificación óptima de tareas:}$$ Ray implementa un algoritmo de planificación basado en un problema de optimización combinatoria $$\mathcal{P} = (J, M, \tau, C, F)$$ donde:
                                   - $$J = \{j_1, j_2, ..., j_m\}$$ es el conjunto de tareas (entrenamientos)
                                   - $$M = \{m_1, m_2, ..., m_n\}$$ es el conjunto de máquinas (trabajadores)
                                   - $$\tau: J \times M \rightarrow \mathbb{R}^+$$ es la función que estima el tiempo de ejecución
                                   - $$C: M \rightarrow \mathbb{R}^k_+$$ define las capacidades de recursos de cada máquina en $$k$$ dimensiones
                                   - $$F: 2^{J \times M} \rightarrow \mathbb{R}$$ es la función objetivo a minimizar
                                   
                                                                      
                                   El problema de asignación óptima se formula como:
                                   $$\begin{align}
                                   \min_{w} \quad & \max_{m_i \in M} \sum_{j_k \in J} w_{ki} \cdot \tau(j_k, m_i) \\
                                   \text{s.a.} \quad & \sum_{m_i \in M} w_{ki} = 1 \quad \forall j_k \in J \\
                                   & \sum_{j_k \in J} w_{ki} \cdot r_k^{(d)} \leq C_i^{(d)} \quad \forall m_i \in M, \forall d \in \{1,2,...,k\} \\
                                   & w_{ki} \in \{0,1\} \quad \forall j_k \in J, \forall m_i \in M
                                   \end{align}$$
                                   
                                   donde $$w_{ki}$$ indica si la tarea $$j_k$$ se asigna a la máquina $$m_i$$, $$r_k^{(d)}$$ es el requisito de la tarea $$j_k$$ en la dimensión de recurso $$d$$, y $$C_i^{(d)}$$ es la capacidad de la máquina $$m_i$$ en la dimensión $$d$$.
                                
                                5. $$\textbf{Cálculo distribuido de gradientes:}$$ En modelos entrenados mediante descenso de gradiente, Ray puede implementar estrategias de paralelismo de datos donde:
                                   - Se particiona el conjunto de datos en $$B$$ lotes: $$\{(X_1,y_1), (X_2,y_2), ..., (X_B,y_B)\}$$
                                   - Cada trabajador $$i$$ calcula el gradiente local: $$g_i = \nabla_\theta \mathcal{L}(f_\theta(X_i), y_i)$$
                                   - El nodo principal agrega los gradientes: $$g = \frac{1}{B}\sum_{i=1}^{B} g_i$$
                                   - Actualiza los parámetros: $$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot g$$
                                   
                                   La convergencia del entrenamiento paralelo está determinada por:
                                   $$\mathbb{E}[\|\theta^{(T)} - \theta^*\|^2] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}} + \frac{\sigma^2}{B\sqrt{T}}\right)$$
                                   
                                   donde $$\sigma^2$$ es la varianza del gradiente estocástico y $$B$$ es el número de trabajadores.
                                
                                6. $$\textbf{Evaluación distribuida y agregación de resultados:}$$ La evaluación paralela de los modelos $$\{f_{\theta_1}, f_{\theta_2}, ..., f_{\theta_m}\}$$ sigue un modelo Map-Reduce:
                                   
                                   - **Map**: Para cada modelo $$f_{\theta_i}$$ y conjunto de prueba $$(X_{\text{test}}, y_{\text{test}})$$:
                                     $$\text{pred}_i = f_{\theta_i}(X_{\text{test}})$$
                                     $$\text{metrics}_i = \text{evaluate}(\text{pred}_i, y_{\text{test}})$$
                                   
                                   - **Reduce**: Agregación de métricas en un diccionario indexado:
                                     $$\text{results} = \bigoplus_{i=1}^{m} \{\text{model}_i: \text{metrics}_i\}$$
                                     
                                   - **Inferencia estadística**: Cálculo de intervalos de confianza para cada métrica:
                                     $$\text{CI}_{1-\alpha}(\text{metric}) = \bar{\text{metric}} \pm t_{1-\alpha/2, m-1} \cdot \frac{s}{\sqrt{m}}$$
                                     
                                     donde $$s$$ es la desviación estándar de la métrica entre los modelos.
                                """)
                            
                            with col2:
                                st.markdown(r"""
                                $$\textbf{Arquitectura Distribuida de Ray: Sistema Formal de Cómputo}$$
                                
                                ```
                                ┌────────────────────────────────────────────────────────────────────┐
                                │                        Ray Head Node (Coordinator)                 │
                                │  ┌────────────────────────┐      ┌────────────────────────────┐    │
                                │  │  Dynamic Task Scheduler│◄────►│ Global Control Store (GCS) │    │
                                │  │  ┌──────────────────┐  │      │ ┌──────────────────────┐   │    │
                                │  │  │Policy: π(s)→a    │  │      │ │Node Registry: V→{IP} │   │    │
                                │  │  │Value: Q(s,a)     │  │      │ │Object Directory      │   │    │
                                │  │  │Resource Matrix Ω │  │      │ │Actor Directory       │   │    │
                                │  │  └──────────────────┘  │      │ └──────────────────────┘   │    │
                                │  └────────────────────────┘      └────────────────────────────┘    │
                                │  ┌─────────────────────────────────────────────────────────────┐   │
                                │  │              Distributed Object Store (DOS)                 │   │
                                │  │  ┌────────────────┐ ┌─────────────────┐ ┌────────────────┐  │   │
                                │  │  │ Memory Manager │ │ Storage Backend │ │ Object Transfer│  │   │
                                │  │  │   Φ: O→{0,1}ᵈ  │ │   S:Φ(O)→bytes  │ │   Network API  │  │   │
                                │  │  └────────────────┘ └─────────────────┘ └────────────────┘  │   │
                                │  └─────────────────────────────────────────────────────────────┘   │
                                └───────────────────────────────┬────────────────────────────────────┘
                                                                │
                                            ┌─────────────────────────────────────┐
                                            │     Ray Transport Layer (gRPC)      │
                                            │  Latency Matrix L = [lᵢⱼ]_{n×n}     │
                                            └──┬────────────────────┬─────────────┘
                                               │                     │
                                               │                     │
                                ┌──────────────▼──────────────┐    ┌─▼────────────────────────┐
                                │       Worker Node 1         │    │      Worker Node 2       │
                                │ ┌─────────────────────────┐ │    │ ┌───────────────────────┐│
                                │ │   Execution Runtime     │ │    │ │  Execution Runtime    ││
                                │ │ ┌───────────────────┐   │ │    │ │┌───────────────────┐  ││
                                │ │ │Task Queue:(Qᵢ,πᵢ) │   │ │    │ ││Task Queue: (Qᵢ,πᵢ)│  ││
                                │ │ │Resource Monitor   │   │ │    │ ││Resource Monitor   │  ││
                                │ │ │Execution Threads  │   │ │◄──►│ ││Execution Threads  │  ││
                                │ │ └───────────────────┘   │ │    │ │└───────────────────┘  ││
                                │ └─────────────────────────┘ │    │ └───────────────────────┘│
                                │ ┌─────────────────────────┐ │    │ ┌───────────────────────┐│
                                │ │    Local Object Store   │ │    │ │   Local Object Store  ││
                                │ │ ┌───────────┐ ┌───────┐ │ │    │ │┌───────────┐┌───────┐ ││
                                │ │ │Shared Mem │ │Cache: │ │ │◄──►│ ││Shared Mem ││Cache: │ ││
                                │ │ │X∈ℝᵐˣᵖ,y∈ℝᵐ│ │ LRU   │ │ │    │ ││X∈ℝᵐˣᵖ,y∈ℝᵐ││ LRU   │ ││
                                │ │ └───────────┘ └───────┘ │ │    │ │└───────────┘└───────┘ ││
                                │ └─────────────────────────┘ │    │ └───────────────────────┘│
                                │ ┌─────────────────────────┐ │    │ ┌───────────────────────┐│
                                │ │  Model Training Workers │ │    │ │ Model Training Workers││
                                │ │ ┌───────────────────┐   │ │    │ │┌───────────────────┐  ││
                                │ │ │f_θ₁:X→ŷ₁ (model 1)│   │ │    │ ││f_θ₂:X→ŷ₂ (model 2)│  ││
                                │ │ │▫ Compute Gradient │   │ │    │ ││▫ Compute Gradient │  ││
                                │ │ │▫ Parameters: θ₁   │   │ │    │ ││▫ Parameters: θ₂   │  ││
                                │ │ │▫ Hyperparams: λ₁  │   │ │    │ ││▫ Hyperparams: λ₂  │  ││
                                │ │ └───────────────────┘   │ │    │ │└───────────────────┘  ││
                                │ └─────────────────────────┘ │    │ └───────────────────────┘│
                                └─────────────────────────────┘    └──────────────────────────┘
                                ```
                                
                                $$\textbf{Flujo de Información y Formalización Matemática:}$$
                                
                                1. $$\textbf{Inicialización del sistema:}$$ Se establece un espacio de memoria global compartida con direccionamiento hash:
                                   $$\Phi: \mathcal{O} \mapsto \{0,1\}^d \quad \text{(Función de hash)}$$
                                   $$\mathcal{A}: \Phi(\mathcal{O}) \times \{1,2,...,n\} \mapsto \{0,1\} \quad \text{(Función de ubicación)}$$
                                
                                2. $$\textbf{Distribución y acceso a datos:}$$ Los tensores de entrenamiento se descomponen en bloques y se distribuyen:
                                   $$X \in \mathbb{R}^{m \times p} \Rightarrow \{X^{(1)}, X^{(2)}, ..., X^{(k)}\}, X^{(i)} \in \mathbb{R}^{\frac{m}{k} \times p}$$
                                   $$\text{ref}_X^{(i)} = \Phi(X^{(i)}) \quad \text{para } i \in \{1,2,...,k\}$$
                                
                                3. $$\textbf{Scheduling de tareas distribuidas:}$$ Cada tarea de entrenamiento se define como un objeto:
                                   $$\tau_i = \{\text{función}: f_i, \text{argumentos}: [\text{ref}_X, \text{ref}_y, \theta_i], \text{recursos}: r_i, \text{prioridad}: p_i\}$$
                                   
                                   El scheduler resuelve un problema de asignación óptima basado en la teoría de colas:
                                   $$\pi^*: \mathcal{S} \rightarrow \mathcal{A}, \pi^*(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$
                                
                                4. $$\textbf{Tolerancia a fallos:}$$ El sistema implementa un mecanismo distribuido de detección y recuperación:
                                   $$P(\text{fallo} | \text{tiempo} = t) = 1 - e^{-\lambda t} \quad \text{(Modelo de fallos Poisson)}$$
                                   $$\text{MTTR} = \mathbb{E}[\text{Tiempo de recuperación}] = \int_0^{\infty} t \cdot f_{\text{recuperación}}(t) dt$$
                                   
                                   donde $$\lambda$$ es la tasa de fallos y $$f_{\text{recuperación}}(t)$$ es la función de densidad del tiempo de recuperación.
                                """)
                            
                            # Mostrar diagrama visual mejorado del proceso
                            st.markdown(r"""
                            $$\textbf{Análisis Comparativo del Rendimiento: Teoría Formal de Sistemas Distribuidos}$$
                            
                            $$\textbf{Modelado formal del entrenamiento secuencial:}$$
                            
                            En un sistema secuencial, la complejidad temporal total viene dada por:
                            
                            $$T_{\text{secuencial}} = \sum_{i=1}^{m} t_i + \sum_{i=1}^{m-1} \delta_i$$
                            
                            Donde:
                            - $$t_i$$ es el tiempo de entrenamiento del modelo $$i$$
                            - $$\delta_i$$ es el tiempo de transición entre el entrenamiento de los modelos $$i$$ e $$i+1$$
                            
                            Si modelamos $$t_i$$ como una variable aleatoria con distribución $$t_i \sim \mathcal{N}(\mu_t, \sigma_t^2)$$, entonces:
                            
                            $$\mathbb{E}[T_{\text{secuencial}}] = m\mu_t + (m-1)\mu_\delta$$
                            $$\text{Var}[T_{\text{secuencial}}] = m\sigma_t^2 + (m-1)\sigma_\delta^2$$
                            
                            ```
                            Tiempo ─────────────────────────────────────────────────────────────▶
                                  │
                            Modelo 1 ━━━━━━━━━━━━━━━━━━━━▶
                                                          δ₁
                            Modelo 2                     ━━━━━━━━━━━━━━━━━━━━▶
                                                                               δ₂
                            Modelo 3                                         ━━━━━━━━━━━━━━━━━━━━▶
                                                                                                  │
                            Tiempo total: ───────────────────────────────────────────────────────▶
                            ```
                            
                            $$\textbf{Modelado formal del entrenamiento distribuido:}$$
                            
                            En un sistema con $$p$$ trabajadores, el tiempo total puede modelarse como:
                            
                            $$T_{\text{paralelo}} = \max_{j \in \{1,2,...,p\}} \left\{ \sum_{i \in \mathcal{A}_j} (t_i + \tau_i^{\text{overhead}}) \right\}$$
                            
                            Donde:
                            - $$\mathcal{A}_j$$ es el conjunto de modelos asignados al trabajador $$j$$
                            - $$\tau_i^{\text{overhead}}$$ es el tiempo adicional por comunicación y sincronización
                            
                            Bajo asignación óptima de tareas y distribución uniforme, tenemos:
                            
                            $$T_{\text{paralelo}} \approx \frac{m\mu_t}{p} + \tau_{\text{comunicación}} + \tau_{\text{sincronización}} + \max_{i \in \{1,2,...,m\}} \{t_i - \mu_t\}$$
                            
                            La desviación de este tiempo sigue:
                            
                            $$\sigma_{T_{\text{paralelo}}} \approx \sqrt{\frac{m\sigma_t^2}{p^2} + \sigma_{\text{comm}}^2 + \sigma_{\text{sync}}^2}$$
                            
                            ```
                            Tiempo ─────────────────────▶
                                  │         │
                            Worker 1 ━━━━━━━┿━━━━━━━━━━━▶  ◀── τₛᵧₙ (sincronización)
                                            │
                            Worker 2 ━━━━━━━┿━━━━━━━━━━━▶ ◀── Load Imbalance
                                            │     τₖₘᵤₙ
                            Worker 3 ━━━━━━━▶      ◀──▶
                                            │           │
                            Tiempo total: ─────────────▶
                            ```
                            
                            $$\textbf{Formalización de las métricas de eficiencia:}$$
                            
                            1. $$\textbf{Aceleración observada (Speedup)}:$$ Relación entre tiempo secuencial y paralelo:
                               $$S(m,p) = \frac{T_{\text{secuencial}}}{T_{\text{paralelo}}} = \frac{\sum_{i=1}^{m} t_i + \sum_{i=1}^{m-1} \delta_i}{\max_{j \in \{1,2,...,p\}} \left\{ \sum_{i \in \mathcal{A}_j} (t_i + \tau_i^{\text{overhead}}) \right\}}$$
                            
                            2. $$\textbf{Eficiencia computacional:}$$ Utilización efectiva de los recursos:
                               $$E(m,p) = \frac{S(m,p)}{p} = \frac{T_{\text{secuencial}}}{p \cdot T_{\text{paralelo}}}$$
                            
                            3. $$\textbf{Ley de Amdahl generalizada:}$$ Límite teórico del speedup considerando la parte paralelizable:
                               $$S_{\text{max}}(p, \alpha) = \frac{1}{(1-\alpha) + \frac{\alpha}{p} + \kappa(p)} \leq \frac{1}{1-\alpha}$$
                               
                               Donde:
                               - $$\alpha$$ es la fracción paralelizable del algoritmo
                               - $$\kappa(p)$$ representa el overhead de comunicación que aumenta con $$p$$
                            
                            4. $$\textbf{Ley de Gustafson-Barsis:}$$ Speedup para problemas escalables:
                               $$S_{\text{Gustafson}}(p, \alpha) = p - \alpha(p-1)$$
                            
                            5. $$\textbf{Utilización de recursos:}$$ Eficiencia en el uso de recursos computacionales:
                               $$U(p) = \frac{1}{T_{\text{paralelo}} \cdot p}\sum_{j=1}^{p}\sum_{i \in \mathcal{A}_j} t_i$$
                            
                            6. $$\textbf{Escalabilidad isoefiencia:}$$ Relación entre aumento de la carga y recursos para mantener eficiencia constante:
                               $$\text{Si } E(m,p) = c \text{ constante, entonces } m = \Theta(p \cdot f(p))$$
                               
                               donde $$f(p)$$ es la función de sobrecarga que depende de la arquitectura y algoritmo.
                            
                            $$\textbf{Análisis cuantitativo del overhead de comunicación:}$$
                            
                            El overhead total de comunicación $$\tau_{\text{comm}}$$ en Ray puede modelarse como:
                            
                            $$\tau_{\text{comm}} = \tau_{\text{serialización}} + \tau_{\text{transferencia}} + \tau_{\text{deserialización}}$$
                            
                            Para datos de dimensión $$d$$ y tamaño $$s$$, tenemos:
                            
                            $$\tau_{\text{serialización}} = \alpha_{\text{ser}} \cdot s \cdot \log(d)$$
                            $$\tau_{\text{transferencia}} = \frac{s}{B} + L$$
                            $$\tau_{\text{deserialización}} = \alpha_{\text{deser}} \cdot s \cdot \log(d)$$
                            
                            Donde $$B$$ es el ancho de banda de la red, $$L$$ es la latencia, y $$\alpha_{\text{ser}}$$, $$\alpha_{\text{deser}}$$ son constantes de proporcionalidad.
                            
                            $$\textbf{Beneficios adicionales del entrenamiento distribuido:}$$
                            
                            1. $$\textbf{Utilización Eficiente de Recursos:}$$ 
                               - $$\text{Utilización} = \frac{\text{Tiempo efectivo de computación}}{\text{Tiempo total} \times \text{Recursos totales}} \times 100\%$$
                               - Formalizado: $$\eta = \frac{\sum_{j=1}^{p} \sum_{t=0}^{T} \text{uso}_j(t)}{p \times T} \times 100\%$$
                               
                            2. $$\textbf{Exploración del Espacio de Hiperparámetros:}$$
                               - Capacidad de exploración: $$\mathcal{C} = \mathcal{O}(\min(m,p))$$ veces más modelos en el mismo tiempo
                               - Búsqueda en grid de hiperparámetros: $$|\Theta| = \prod_{i=1}^{d} |\theta_i|$$ configuraciones posibles
                           
                            3. $$\textbf{Escalabilidad con Grandes Volúmenes de Datos:}$$
                               - Para datasets $$X \in \mathbb{R}^{n \times d}$$ donde $$n \gg 0$$ o $$d \gg 0$$
                               - Complejidad espacial distribuida: $$\mathcal{O}\left(\frac{n \times d}{p} + \text{overhead}\right)$$ por nodo
                               - Reducción de memoria por nodo: $$\Delta \text{Mem} = \text{Mem}_{\text{secuencial}} \times \left(1 - \frac{1}{p}\right)$$
                               
                            4. $$\textbf{Tolerancia a Fallos:}$$
                               - Probabilidad de finalización exitosa: $$P(\text{éxito}) = 1 - \prod_{i=1}^{p}(1-r_i)$$
                               - Donde $$r_i$$ es la fiabilidad del nodo $$i$$
                               
                            5. $$\textbf{Monitorización Avanzada:}$$
                               - Métricas temporales: $$\{t_{\text{inicio}}^i, t_{\text{fin}}^i, t_{\text{total}}^i\}$$ para cada modelo $$i$$
                               - Métricas de rendimiento: $$\{\text{precisión}_i, \text{recall}_i, \text{f1-score}_i, \text{auc}_i\}$$""")
                    else:
                        st.info("Ray no está activado. El entrenamiento se realizará en modo secuencial.")
                    
                    progress_placeholder = st.empty()
                    progress_placeholder.text("Preparando datos...")
                    
                    # Entrenar el modelo con mensajes de progreso
                    start_time = time.time()
                    
                    # Crear una interfaz de progreso más visual
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Crear placeholders para cada modelo
                    model_statuses = {}
                    if train_multiple and len(models_config) > 1:
                        st.subheader("Estado de entrenamiento de modelos")
                        cols = st.columns(min(len(models_config), 3))  # Máximo 3 columnas
                        for i, model_config in enumerate(models_config):
                            col_idx = i % min(len(models_config), 3)
                            with cols[col_idx]:
                                model_name = model_config['name']
                                model_statuses[model_name] = {
                                    'container': st.container(),
                                    'status': 'Pendiente',
                                    'start_time': None,
                                    'end_time': None
                                }
                                model_statuses[model_name]['container'].markdown(
                                    f"<div style='padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>"
                                    f"<h5>{model_name}</h5>"
                                    f"<p>Estado: ⏳ Pendiente</p>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                        temp_csv_path = tmp_file.name
                        # Guardar los datos en un CSV temporal
                        combined_df = X.copy()
                        combined_df[target_column] = y
                        combined_df.to_csv(temp_csv_path, index=False)
                        progress_placeholder.text("Datos preparados. Iniciando entrenamiento...")
                    
                    try:
                        # Mostrar progreso del entrenamiento
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Paso 1: Cargar y preprocesar datos
                        progress_placeholder.text("Cargando y preprocesando datos...")
                        status_text.text("Paso 1/4: Cargando y distribuyendo datos...")
                        progress_bar.progress(10)
                        
                        # Actualizar estado de los modelos
                        if train_multiple and len(models_config) > 1:
                            for model_name in model_statuses:
                                model_statuses[model_name]['status'] = 'Iniciando'
                                model_statuses[model_name]['start_time'] = time.time()
                                model_statuses[model_name]['container'].markdown(
                                    f"<div style='padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>"
                                    f"<h5>{model_name}</h5>"
                                    f"<p>Estado: 🔄 Iniciando entrenamiento...</p>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                        
                        # Ejecutar el pipeline utilizando el método run
                        model_ids = pipeline.run(
                            dataset_path=temp_csv_path,
                            target_column=target_column,
                            models_config=models_config,
                            test_size=test_size,
                            random_state=random_state,
                            evaluation_metrics=["accuracy", "precision", "recall", "f1"],
                            task_type="classification",
                            experiment_name=f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
                        )
                        
                        # Actualizar el progreso y el estado de los modelos
                        progress_bar.progress(70)
                        status_text.text("Procesando resultados...")
                        
                        # Actualizar tarjetas de modelos con resultados
                        if train_multiple and len(models_config) > 1:
                            for model_name in model_statuses:
                                if model_name in model_ids:
                                    model_id = model_ids[model_name]
                                    metrics = model_registry.get_metadata(model_id).get('metrics', {})
                                    model_score = metrics.get('accuracy', 0) * 100
                                    model_statuses[model_name]['status'] = 'Completado'
                                    model_statuses[model_name]['end_time'] = time.time()
                                    duration = model_statuses[model_name]['end_time'] - model_statuses[model_name]['start_time']
                                    
                                    model_statuses[model_name]['container'].markdown(
                                        f"<div style='padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>"
                                        f"<h5>{model_name}</h5>"
                                        f"<p>Estado: ✅ Completado</p>"
                                        f"<p>Precisión: {model_score:.2f}%</p>"
                                        f"<p>Tiempo: {duration:.2f} segundos</p>"
                                        f"</div>", 
                                        unsafe_allow_html=True
                                    )
                                else:
                                    model_statuses[model_name]['container'].markdown(
                                        f"<div style='padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>"
                                        f"<h5>{model_name}</h5>"
                                        f"<p>Estado: ❌ Error en entrenamiento</p>"
                                        f"</div>", 
                                        unsafe_allow_html=True
                                    )
                        
                        # Actualizar progreso después de completar el entrenamiento
                        progress_bar.progress(100)
                        status_text.text("Entrenamiento completado.")
                        progress_placeholder.text("¡Entrenamiento completado!")
                        
                        end_time = time.time()
                        
                        # Mostrar resultados para todos los modelos entrenados
                        st.success(f"¡Se han entrenado {len(model_ids)} modelos correctamente!")
                        
                        # Tabla de modelos y métricas
                        models_data = []
                        for model_name, model_id in model_ids.items():
                            metrics = model_registry.get_metadata(model_id).get('metrics', {})
                            models_data.append({
                                "Modelo": model_name,
                                "ID": model_id,
                                "Precisión": metrics.get("accuracy", 0),
                                "F1-Score": metrics.get("f1", 0),
                                "Tiempo (s)": end_time - start_time
                            })
                        
                        models_results_df = pd.DataFrame(models_data)
                        st.dataframe(models_results_df, use_container_width=True)
                        
                        # Visualizar gráficas de rendimiento si hay modelos entrenados
                        if len(models_data) > 0:
                            st.subheader("Visualización de métricas")
                            
                            # Gráfica comparativa de modelos
                            metrics_fig = px.bar(
                                models_results_df, 
                                x='Modelo', 
                                y=['Precisión', 'F1-Score'],
                                barmode='group',
                                title="Comparación de rendimiento de modelos"
                            )
                            metrics_fig.update_layout(yaxis=dict(range=[0, 1]))
                            st.plotly_chart(metrics_fig, use_container_width=True)
                            
                            # Si hay más de un modelo, mostrar comparativa de tiempos y detalles del entrenamiento distribuido
                            if len(models_data) > 1:
                                st.subheader("Análisis del entrenamiento distribuido")
                                
                                # Obtener los tiempos reales de entrenamiento de cada modelo desde el pipeline
                                training_times = pipeline.performance_metrics["training_times"]
                                if training_times:
                                    # Crear un dataframe para los tiempos de entrenamiento específicos
                                    times_df = pd.DataFrame([
                                        {"Modelo": model_name, "Tiempo (s)": time_value} 
                                        for model_name, time_value in training_times.items()
                                    ])
                                    
                                    # Gráfica de tiempos de entrenamiento
                                    time_fig = px.bar(
                                        times_df,
                                        x='Modelo',
                                        y='Tiempo (s)',
                                        title="Tiempo de entrenamiento por modelo (entrenados en paralelo)"
                                    )
                                    st.plotly_chart(time_fig, use_container_width=True)
                                    
                                    # Mostrar la eficiencia del entrenamiento paralelo
                                    total_sequential_time = sum(training_times.values())
                                    actual_parallel_time = end_time - start_time
                                    speedup = total_sequential_time / actual_parallel_time if actual_parallel_time > 0 else 0
                                    efficiency = speedup / min(len(models_config), num_workers) * 100 if num_workers > 0 else 0
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Tiempo secuencial total", f"{total_sequential_time:.2f} s")
                                    with col2:
                                        st.metric("Tiempo paralelo real", f"{actual_parallel_time:.2f} s")
                                    with col3:
                                        st.metric("Aceleración (speedup)", f"{speedup:.2f}x")
                                    
                                    # Mostrar fórmulas y eficiencia con LaTeX
                                    st.markdown(r"""
                                    $$\textbf{Análisis Matemático del Rendimiento Obtenido:}$$
                                    
                                    $$\text{Modelo teórico de aceleración (Ley de Amdahl modificada):}$$
                                    
                                    $$S(m,p) = \frac{T_{\text{secuencial}}}{T_{\text{paralelo}}} = \frac{\sum_{i=1}^{m} t_i}{\max\{\text{overhead} + \max_{i \in \text{batch}_j} t_i\}_{j=1}^{\lceil m/p \rceil}}$$
                                    
                                    $$\text{En nuestro caso:}$$
                                    
                                    $$\text{Speedup}_{\text{medido}} = \frac{T_{\text{secuencial}}}{T_{\text{paralelo}}} = \frac{%s}{%s} = %s$$
                                    
                                    $$\text{Eficiencia}_{\text{medida}} = \frac{\text{Speedup}_{\text{medido}}}{\min(n_{\text{modelos}}, n_{\text{workers}})} \times 100\%% = \frac{%s}{\min(%s, %s)} \times 100\%% = %s\%%$$
                                    
                                    $$\text{Tiempo teórico óptimo (cota inferior)} = \frac{T_{\text{secuencial}}}{\min(n_{\text{modelos}}, n_{\text{workers}})} = \frac{%s}{\min(%s, %s)} = %s \text{ s}$$
                                    
                                    $$\text{Overhead de comunicación estimado} = T_{\text{paralelo}} - \frac{T_{\text{secuencial}}}{\min(n_{\text{modelos}}, n_{\text{workers}})} = %s - %s = %s \text{ s}$$
                                    """ % (
                                        total_sequential_time, 
                                        actual_parallel_time, 
                                        f"{speedup:.2f}",
                                        f"{speedup:.2f}",
                                        len(models_config),
                                        num_workers,
                                        f"{efficiency:.2f}",
                                        total_sequential_time,
                                        len(models_config),
                                        num_workers,
                                        f"{total_sequential_time/min(len(models_config), num_workers):.2f}",
                                        actual_parallel_time,
                                        f"{total_sequential_time/min(len(models_config), num_workers):.2f}",
                                        f"{actual_parallel_time - total_sequential_time/min(len(models_config), num_workers):.2f}"
                                    ))
                                        
                                    st.info(f"El entrenamiento distribuido con Ray ha permitido una aceleración de {speedup:.2f}x comparado con el entrenamiento secuencial.")
                                else:
                                    time_fig = px.bar(
                                        models_results_df,
                                        x='Modelo',
                                        y='Tiempo (s)',
                                        title="Tiempo total de entrenamiento"
                                    )
                                    st.plotly_chart(time_fig, use_container_width=True)
                            
                            # Seleccionar un modelo para ver detalles
                            selected_model_idx = st.selectbox(
                                "Seleccione un modelo para ver detalles:",
                                range(len(models_data)),
                                format_func=lambda i: models_data[i]["Modelo"]
                            )
                            
                            if selected_model_idx is not None:
                                selected_data = models_data[selected_model_idx]
                                selected_model_id = selected_data["ID"]
                                selected_metrics = model_registry.get_metadata(selected_model_id).get('metrics', {})
                    
                                # Mostrar métricas detalladas del modelo seleccionado
                                st.subheader(f"Métricas detalladas: {selected_data['Modelo']}")
                                
                                # Mostrar fórmulas matemáticas para cada métrica
                                st.markdown(r"""
                                $$\textbf{Fórmulas de las métricas de evaluación:}$$
                                
                                $$\text{Exactitud (Accuracy)} = \frac{VP + VN}{VP + FP + VN + FN}$$
                                
                                $$\text{Precisión (Precision)} = \frac{VP}{VP + FP}$$
                                
                                $$\text{Exhaustividad (Recall)} = \frac{VP}{VP + FN}$$
                                
                                $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
                                
                                Donde:
                                - $$VP$$ = Verdaderos Positivos
                                - $$VN$$ = Verdaderos Negativos
                                - $$FP$$ = Falsos Positivos
                                - $$FN$$ = Falsos Negativos
                                """)
                                
                                # Tabla de métricas
                                metrics_df = pd.DataFrame(selected_metrics.items(), columns=["Métrica", "Valor"])
                                st.dataframe(metrics_df, use_container_width=True)
                                
                                # Gráfica de barras de métricas
                                fig = px.bar(metrics_df, x='Métrica', y='Valor', text='Valor')
                                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                                fig.update_layout(
                                    title=f"Métricas de rendimiento: {selected_data['Modelo']}",
                                    xaxis_title="Métrica",
                                    yaxis_title="Valor",
                                    yaxis=dict(range=[0, 1])
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Añadir explicación matemática del modelo seleccionado
                                with st.expander("Explicación matemática del modelo"):
                                    model_type_name = selected_data['Modelo'].split('_')[0] if '_' in selected_data['Modelo'] else selected_data['Modelo']
                                    
                                    if "RandomForest" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Random Forest - Teoría Estadística y Computacional Avanzada}$$
                                        
                                        El algoritmo Random Forest es un método ensemble basado en el principio de reducción de varianza mediante bootstrap aggregating (bagging) y la decorrelación de árboles individuales a través de selección aleatoria de características.
                                        
                                        $$\textbf{Formulación matemática del modelo:}$$
                                        
                                        Un Random Forest $$f_{RF}$$ está compuesto por $$B$$ árboles de decisión $$\{T_1, T_2, ..., T_B\}$$, cada uno entrenado bajo dos fuentes de aleatoriedad:
                                        
                                        1. Remuestreo bootstrap del conjunto de entrenamiento $$\mathcal{D} = \{(X_1, y_1), (X_2, y_2), ..., (X_n, y_n)\}$$ para generar $$\mathcal{D}_b$$.
                                        
                                        2. En cada nodo, selección aleatoria de un subconjunto de $$m_{try} < p$$ características para la división.
                                        
                                        Para problemas de clasificación, la predicción se obtiene mediante voto mayoritario:
                                        
                                        $$\hat{f}_{RF}(x) = \underset{y \in \mathcal{Y}}{\operatorname{argmax}} \sum_{b=1}^{B} \mathbb{I}(\hat{f}_b(x) = y)$$
                                        
                                        Para problemas de regresión, se utiliza el promedio:
                                        
                                        $$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$
                                        
                                        $$\textbf{Construcción de árboles individuales:}$$
                                        
                                        Cada árbol $$T_b$$ se construye recursivamente minimizando una función de impureza $$I(t)$$ en cada nodo $$t$$:
                                        
                                        $$\Delta I(s, t) = I(t) - p_L \cdot I(t_L) - p_R \cdot I(t_R)$$
                                        
                                        Donde:
                                        - $$s$$ es la división candidata
                                        - $$t_L$$ y $$t_R$$ son los nodos hijos izquierdo y derecho
                                        - $$p_L$$ y $$p_R$$ son las proporciones de muestras que van a cada hijo
                                        
                                        Para clasificación, se utilizan medidas como:
                                        
                                        - Índice Gini: $$I_{Gini}(t) = 1 - \sum_{j=1}^{K} p^2_{j,t}$$
                                        - Entropía: $$I_{Entropy}(t) = -\sum_{j=1}^{K} p_{j,t} \log(p_{j,t})$$
                                        
                                        Para regresión:
                                        
                                        - Varianza: $$I_{Var}(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$$
                                        
                                        $$\textbf{Propiedades estadísticas:}$$
                                        
                                        El error de generalización se puede descomponer como:
                                        
                                        $$Err(x) = Var(\hat{f}(x)) + [Bias(\hat{f}(x))]^2 + \sigma^2_{\epsilon}$$
                                        
                                        El bagging y la selección aleatoria de características reducen la varianza a costa de un ligero aumento en el sesgo. Para $$B$$ árboles correlacionados con factor $$\rho$$:
                                        
                                        $$Var(\hat{f}_{RF}(x)) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$
                                        
                                        $$\textbf{Importancia de características:}$$
                                        
                                        La importancia de una característica $$X_j$$ se calcula como la disminución promedio en la impureza ponderada por la probabilidad de alcanzar el nodo:
                                        
                                        $$Imp(X_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b: v(t)=j} p(t) \Delta I(j, t)$$
                                        
                                        $$\textbf{Muestras Out-of-Bag (OOB):}$$
                                        
                                        Aproximadamente el 36.8% de las muestras no son seleccionadas en cada bootstrap, formando el conjunto OOB que sirve como validación interna:
                                        
                                        $$Err_{OOB} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{f}_{OOB}(x_i))$$
                                        
                                        $$\textbf{Entrenamiento distribuido:}$$
                                        
                                        En entornos paralelos, la construcción de árboles se distribuye naturalmente:
                                        
                                        $$T_b = \mathcal{A}_b(\mathcal{D}_b) \quad \text{para } b=1,2,...,B \text{ en paralelo}$$
                                        
                                        Donde $$\mathcal{A}_b$$ es el algoritmo de construcción del árbol $$b$$ aplicado al conjunto bootstrap $$\mathcal{D}_b$$.
                                        
                                        La complejidad computacional total es $$\mathcal{O}(B \cdot n \log n \cdot m_{try})$$, reducible a $$\mathcal{O}(n \log n \cdot m_{try})$$ con $$B$$ procesadores.
                                        """)
                                    
                                    elif "Logistic" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Regresión Logística - Teoría de Modelos Lineales Generalizados}$$
                                        
                                        La regresión logística pertenece a la familia de Modelos Lineales Generalizados (GLM) y modela la probabilidad condicional de una variable binaria mediante la función logística.
                                        
                                        $$\textbf{Modelo probabilístico:}$$
                                        
                                        Para una variable respuesta binaria $$Y \in \{0, 1\}$$ y un vector de características $$X \in \mathbb{R}^p$$, el modelo logístico define:
                                        
                                        $$P(Y=1|X=x) = \sigma(\beta^T x) = \frac{1}{1 + e^{-\beta^T x}}$$
                                        
                                        Donde:
                                        - $$\beta = (\beta_0, \beta_1, ..., \beta_p)^T$$ es el vector de coeficientes incluyendo el intercepto
                                        - $$\sigma(z) = \frac{1}{1+e^{-z}}$$ es la función logística (sigmoide)
                                        - $$x$$ se asume aumentado con 1 para el intercepto: $$x = (1, x_1, x_2, ..., x_p)^T$$
                                        
                                        $$\textbf{Interpretación del modelo:}$$
                                        
                                        La regresión logística modela el logaritmo de odds (log-odds o logit):
                                        
                                        $$\log\left(\frac{P(Y=1|X=x)}{P(Y=0|X=x)}\right) = \log\left(\frac{P(Y=1|X=x)}{1-P(Y=1|X=x)}\right) = \beta^T x$$
                                        
                                        El coeficiente $$\beta_j$$ representa el cambio en log-odds asociado a un incremento unitario en $$X_j$$, manteniéndose constantes las demás variables:
                                        
                                        $$\Delta \text{logit} = \beta_j \cdot \Delta X_j$$
                                        
                                        En términos de odds ratio: $$OR = e^{\beta_j}$$
                                        
                                        $$\textbf{Estimación por máxima verosimilitud (MLE):}$$
                                        
                                        Para un conjunto de datos $$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$$, la función de verosimilitud es:
                                        
                                        $$L(\beta) = \prod_{i=1}^{n} P(Y=y_i|X=x_i) = \prod_{i=1}^{n} \sigma(\beta^T x_i)^{y_i} \cdot (1-\sigma(\beta^T x_i))^{1-y_i}$$
                                        
                                        La log-verosimilitud negativa (función de pérdida) a minimizar es:
                                        
                                        $$\ell(\beta) = -\sum_{i=1}^{n} \left[ y_i \log \sigma(\beta^T x_i) + (1-y_i) \log (1-\sigma(\beta^T x_i)) \right]$$
                                        
                                        $$\textbf{Optimización y regularización:}$$
                                        
                                        El problema de optimización con regularización L2 (Ridge) es:
                                        
                                        $$\min_{\beta} \ell(\beta) + \frac{\lambda}{2}\|\beta\|_2^2$$
                                        
                                        Con regularización L1 (Lasso):
                                        
                                        $$\min_{\beta} \ell(\beta) + \lambda\|\beta\|_1$$
                                        
                                        Con regularización ElasticNet (combinación de L1 y L2):
                                        
                                        $$\min_{\beta} \ell(\beta) + \lambda[(1-\alpha)\|\beta\|_2^2/2 + \alpha\|\beta\|_1]$$
                                        
                                        Donde $$\lambda = 1/C$$ controla la fuerza de la regularización y $$\alpha$$ el balance entre L1 y L2.
                                        
                                        $$\textbf{Algoritmos de optimización:}$$
                                        
                                        - LBFGS: Aproximación de memoria limitada del algoritmo BFGS de segundo orden
                                        - Newton-CG: Método de Newton con gradientes conjugados para subespacios grandes
                                        - SAGA: Variante estocástica de promediación de gradientes con paso adaptativo
                                        - Liblinear: Implementación eficiente para problemas lineales de gran escala
                                        
                                        $$\textbf{Propiedades estadísticas:}$$
                                        
                                        - $$\beta_{MLE}$$ es asintóticamente normal: $$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} \mathcal{N}(0, I^{-1}(\beta))$$
                                        - La matriz de información de Fisher es: $$I(\beta) = X^T D X$$ donde $$D = \text{diag}[p_i(1-p_i)]$$
                                        
                                        $$\textbf{Paralelización:}$$
                                        
                                        En entornos distribuidos, se utilizan técnicas como:
                                        
                                        - Actualización asíncrona de parámetros: $$\beta^{(t+1)} = \beta^{(t)} - \eta \nabla \ell_{\text{batch}}(\beta^{(t)})$$
                                        - Descenso de gradiente estocástico paralelo (Hogwild!)
                                        - Promediación de modelos entrenados en subconjuntos de datos: $$\bar{\beta} = \frac{1}{k}\sum_{j=1}^{k} \hat{\beta}_j$$
                                        """)
                                    
                                    elif "Gradient" in model_type_name or "Boosting" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Gradient Boosting Classifier - Fundamentos Matemáticos Avanzados}$$
                                        
                                        Gradient Boosting es un método de ensemble que construye modelos secuencialmente, donde cada nuevo modelo intenta corregir los errores de sus predecesores mediante optimización en la dirección del gradiente negativo de la función de pérdida.
                                        
                                        $$\textbf{Formulación del modelo aditivo:}$$
                                        
                                        Gradient Boosting construye un modelo aditivo en forma incremental:
                                        
                                        $$F_M(x) = \sum_{m=0}^{M} \gamma_m h_m(x)$$
                                        
                                        Donde:
                                        - $$F_M(x)$$ es el modelo final después de $$M$$ iteraciones
                                        - $$h_m(x)$$ son los modelos base (típicamente árboles de decisión)
                                        - $$\gamma_m$$ son los coeficientes de expansión (pesos de los modelos)
                                        
                                        $$\textbf{Proceso de optimización:}$$
                                        
                                        En cada iteración $$m$$, se busca una función $$h_m$$ que minimice la función de pérdida $$L$$ sobre el conjunto de entrenamiento $$\{(x_i, y_i)\}_{i=1}^n$$:
                                        
                                        $$h_m = \underset{h}{\arg\min} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + h(x_i))$$
                                        
                                        Para hacerlo computacionalmente tratable, se aproxima mediante una expansión de Taylor de primer orden:
                                        
                                        $$h_m \approx \underset{h}{\arg\min} \sum_{i=1}^{n} [-g_i \cdot h(x_i) + \frac{1}{2} h^2(x_i) \cdot h_{ii}]$$
                                        
                                        Donde:
                                        - $$g_i = \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$$ es el gradiente negativo (residuo)
                                        - $$h_{ii} = \left[ \frac{\partial^2 L(y_i, F(x_i))}{\partial F(x_i)^2} \right]_{F=F_{m-1}}$$ es la segunda derivada (usado en Newton boosting)
                                        
                                        $$\textbf{Funciones de pérdida comunes:}$$
                                        
                                        1. Pérdida logarítmica (para clasificación binaria):
                                           $$L(y, F) = \log(1 + e^{-yF}) = -y\log(p) - (1-y)\log(1-p)$$
                                           donde $$p = \frac{1}{1 + e^{-F}}$$ es la probabilidad estimada
                                        
                                        2. Pérdida exponencial (tipo AdaBoost):
                                           $$L(y, F) = e^{-yF}$$
                                        
                                        Para clasificación multiclase, se utiliza la pérdida de entropía cruzada:
                                        $$L(y, F) = -\sum_{k=1}^{K} y_k \log(p_k)$$
                                        donde $$y_k$$ es la indicadora de clase y $$p_k = \frac{e^{F_k}}{\sum_{j=1}^{K} e^{F_j}}$$ (softmax)
                                         3. $$\textbf{Mejoras algorítmicas:}$$

                                        1. $$\textbf{Submuestreo estocástico:}$$ En cada iteración se utiliza solo una fracción aleatoria del conjunto de entrenamiento:
                                           $$h_m = \underset{h}{\arg\min} \sum_{i \in \mathcal{S}_m} L(y_i, F_{m-1}(x_i) + h(x_i))$$
                                           donde $$\mathcal{S}_m \subset \{1,2,...,n\}$$ con $$|\mathcal{S}_m| = \lceil s \cdot n \rceil$$
                                        
                                        2. $$\textbf{Regularización por tasa de aprendizaje:}$$ Se actualiza el modelo con un paso pequeño:
                                           $$F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m h_m(x)$$
                                           donde $$\eta \in (0,1]$$ es la tasa de aprendizaje
                                        
                                        3. $\textbf{Early stopping:}$ Se detiene el algoritmo cuando la pérdida en validación deja de mejorar:
                                        3. $$\textbf{Early stopping:}$$ Se detiene el algoritmo cuando la pérdida en validación deja de mejorar:
                                           $$m^* = \underset{m \in \{1,2,...,M\}}{\arg\min} L_{val}(F_m)$$
                                        
                                        $$\textbf{Detalles de implementación:}$$
                                        Para árboles de regresión como base learners, el algoritmo sigue estos pasos:
                                        
                                        1. Inicializar $$F_0(x) = \underset{\gamma}{\arg\min} \sum_{i=1}^{n} L(y_i, \gamma)$$
                                        
                                        2. Para $$m = 1$$ hasta $$M$$:
                                           - Calcular el gradiente negativo $$\tilde{y}_i = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$$
                                           - Ajustar un árbol de regresión $$h_m(x)$$ a los targets $$\{\tilde{y}_i\}_{i=1}^n$$
                                           - Para cada región $$R_{jm}$$ del árbol, calcular $$\gamma_{jm} = \underset{\gamma}{\arg\min} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$
                                           - Actualizar $$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} \mathbf{1}(x \in R_{jm})$$
                                        
                                        $$\textbf{Análisis teórico:}$$
                                        $$\textbf{Análisis teórico:}$$
                                        
                                        El error de generalización satisface:
                                        
                                        $$\mathbb{E}[L(Y, F_M(X))] \leq \mathbb{E}[L(Y, F_0(X))] \cdot \exp\left(-\frac{\eta^2 M}{2} \cdot \frac{\psi^2}{\beta + \psi^2}\right)$$
                                        
                                        donde $$\psi^2$$ es la constante de fuerte convexidad de $$L$$ y $$\beta$$ es la constante de suavidad.
                                        
                                        $$\textbf{Complejidad computacional:}$$
                                        - Entrenamiento: $$\mathcal{O}(M \cdot n \cdot \log(n) \cdot d)$$ donde $$M$$ es el número de iteraciones, $$n$$ es el número de muestras y $$d$$ es la dimensión
                                        - Entrenamiento: $$\mathcal{O}(M \cdot n \cdot \log(n) \cdot d)$$ donde $$M$$ es el número de iteraciones, $$n$$ es el número de muestras y $$d$$ es la dimensión
                                        - Predicción: $$\mathcal{O}(M \cdot \log(n))$$ para un punto
                                        
                                        $$\textbf{Distribución y paralelización:}$$
                                        Gradient Boosting es inherentemente secuencial, pero permite paralelización mediante:
                                        
                                        1. Paralelización del nivel de árbol: Se construye cada árbol en paralelo en una submuestra de características
                                        1. Paralelización del nivel de árbol: Se construye cada árbol en paralelo en una submuestra de características
                                        2. Paralelización de evaluación de divisiones candidatas: Se evalúan las divisiones en paralelo
                                        3. Boosting distribuido es factible mediante algoritmos como XGBOOST:
                                           $$L^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \Omega(f_t)$$
                                           donde $$\Omega(f_t)$$ es un término de regularización""")
                                    
                                    elif "SVC" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Support Vector Classifier - Teoría Avanzada de Optimización}$$
                                        
                                        Las Máquinas de Vectores de Soporte (SVM) representan un paradigma de aprendizaje basado en la teoría de optimización convexa y la teoría estadística del aprendizaje de Vapnik-Chervonenkis.
                                        
                                        $$\textbf{Formulación del problema primal:}$$
                                        
                                        Para datos linealmente separables, SVM busca el hiperplano óptimo $$\{x: w^Tx + b = 0\}$$ que maximiza el margen geométrico $$\frac{2}{\|w\|}$$ entre clases:
                                        
                                        $$\begin{align}
                                        \min_{w,b} \quad & \frac{1}{2}\|w\|^2 \\
                                        \text{s.a.} \quad & y_i(w^Tx_i + b) \geq 1, \quad i = 1,2,...,n
                                        \end{align}$$
                                        
                                        Para datos no separables, se introduce variables de holgura $$\xi_i \geq 0$$ y el parámetro de penalización $$C$$:
                                        
                                        $$\begin{align}
                                        \min_{w,b,\xi} \quad & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n} \xi_i \\
                                        \text{s.a.} \quad & y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad i = 1,2,...,n \\
                                        & \xi_i \geq 0, \quad i = 1,2,...,n
                                        \end{align}$$
                                        
                                        $$\textbf{Formulación del problema dual (Wolfe):}$$
                                        
                                        Aplicando multiplicadores de Lagrange y condiciones KKT, el problema se convierte en:
                                        
                                        $$\begin{align}
                                        \max_{\alpha} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
                                        \text{s.a.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1,2,...,n \\
                                        & \sum_{i=1}^{n} \alpha_i y_i = 0
                                        \end{align}$$
                                        
                                        $$\textbf{Truco del kernel:}$$
                                        
                                        Para modelar fronteras no lineales, se aplica la transformación $$\phi: \mathcal{X} \rightarrow \mathcal{H}$$ a un espacio de Hilbert $$\mathcal{H}$$ de alta dimensión, donde el producto interno es:
                                        
                                        $$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{H}}$$
                                        
                                        Kernels comunes incluyen:
                                        
                                        - Lineal: $$K(x_i, x_j) = x_i^T x_j$$
                                        - Polinómico: $$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$
                                        - RBF (Gaussiano): $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$
                                        - Sigmoide: $$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$$
                                        
                                        $$\textbf{Función de decisión:}$$
                                        
                                        La predicción para un nuevo punto $$x$$ viene dada por:
                                        
                                        $$f(x) = \text{sgn}\left(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b\right)$$
                                        
                                        Donde los vectores de soporte son aquellos puntos $$x_i$$ para los cuales $$\alpha_i > 0$$.
                                        
                                        $$\textbf{Probabilidades:}$$
                                        
                                        Para estimar probabilidades, se aplica un modelo logístico a la distancia signada:
                                        
                                        $$P(y=1|x) = \frac{1}{1 + \exp(Af(x) + B)}$$
                                        
                                        Donde $$A$$ y $$B$$ se determinan mediante validación cruzada y maximización de verosimilitud.
                                        
                                        $$\textbf{Complejidad computacional:}$$
                                        
                                        - Entrenamiento: $$\mathcal{O}(n^2 \cdot d)$$ a $$\mathcal{O}(n^3 \cdot d)$$, donde $$n$$ es el número de muestras y $$d$$ es la dimensión
                                        - Predicción: $$\mathcal{O}(n_{sv} \cdot d)$$, donde $$n_{sv}$$ es el número de vectores de soporte
                                        
                                        $$\textbf{Paralelización:}$$
                                        
                                        El entrenamiento de SVM en entornos distribuidos utiliza descomposición de dominio, donde múltiples subproblemas se resuelven en paralelo y luego se combinan mediante métodos de consenso distribuido.
                                        """)
                                    
                                    elif "KNN" in model_type_name or "Neighbors" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{K-Nearest Neighbors - Análisis Teórico Avanzado}$$
                                        
                                        K-NN es un método no paramétrico que se fundamenta en el principio de localidad, basado en la suposición de que los puntos cercanos en el espacio de características tienen alta probabilidad de pertenecer a la misma clase.
                                        
                                        $$\textbf{Definición formal:}$$
                                        
                                        Dado un conjunto de entrenamiento $$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$$ donde $$x_i \in \mathbb{R}^d$$ y $$y_i \in \mathcal{Y}$$, la predicción para un punto nuevo $$x \in \mathbb{R}^d$$ se define como:
                                        
                                        $$\hat{y}(x) = \underset{y \in \mathcal{Y}}{\operatorname{argmax}} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = y)$$
                                        
                                        Donde:
                                        - $$\mathcal{N}_k(x)$$ es el conjunto de índices de los $$k$$ puntos más cercanos a $$x$$
                                        - $\mathbb{I}(\cdot)$ es la función indicadora que devuelve 1 si el argumento es verdadero y 0 en caso contrario
                                        
                                        Para problemas de regresión, la función se convierte en:
                                        
                                        $$\hat{y}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i$$
                                        
                                        $$\textbf{Métricas de distancia:}$$
                                        
                                        La familia general de métricas de Minkowski se define como:
                                        
                                        $$d_p(x_i, x_j) = \left( \sum_{l=1}^{d} |x_{il} - x_{jl}|^p \right)^{1/p}$$
                                        
                                        Casos particulares:
                                        - $$p=1$$: Distancia Manhattan (norma L₁): $$d_1(x_i, x_j) = \sum_{l=1}^{d} |x_{il} - x_{jl}|$$
                                        - $$p=2$$: Distancia Euclidiana (norma L₂): $$d_2(x_i, x_j) = \sqrt{\sum_{l=1}^{d} (x_{il} - x_{jl})^2}$$
                                        - $$p=\infty$$: Distancia Chebyshev (norma L∞): $$d_{\infty}(x_i, x_j) = \max_{l} |x_{il} - x_{jl}|$$
                                        
                                        $$\textbf{Ponderación por distancia:}$$
                                        
                                        En K-NN con pesos, la contribución de cada vecino se pondera inversamente a su distancia:
                                        
                                        $$\hat{y}(x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \cdot y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}$$
                                        
                                        Donde $$w_i = 1/d(x, x_i)^2$$ para ponderación uniforme, o $$w_i = \exp(-d(x, x_i)^2 / h^2)$$ para kernel gaussiano con ancho de banda $$h$$.
                                        
                                        $$\textbf{Propiedades teóricas:}$$
                                        
                                        - Consistencia de Bayes: Cuando $$n \rightarrow \infty$$ y $$k \rightarrow \infty$$ con $$k/n \rightarrow 0$$, el error de clasificación se aproxima al error óptimo de Bayes.
                                        
                                        - Convergencia y tasa de error: Si $$k = \mathcal{O}(n^{4/(d+4)})$$, entonces la tasa de convergencia es $$\mathcal{O}(n^{-2/(d+4)})$$, donde $$d$$ es la dimensionalidad.
                                        
                                        - Maldición de la dimensionalidad: El rendimiento se degrada exponencialmente con la dimensión del espacio debido a la dispersión de los datos.
                                        
                                        $$\textbf{Complejidad computacional:}$$
                                        
                                        - Entrenamiento: $$\mathcal{O}(nd)$$ para almacenar los datos
                                        - Predicción: $$\mathcal{O}(nd + nk)$$ para calcular las distancias y seleccionar los $$k$$ vecinos más cercanos
                                        - Estructuras eficientes: KD-Tree o Ball-Tree reducen la complejidad a $$\mathcal{O}(\log n \cdot d)$$ en el caso promedio
                                        
                                        $$\textbf{Entrenamiento en entornos distribuidos:}$$
                                        
                                        En arquitecturas paralelas, se implementa mediante técnicas de particionamiento espacial y de datos:
                                        
                                        $$\mathcal{D} = \bigcup_{j=1}^{p} \mathcal{D}_j \quad \text{donde } \mathcal{D}_j \text{ reside en el nodo } j$$
                                        
                                        La búsqueda de vecinos se realiza en paralelo y los resultados se combinan mediante una estrategia de consenso distribuido.
                                        """)
                                    
                                    elif "NaiveBayes" in model_type_name or "Gaussian" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Gaussian Naive Bayes - Teoría Probabilística Avanzada}$$
                                        
                                        El fundamento teórico del clasificador Naive Bayes es el Teorema de Bayes, que establece formalmente:
                                        
                                        $$P(y|X) = \frac{P(y)P(X|y)}{P(X)} = \frac{P(y)P(X|y)}{\sum_{y'} P(y')P(X|y')}$$
                                        
                                        Donde $$X = (x_1, x_2, ..., x_p)$$ es el vector de características y $$y$$ es la clase.
                                        
                                        La "ingenuidad" del modelo viene de la suposición de independencia condicional entre características dado $$y$$:
                                        
                                        $$P(X|y) = \prod_{i=1}^{p} P(x_i|y)$$
                                        
                                        Esta suposición reduce la complejidad computacional de $$\mathcal{O}(|Y| \cdot |X|^p)$$ a $$\mathcal{O}(|Y| \cdot p \cdot |X|)$$, donde $$|Y|$$ y $$|X|$$ son las cardinalidades del dominio.
                                        
                                        Para el caso Gaussiano, cada $$P(x_i|y)$$ sigue una distribución normal con parámetros específicos por clase:
                                        
                                        $$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma^2_{y,i}}} \exp\left(-\frac{(x_i-\mu_{y,i})^2}{2\sigma^2_{y,i}}\right)$$
                                        
                                        Los parámetros de esta distribución se calculan mediante estimadores de máxima verosimilitud (MLE):
                                        
                                        $$\mu_{y,i} = \frac{1}{n_y}\sum_{j: y_j=y} x_{j,i} \quad \text{(Media muestral)}$$
                                        
                                        $$\sigma^2_{y,i} = \frac{1}{n_y}\sum_{j: y_j=y} (x_{j,i} - \mu_{y,i})^2 \quad \text{(Varianza muestral)}$$
                                        
                                        Donde $$n_y = |\{j : y_j = y\}|$$ es el número de instancias de la clase $$y$$.
                                        
                                        El parámetro de suavizado (var_smoothing) es una técnica de regularización que previene varianzas cercanas a cero:
                                        
                                        $$\hat{\sigma}^2_{y,i} = \sigma^2_{y,i} + \epsilon \cdot \max_i \sigma^2_i \quad \text{(Suavizado de varianza)}$$
                                        
                                        Esto es equivalente a un estimador MAP (Máximo a Posteriori) con una distribución a priori $\sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta)$.
                                        
                                        Para la clasificación, utilizamos regla MAP (Maximum A Posteriori):
                                        
                                        $$\hat{y} = \arg\max_y P(y|X) = \arg\max_y \left[ P(y) \prod_{i=1}^{p} P(x_i|y) \right]$$
                                        
                                        En la implementación, para evitar subdesbordamiento numérico, trabajamos en el espacio logarítmico:
                                        
                                        $$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{i=1}^{p} \log P(x_i|y) \right]$$
                                        
                                        $$= \arg\max_y \left[ \log P(y) - \frac{1}{2}\sum_{i=1}^{p} \left( \log(2\pi\sigma^2_{y,i}) + \frac{(x_i-\mu_{y,i})^2}{\sigma^2_{y,i}} \right) \right]$$
                                        
                                        La complejidad computacional del entrenamiento es $$\mathcal{O}(n \cdot p)$$ y la de inferencia es $$\mathcal{O}(|Y| \cdot p)$$, donde $$n$$ es el número de instancias de entrenamiento.
                                        
                                        Esta eficiencia es particularmente útil en entornos distribuidos, donde los parámetros $\mu_{y,i}$ y $\sigma^2_{y,i}$ pueden calcularse en paralelo para subconjuntos de datos y luego combinarse mediante fórmulas de actualización secuencial.
                                        """)
                                    
                                    elif "SGD" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Stochastic Gradient Descent Classifier - Fundamentos de Optimización Estocástica}$$
                                        
                                        El clasificador SGD implementa un enfoque iterativo de optimización basado en gradientes estocásticos para minimizar funciones objetivo convexas con diferentes términos de regularización.
                                        
                                        $$\textbf{Formulación general del problema:}$$
                                        
                                        Para un conjunto de datos $$\{(x_i, y_i)\}_{i=1}^n$$ con $$x_i \in \mathbb{R}^p$$ y $$y_i \in \{-1, 1\}$$ (caso binario), se busca resolver:
                                        
                                        $$\min_{\theta \in \mathbb{R}^p} J(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i, \theta)) + \alpha R(\theta)$$
                                        
                                        Donde:
                                        - $$\theta = (w, b)$$ son los parámetros del modelo con $$w \in \mathbb{R}^p$$ y $$b \in \mathbb{R}$$
                                        - $$f(x, \theta) = w^T x + b$$ es la función de decisión lineal
                                        - $$L(y, f)$$ es la función de pérdida
                                        - $$R(\theta)$$ es el término de regularización
                                        - $$\alpha > 0$$ es el coeficiente de regularización (inverse strength $$C = 1/\alpha$$)
                                        
                                        $$\textbf{Funciones de pérdida implementadas:}$$
                                        
                                        1. Pérdida Hinge (SVM): $$L(y, f) = \max(0, 1 - yf)$$
                                           - Gradiente: $$\nabla_{\theta} L = \begin{cases} -y x, & \text{si } yf < 1 \\ 0, & \text{en otro caso} \end{cases}$$
                                        
                                        2. Pérdida logarítmica (Regresión Logística): $$L(y, f) = \log(1 + e^{-yf})$$
                                           - Gradiente: $$\nabla_{\theta} L = -\frac{yx}{1 + e^{yf}}$$
                                        
                                        3. Pérdida Huber modificada: $$L(y, f) = \begin{cases} \max(0, 1 - yf)^2, & \text{si } yf > -1 \\ -4yf, & \text{en otro caso} \end{cases}$$
                                           - Combina propiedades de hinge y log-loss, más robusta y diferenciable
                                        
                                        4. Pérdida Perceptrón: $$L(y, f) = \max(0, -yf)$$
                                           - Equivalente a hinge con margen cero
                                        
                                        5. Pérdida Squared Hinge: $$L(y, f) = \max(0, 1 - yf)^2$$
                                           - Versión suavizada y diferenciable de hinge
                                        
                                        $$\textbf{Términos de regularización:}$$
                                        
                                        1. L2 (Ridge): $$R(\theta) = \frac{1}{2}\|w\|_2^2 = \frac{1}{2}\sum_{j=1}^{p} w_j^2$$
                                           - Promueve pesos pequeños uniformemente
                                           - Gradiente: $$\nabla_{w} R = w$$
                                        
                                        2. L1 (Lasso): $$R(\theta) = \|w\|_1 = \sum_{j=1}^{p} |w_j|$$
                                           - Induce soluciones dispersas (muchos coeficientes nulos)
                                           - Subgradiente: $$\partial_{w_j} R = \text{sign}(w_j)$$
                                        
                                        3. ElasticNet (combinación L1+L2): $$R(\theta) = \rho\|w\|_1 + \frac{1-\rho}{2}\|w\|_2^2$$
                                           - Combina las ventajas de L1 y L2
                                           - Parámetro $$\rho \in [0,1]$$ controla el balance entre L1 y L2
                                        
                                        $$\textbf{Algoritmo de optimización SGD:}$$
                                        
                                        1. Inicializar $$\theta^{(0)} = 0$$ o aleatoriamente en $$\mathbb{R}^p$$
                                        2. Para $$t = 1, 2, \ldots, T$$:
                                           - Seleccionar aleatoriamente un índice $$i_t \in \{1, 2, \ldots, n\}$$
                                           - Calcular el gradiente: $$g^{(t)} = \nabla_{\theta} L(y_{i_t}, f(x_{i_t}, \theta^{(t-1)})) + \alpha \nabla_{\theta} R(\theta^{(t-1)})$$
                                           - Actualizar: $$\theta^{(t)} = \theta^{(t-1)} - \eta_t \cdot g^{(t)}$$
                                        3. Devolver $$\theta^{(T)}$$
                                        
                                        Donde $$\eta_t > 0$$ es la tasa de aprendizaje en la iteración $$t$$.
                                        
                                        $$\textbf{Programación de la tasa de aprendizaje:}$$
                                        
                                        1. Constante: $\eta_t = \eta_0$
                                           - Simple pero puede ser subóptima
                                        
                                        2. Optimal: $\eta_t = \frac{1}{\alpha (t_0 + t)}$
                                           - Programación teóricamente óptima para SGD
                                           - $$t_0$$ es un parámetro de escala (por defecto 1.0)
                                        
                                        3. Invscaling: $\eta_t = \eta_0 \cdot t^{-\text{power}}$
                                           - Disminución por ley de potencia
                                           - $\text{power} \in (0,1]$ controla la velocidad de disminución
                                        
                                        4. Adaptive: $\eta_t = \eta_0$ si la pérdida mejora, sino $\eta_t = \eta_{t-1} / 5$
                                           - Reduce la tasa cuando el progreso se estanca
                                        
                                        $$\textbf{Análisis teórico:}$$
                                        
                                        Para funciones convexas y Lipschitz continuas, con regularización L2 y esquema optimal, SGD converge a la solución óptima $$\theta^*$$ con tasa:
                                        
                                        $$\mathbb{E}[J(\bar{\theta}_T) - J(\theta^*)] \leq \frac{R^2 + G^2 \log(T)}{2\alpha T}$$
                                        
                                        Donde:
                                        - $$\bar{\theta}_T = \frac{1}{T}\sum_{t=1}^{T} \theta^{(t)}$$ es el promedio de iteraciones
                                        - $$R = \|\theta^{(0)} - \theta^*\|_2$$ es la distancia inicial a la solución
                                        - $$G$$ es una cota superior para la norma del gradiente estocástico
                                        
                                        $$\textbf{Extensión a clasificación multiclase:}$$
                                        
                                        1. One-vs-Rest (OvR): Entrena $$K$$ clasificadores binarios (uno por clase)
                                           - Cada clasificador $$f_k(x) = w_k^T x + b_k$$ separa la clase $$k$$ del resto
                                           - Predicción: $$\hat{y} = \arg\max_k f_k(x)$$
                                        
                                        2. Pérdida multiclase directa:
                                           $$L(y, f) = -\log\left(\frac{e^{f_y}}{\sum_{k=1}^{K} e^{f_k}}\right)$$
                                           donde $$f_k$$ es la puntuación para la clase $$k$$
                                        
                                        $$\textbf{Implementación distribuida:}$$
                                        
                                        En entornos distribuidos, SGD puede implementarse mediante:
                                        
                                        1. Paralelización asíncrona (Hogwild!): Múltiples hilos actualizan parámetros en una variable compartida sin bloqueos
                                           - Converge para problemas dispersos donde las actualizaciones concurrentes afectan a diferentes subconjuntos de parámetros
                                        
                                        2. Paralelización por minilotes: Divide cada minilote entre trabajadores y agrega gradientes
                                           - Actualización: $$\theta^{(t+1)} = \theta^{(t)} - \eta_t \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_{\theta} L(y_i, f(x_i, \theta^{(t)}))$$
                                        
                                        3. Promediado de modelos (model averaging): Entrena modelos independientes en subconjuntos de datos y promedia parámetros
                                           - $$\bar{\theta} = \frac{1}{K}\sum_{k=1}^{K} \theta^{(k)}$$
                                        
                                        $$\textbf{Complejidad computacional:}$$
                                        
                                        - Entrenamiento: $$\mathcal{O}(n \cdot p \cdot T)$$ en el peor caso, pero típicamente mucho menor por la convergencia temprana
                                        - Predicción: $$\mathcal{O}(p)$$ para clasificación binaria, $$\mathcal{O}(K \cdot p)$$ para $$K$$ clases con OvR
                                        """)
                                    
                                    elif "Tree" in model_type_name or "Decision" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{Decision Tree Classifier - Teoría Avanzada y Formalización Matemática}$$
                                        
                                        Los árboles de decisión son modelos no paramétricos que particionan recursivamente el espacio de características en regiones disjuntas, asignando a cada región una etiqueta de clase mediante un proceso jerárquico de toma de decisiones.
                                        
                                        $$\textbf{Estructura matemática formal:}$$
                                        
                                        Un árbol de decisión $$T$$ se define como un grafo acíclico dirigido $$T = (V, E)$$ donde:
                                        - $$V = V_{internal} \cup V_{leaf}$$ es el conjunto de nodos (internos y hojas)
                                        - $$E \subset V \times V$$ es el conjunto de aristas que conectan nodos padre con hijos
                                        - Cada nodo interno $$v \in V_{internal}$$ tiene asociada una función de decisión $$d_v: \mathcal{X} \rightarrow \{0,1\}$$
                                        - Cada nodo hoja $$l \in V_{leaf}$$ tiene asociada una distribución de probabilidad $$p_l$$ sobre el espacio de etiquetas $$\mathcal{Y}$$
                                        
                                        Para clasificación, la predicción se obtiene recorriendo el árbol desde la raíz hasta una hoja siguiendo las decisiones en cada nodo interno, y seleccionando la clase más probable en la hoja alcanzada:
                                        
                                        $$\hat{y}(x) = \arg\max_{y \in \mathcal{Y}} p_{l(x)}(y)$$
                                        
                                        donde $$l(x)$$ es la hoja alcanzada por el punto $$x$$.
                                        
                                        $\textbf{Construcción óptima del árbol:}$
                                        
                                        El problema de construir un árbol óptimo es NP-completo, por lo que se utilizan algoritmos voraces como CART, ID3, C4.5 o C5.0 que construyen el árbol de arriba hacia abajo maximizando la ganancia de información en cada división.
                                        
                                        Sea $$t$$ un nodo con conjunto de muestras $$S_t$$. Para cada posible división $$s$$ que genera nodos hijos $$t_L$$ y $$t_R$$ con conjuntos de muestras $$S_{t_L}$$ y $$S_{t_R}$$ respectivamente, se define:
                                        
                                        $$\Delta I(s, t) = I(t) - \frac{|S_{t_L}|}{|S_t|} I(t_L) - \frac{|S_{t_R}|}{|S_t|} I(t_R)$$
                                        
                                        Donde $$I(t)$$ es una medida de impureza del nodo $$t$$. La división óptima $$s^*$$ es:
                                        
                                        $$s^* = \arg\max_{s \in \mathcal{S}} \Delta I(s, t)$$
                                        
                                        siendo $\mathcal{S}$ el conjunto de todas las posibles divisiones.
                                        
                                        $\textbf{Medidas de impureza para clasificación:}$
                                        
                                        1. Índice de Gini:
                                        
                                           $$I_{Gini}(t) = 1 - \sum_{c=1}^{C} p(c|t)^2 = \sum_{c=1}^{C} p(c|t)(1-p(c|t))$$
                                           
                                           Mide la probabilidad esperada de clasificación incorrecta si las etiquetas se asignan aleatoriamente según la distribución de clases en el nodo.
                                        
                                        2. Entropía:
                                        
                                           $$I_{Entropy}(t) = -\sum_{c=1}^{C} p(c|t) \log_2 p(c|t)$$
                                           
                                           Mide la incertidumbre sobre la clase de un elemento elegido aleatoriamente del nodo.
                                        
                                        3. Log-Loss (Error de Clasificación):
                                        
                                           $$I_{LogLoss}(t) = 1 - \max_{c} p(c|t)$$
                                           
                                           Mide la probabilidad de error si se asigna la clase mayoritaria.
                                        
                                        donde $$p(c|t) = \frac{|S_{t,c}|}{|S_t|}$$ es la proporción de muestras de clase $$c$$ en el nodo $$t$$.
                                        
                                        $\textbf{Criterios de división para características continuas:}$
                                        
                                        Para una característica continua $$X_j$$, se consideran divisiones del tipo $$X_j \leq \theta$$. Los posibles valores de $$\theta$$ son los puntos medios entre valores consecutivos de $$X_j$$ en las muestras ordenadas.
                                        
                                        $\textbf{Criterios de división para características categóricas:}$
                                        
                                        Para una característica categórica $$X_j$$ con dominio $$\{a_1, a_2, ..., a_k\}$$, se consideran divisiones que particionan este dominio en dos subconjuntos complementarios: $$X_j \in A$$ y $$X_j \in \bar{A}$$ donde $$A \subset \{a_1, a_2, ..., a_k\}$$.
                                        
                                        $\textbf{Criterios de parada y poda:}$
                                        
                                        Para evitar el sobreajuste, se utilizan criterios como:
                                        
                                        1. Límite en la profundidad máxima $$D$$
                                        2. Número mínimo de muestras para dividir un nodo: $$|S_t| \geq \text{min\_samples\_split}$$
                                        3. Número mínimo de muestras en cada nodo hijo: $$|S_{t_L}|, |S_{t_R}| \geq \text{min\_samples\_leaf}$$
                                        4. Umbral mínimo de reducción de impureza: $$\Delta I(s^*, t) \geq \text{min\_impurity\_decrease}$$
                                        5. Número máximo de características a considerar en cada división: $$\sqrt{p}$$ (clasificación) o $$p/3$$ (regresión)
                                        
                                        Además, se pueden aplicar técnicas de poda post-entrenamiento como cost-complexity pruning:
                                        
                                        $$T_{\alpha} = \arg\min_{T'} \sum_{t \in T'_{leaf}} |S_t| I(t) + \alpha |T'_{leaf}|$$
                                        
                                        donde $$\alpha \geq 0$$ es el parámetro de complejidad y $$|T'_{leaf}|$$ es el número de hojas en el subárbol $$T'$$.
                                        
                                        $\textbf{Propiedades teóricas:}$
                                        
                                        1. Consistencia: Un árbol con profundidad $$\log_2(n)$$ donde $$n$$ es el tamaño de la muestra, puede aproximar cualquier función de decisión con error arbitrariamente pequeño cuando $$n \rightarrow \infty$$.
                                        
                                        2. Complejidad de muestra: Para alcanzar un error $$\epsilon$$ con probabilidad $$1-\delta$$, se requieren $$\mathcal{O}(h \log(1/\epsilon) + \log(1/\delta)/\epsilon)$$ muestras, donde $$h$$ es la dimensión VC del espacio de hipótesis.
                                        
                                        3. Varianza y sesgo: Los árboles tienen bajo sesgo pero alta varianza, lo que los hace candidatos ideales para técnicas de ensemble como bagging (Random Forest) o boosting.
                                        
                                        $\textbf{Importancia de características:}$
                                        
                                        La importancia de una característica $$X_j$$ se puede cuantificar como la reducción total ponderada de impureza aportada por todas las divisiones basadas en $$X_j$$:
                                        
                                        $$Imp(X_j) = \sum_{t \in V_{internal}} \mathbb{I}(v(t) = j) \cdot \frac{|S_t|}{|S|} \cdot \Delta I(s_t, t)$$
                                        
                                        donde $$v(t)$$ es el índice de la característica utilizada para la división en el nodo $$t$$, y $$s_t$$ es la división elegida.
                                        
                                        $\textbf{Complejidad computacional:}$
                                        
                                        - Construcción: $$\mathcal{O}(n \log(n) \cdot p \cdot D)$$ donde $$n$$ es el número de muestras, $$p$$ es el número de características y $$D$$ es la profundidad máxima
                                        - Predicción: $$\mathcal{O}(D)$$ por muestra, típicamente logarítmica en el número de hojas
                                        
                                        $\textbf{Implementación distribuida:}$
                                        
                                        En entornos distribuidos, se pueden implementar variantes como:
                                        
                                        1. Paralelización de la evaluación de divisiones candidatas
                                        2. PLANET: construcción distribuida donde los nodos se procesan en lotes del mismo nivel
                                        3. Árboles de histograma: discretización de características continuas para permitir paralelización eficiente
                                        
                                        $\textbf{Limitaciones y extensiones:}$
                                        
                                        1. Inestabilidad ante pequeñas variaciones en los datos (mitigado mediante ensemble methods)
                                        2. Dificultad para capturar relaciones lineales simples (requiere muchas divisiones)
                                        3. Extensiones: árboles de decisión oblicuos, árboles Bayesianos, árboles con multivariate splits
                                        """)
                                    
                                    elif "Ada" in model_type_name:
                                        st.markdown(r"""
                                        $$\textbf{AdaBoost Classifier - Teoría de Boosting y Análisis Matemático}$$
                                        
                                        AdaBoost (Adaptive Boosting) es un algoritmo meta-estimador que combina múltiples clasificadores débiles en un clasificador robusto mediante el enfoque de boosting adaptativo, centrado en las muestras difíciles de clasificar.
                                        
                                        $$\textbf{Marco teórico formal:}$$
                                        
                                        Dado un conjunto de entrenamiento $$\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$$ donde $$x_i \in \mathcal{X}$$ y $$y_i \in \{-1, +1\}$$ (caso binario), AdaBoost construye un clasificador fuerte como combinación lineal de clasificadores débiles:
                                        
                                        $$F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$
                                        
                                        Donde:
                                        - $$h_m: \mathcal{X} \rightarrow \{-1, +1\}$$ es el clasificador débil de la iteración $$m$$
                                        - $$\alpha_m > 0$$ es el peso asignado al clasificador $$h_m$$
                                        - $$M$$ es el número total de clasificadores débiles
                                        
                                        La regla de decisión final es $$\hat{y} = \text{sign}(F(x))$$.
                                        
                                        $$\textbf{Algoritmo SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss):}$$
                                        
                                        1. Inicializar pesos uniformes $$w_i^{(1)} = \frac{1}{n}$$ para $$i = 1, 2, ..., n$$
                                        
                                        2. Para $$m = 1$$ hasta $$M$$:
                                           - Normalizar los pesos: $$\tilde{w}_i^{(m)} = \frac{w_i^{(m)}}{\sum_{j=1}^{n} w_j^{(m)}}$$
                                           - Ajustar clasificador débil $$h_m(x)$$ usando pesos $$\tilde{w}_i^{(m)}$$
                                           - Calcular error ponderado: $$\epsilon_m = \sum_{i=1}^{n} \tilde{w}_i^{(m)} \mathbb{I}(y_i \neq h_m(x_i))$$
                                           - Calcular coeficiente: $$\alpha_m = \log\left(\frac{1-\epsilon_m}{\epsilon_m}\right) + \log(K-1)$$ donde $$K$$ es el número de clases (para binario $$K=2$$)
                                           - Actualizar pesos: $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp(\alpha_m \cdot \mathbb{I}(y_i \neq h_m(x_i)))$$
                                        
                                        3. Devolver clasificador final: $$F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$
                                        
                                        $$\textbf{Algoritmo SAMME.R (Real AdaBoost):}$$
                                        
                                        Variante que utiliza estimaciones de probabilidad en lugar de clasificaciones discretas:
                                        
                                        1. Inicializar pesos uniformes $$w_i^{(1)} = \frac{1}{n}$$ para $$i = 1, 2, ..., n$$
                                        
                                        2. Para $$m = 1$$ hasta $$M$$:
                                           - Normalizar los pesos: $$\tilde{w}_i^{(m)} = \frac{w_i^{(m)}}{\sum_{j=1}^{n} w_j^{(m)}}$$
                                           - Ajustar clasificador débil $$h_m(x)$$ usando pesos $$\tilde{w}_i^{(m)}$$
                                           - Estimar probabilidades de clase $$p_m(y|x)$$ para cada $$y \in \{-1, +1\}$$
                                           - Calcular predicción: $$\hat{h}_m(x) = \frac{1}{2} \log\frac{p_m(y=1|x)}{p_m(y=-1|x)}$$
                                           - Actualizar pesos: $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp(-y_i \hat{h}_m(x_i))$$
                                        
                                        3. Devolver clasificador final: $$F(x) = \sum_{m=1}^{M} \hat{h}_m(x)$$
                                        
                                        $$\textbf{Interpretación como optimización en el espacio funcional:}$$
                                        
                                        AdaBoost minimiza la función de pérdida exponencial de forma aditiva paso a paso:
                                        
                                        $$J(F) = \frac{1}{n} \sum_{i=1}^{n} e^{-y_i F(x_i)}$$
                                        
                                        En cada iteración $$m$$:
                                        
                                        $$(\alpha_m, h_m) = \underset{\alpha, h}{\arg\min} \sum_{i=1}^{n} w_i^{(m)} \exp(-\alpha y_i h(x_i))$$
                                        
                                        Esto es equivalente a un descenso de gradiente funcional, donde:
                                        
                                        1. El gradiente funcional negativo es $\tilde{y}_i = \frac{\partial e^{-y_i F(x_i)}}{\partial F(x_i)} = -y_i e^{-y_i F(x_i)}$
                                        2. El clasificador débil $h_m$ aproxima la dirección de este gradiente
                                        3. El tamaño de paso $$\alpha_m$$ se determina mediante búsqueda lineal
                                        
                                        $$\textbf{Propiedades teóricas:}$$
                                        
                                        1. $$\textbf{Cota superior del error de entrenamiento:}$$ Si cada clasificador débil tiene un error $$\epsilon_m < 1/2$$, entonces:
                                        
                                           $$\frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(y_i \neq \text{sign}(F_M(x_i))) \leq \frac{1}{n} \sum_{i=1}^{n} e^{-y_i F_M(x_i)} \leq \prod_{m=1}^{M} \sqrt{4\epsilon_m(1-\epsilon_m)}$$
                                           
                                           Esta cota decrece exponencialmente con $$M$$ si cada $$\epsilon_m < 1/2 - \delta$$ para algún $$\delta > 0$$.
                                        
                                        2. $$\textbf{Margen y generalización:}$$ El error de generalización está acotado por:
                                        
                                           $$\text{Error de generalización} \leq \text{Error emp. de margen} + \tilde{O}\left(\sqrt{\frac{d}{n}}\right)$$
                                           
                                           donde $$d$$ es la dimensión VC del espacio de clasificadores débiles.
                                        
                                        3. $$\textbf{Resistencia al sobreajuste:}$$ AdaBoost es sorprendentemente resistente al sobreajuste cuando se incrementa $$M$$, debido a la maximización implícita del margen.
                                        
                                        $$\textbf{Extensión a problemas multiclase:}$$
                                        
                                        Para $$K > 2$$ clases, AdaBoost utiliza estrategias como:
                                        
                                        1. SAMME: extensión directa del algoritmo binario al caso multiclase
                                           - El término adicional $$\log(K-1)$$ en $$\alpha_m$$ asegura que el clasificador sea mejor que la asignación aleatoria
                                        
                                        2. Formulación de codificación uno-contra-todos:
                                           - Se entrenan $$K$$ clasificadores binarios $$F_k(x)$$, cada uno distinguiendo la clase $$k$$ del resto
                                           - La predicción final es $$\hat{y} = \arg\max_k F_k(x)$$
                                        
                                        $$\textbf{Complejidad computacional:}$$
                                        
                                        - Entrenamiento: $$\mathcal{O}(M \cdot n \cdot T_h)$$ donde $$T_h$$ es el tiempo para entrenar un clasificador débil
                                        - Predicción: $$\mathcal{O}(M \cdot T_p)$$ donde $$T_p$$ es el tiempo para una predicción del clasificador débil
                                        
                                        $\textbf{Implementación distribuida:}$
                                        
                                        Aunque AdaBoost es secuencial por naturaleza, admite ciertas paralelizaciones:
                                        
                                        1. Entrenamiento paralelo de clasificadores débiles en cada iteración
                                        2. Particionamiento horizontal de datos con actualizaciones de pesos sincronizadas
                                        3. Esquemas aproximados donde conjuntos de clasificadores débiles se entrenan en paralelo y luego se combinan
                                        """)
                                    
                                    else:
                                        st.markdown(r"""
                                        $$\textbf{Modelo de Machine Learning - Formulación General}$$
                                        
                                        Los modelos de clasificación buscan una función $$f: \mathcal{X} \rightarrow \mathcal{Y}$$ que mapee características a etiquetas de clase:
                                        
                                        $$\hat{y} = f(x)$$
                                        
                                        El objetivo del entrenamiento es encontrar los parámetros $$\theta$$ que minimicen una función de pérdida $$L$$:
                                        
                                        $$\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i, \theta)) + \lambda R(\theta)$$
                                        
                                        Donde:
                                        - $$L$$ es la función de pérdida que mide el error de predicción
                                        - $$R$$ es un término de regularización para evitar el sobreajuste
                                        - $$\lambda$$ es un hiperparámetro que controla la fuerza de la regularización
                                        """)
                                
                                # Guardar información del dataset y actualizar metadatos solo para el modelo seleccionado
                                model_id = selected_model_id
                                metrics = selected_metrics
                    
                    finally:
                        # Eliminar el archivo temporal
                        import os
                        if os.path.exists(temp_csv_path):
                            os.unlink(temp_csv_path)
                    
                    # Para cada modelo, guardar información del dataset
                    for model_name, model_id in model_ids.items():
                        model_metrics = model_registry.get_metadata(model_id).get('metrics', {})
                        model_type_name = model_name.split('_')[0] if '_' in model_name else model_name
                        
                        # Obtener los parámetros correctos para este modelo específico
                        model_specific_params = {}
                        for config in models_config:
                            if config['name'] == model_name:
                                model_specific_params = config.get('params', {})
                                break
                        
                        dataset_info = {
                            "name": dataset_choice if data_option == "Conjuntos de datos de ejemplo" else "Custom Dataset",
                            "n_samples": len(df),
                            "n_features": len(df.columns) - 1,
                            "target_column": target_column,
                            "source_path": file_path if data_option == "Ingresar ruta a un archivo" else None
                        }
                        
                        # Actualizar metadatos de cada modelo
                        model_registry.update_metadata(
                            model_id,
                            {
                                "name": f"{model_type_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                                "model_type": model_type_name,
                                "params": model_specific_params,
                                "training_time": end_time - start_time,
                                "dataset_info": dataset_info,
                                "test_size": test_size,
                                "ray_config": {
                                    "use_ray": use_ray,
                                    "num_workers": num_workers
                                }
                            }
                        )
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

# Página de Modelos
elif pagina == "Modelos":
    st.header("📊 Gestión de Modelos")
    st.markdown("""
    Esta sección muestra información detallada sobre los modelos entrenados y permite gestionar el registro de modelos.
    """)
    
    # Comprobar si hay modelos en el registro
    try:
        all_models = model_registry.list_models()
        
        if not all_models:
            st.warning("No hay modelos entrenados en el registro. Vaya a la página de Entrenamiento para entrenar modelos.")
        else:
            # Mostrar lista de modelos
            st.subheader("Modelos Disponibles")
            
            # Crear una tabla con información básica de los modelos
            models_data = []
            for model_id in all_models:
                metadata = model_registry.get_metadata(model_id)
                models_data.append({
                    "ID": model_id,
                    "Nombre": metadata.get('name', 'Sin nombre'),
                    "Tipo": metadata.get('model_type', 'Desconocido'),
                    "Precisión": f"{metadata.get('metrics', {}).get('accuracy', 0):.4f}",
                    "F1-Score": f"{metadata.get('metrics', {}).get('f1', 0):.4f}",
                    "Entrenado": metadata.get('created_at', 'Desconocido')
                })
            
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df, use_container_width=True)
            
            # Permitir seleccionar un modelo para ver detalles
            selected_model_id = st.selectbox(
                "Seleccione un modelo para ver detalles:",
                all_models,
                format_func=lambda x: f"{model_registry.get_metadata(x).get('name', 'Sin nombre')} (ID: {x})"
            )
            
            if selected_model_id:
                st.subheader(f"Detalles del Modelo: {model_registry.get_metadata(selected_model_id).get('name', 'Sin nombre')}")
                
                # Obtener metadatos completos
                metadata = model_registry.get_metadata(selected_model_id)
                
                # Organizar la información en pestañas
                tab1, tab2, tab3, tab4 = st.tabs(["Información General", "Parámetros", "Métricas", "Dataset"])
                
                with tab1:
                    st.markdown("### Información General")
                    st.write(f"**ID del Modelo:** {selected_model_id}")
                    st.write(f"**Nombre:** {metadata.get('name', 'Sin nombre')}")
                    st.write(f"**Tipo:** {metadata.get('model_type', 'Desconocido')}")
                    st.write(f"**Fecha de Creación:** {metadata.get('created_at', 'Desconocido')}")
                    st.write(f"**Tiempo de Entrenamiento:** {metadata.get('training_time', 0):.2f} segundos")
                    
                    # Botones de acción
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Exportar Modelo", key=f"export_{selected_model_id}"):
                            try:
                                export_path = os.path.join(os.getcwd(), "exported_models", f"{metadata.get('name', 'model')}_{selected_model_id}.pkl")
                                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                                model_registry.export_model(selected_model_id, export_path)
                                st.success(f"Modelo exportado correctamente a: {export_path}")
                            except Exception as e:
                                st.error(f"Error al exportar modelo: {str(e)}")
                    
                    with col2:
                        if st.button("Eliminar Modelo", key=f"delete_{selected_model_id}"):
                            try:
                                model_registry.delete_model(selected_model_id)
                                st.success(f"Modelo {selected_model_id} eliminado correctamente.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error al eliminar modelo: {str(e)}")
                    
                    with col3:
                        if st.button("Actualizar Nombre", key=f"rename_{selected_model_id}"):
                            new_name = st.text_input("Nuevo nombre:", value=metadata.get('name', ''))
                            if new_name and new_name != metadata.get('name', ''):
                                try:
                                    metadata['name'] = new_name
                                    model_registry.update_metadata(selected_model_id, metadata)
                                    st.success(f"Nombre actualizado a: {new_name}")
                                except Exception as e:
                                    st.error(f"Error al actualizar nombre: {str(e)}")
                
                with tab2:
                    st.markdown("### Parámetros del Modelo")
                    params = metadata.get('params', {})
                    if params:
                        params_df = pd.DataFrame(params.items(), columns=["Parámetro", "Valor"])
                        st.dataframe(params_df)
                    else:
                        st.info("No hay información sobre los parámetros del modelo.")
                
                with tab3:
                    st.markdown("### Métricas de Rendimiento")
                    metrics = metadata.get('metrics', {})
                    if metrics:
                        metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
                        
                        # Mostrar métricas en una tabla
                        st.dataframe(metrics_df)
                        
                        # Visualizar métricas como gráfico
                        fig = px.bar(metrics_df, x='Métrica', y='Valor', text='Valor')
                        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig.update_layout(
                            title="Métricas de rendimiento del modelo",
                            xaxis_title="Métrica",
                            yaxis_title="Valor",
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay métricas disponibles para este modelo.")
                
                with tab4:
                    st.markdown("### Información del Dataset")
                    dataset_info = metadata.get('dataset_info', {})
                    if dataset_info:
                        st.write(f"**Nombre:** {dataset_info.get('name', 'No especificado')}")
                        st.write(f"**Número de muestras:** {dataset_info.get('n_samples', 'No especificado')}")
                        st.write(f"**Número de características:** {dataset_info.get('n_features', 'No especificado')}")
                        st.write(f"**Columna objetivo:** {dataset_info.get('target_column', 'No especificado')}")
                        
                        # Si hay una ruta al archivo fuente, mostrarla
                        if dataset_info.get('source_path'):
                            st.write(f"**Ruta al archivo fuente:** {dataset_info.get('source_path')}")
                    else:
                        st.info("No hay información disponible sobre el dataset de entrenamiento.")
    
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")

# Página de Predicciones
elif pagina == "Predicciones":
    st.header("🔮 Predicciones")
    st.markdown("""
    En esta sección puede utilizar los modelos entrenados para hacer predicciones con nuevos datos.
    """)
    
    # Comprobar si hay modelos en el registro
    try:
        all_models = model_registry.list_models()
        
        if not all_models:
            st.warning("No hay modelos entrenados en el registro. Vaya a la página de Entrenamiento para entrenar modelos.")
        else:
            # Selector de modelo para hacer predicciones
            selected_model_id = st.selectbox(
                "Seleccione un modelo para hacer predicciones:",
                all_models,
                format_func=lambda x: f"{model_registry.get_metadata(x).get('name', 'Sin nombre')} (ID: {x})"
            )
            
            if selected_model_id:
                # Obtener metadatos del modelo
                metadata = model_registry.get_metadata(selected_model_id)
                dataset_info = metadata.get('dataset_info', {})
                
                # Opciones para entrada de datos
                input_option = st.radio(
                    "Seleccione el método de entrada:",
                    ["Formulario", "Archivo CSV"]
                )
                
                # Función para hacer predicciones
                def hacer_prediccion(model_id, data):
                    try:
                        # Cargar el modelo
                        model = model_registry.load_model(model_id)
                        # Hacer predicción
                        predictions = model.predict(data)
                        return predictions
                    except Exception as e:
                        st.error(f"Error al hacer predicción: {str(e)}")
                        return None
                
                if input_option == "Formulario":
                    # Si tenemos información del dataset, intentamos crear un formulario apropiado
                    if dataset_info and dataset_info.get('source_path'):
                        try:
                            # Cargar el dataset original para obtener los nombres de las columnas
                            orig_dataset = pd.read_csv(dataset_info['source_path'])
                            feature_columns = [col for col in orig_dataset.columns if col != dataset_info.get('target_column')]
                            
                            st.subheader("Ingrese los valores para las características:")
                            
                            # Crear formulario dinámicamente basado en las columnas
                            with st.form("prediction_form"):
                                # Crear un diccionario para almacenar los valores
                                values = {}
                                
                                # Crear varios campos para las características
                                cols = st.columns(3)
                                for i, col in enumerate(feature_columns):
                                    idx = i % 3
                                    with cols[idx]:
                                        # Usar estadísticas del conjunto original para sugerir valores
                                        mean_val = orig_dataset[col].mean()
                                        min_val = orig_dataset[col].min()
                                        max_val = orig_dataset[col].max()
                                        
                                        # Decidir si usar slider o input
                                        if len(orig_dataset[col].unique()) < 10 and orig_dataset[col].dtype != 'object':
                                            values[col] = st.slider(
                                                f"{col}:",
                                                min_value=float(min_val),
                                                max_value=float(max_val),
                                                value=float(mean_val)
                                            )
                                        else:
                                            values[col] = st.number_input(
                                                f"{col}:",
                                                value=float(mean_val),
                                                step=0.1
                                            )
                                
                                submitted = st.form_submit_button("Hacer Predicción")
                                
                                if submitted:
                                    # Convertir a DataFrame
                                    input_df = pd.DataFrame([values])
                                    
                                    # Hacer predicción
                                    predictions = hacer_prediccion(selected_model_id, input_df)
                                    
                                    if predictions is not None:
                                        st.success(f"Predicción: {predictions[0]}")
                                        
                                        # Mostrar probabilidades si el modelo lo soporta
                                        try:
                                            model = model_registry.load_model(selected_model_id)
                                            if hasattr(model, 'predict_proba'):
                                                probas = model.predict_proba(input_df)
                                                if len(probas[0]) > 1:  # Para modelos de clasificación
                                                    st.subheader("Probabilidades por clase:")
                                                    prob_df = pd.DataFrame(probas[0], columns=['Probabilidad'])
                                                    prob_df.index.name = 'Clase'
                                                    st.dataframe(prob_df)
                                                    
                                                    # Crear gráfico de probabilidades
                                                    fig = px.bar(
                                                        x=list(range(len(probas[0]))),
                                                        y=probas[0],
                                                        labels={'x': 'Clase', 'y': 'Probabilidad'}
                                                    )
                                                    fig.update_layout(title="Probabilidades por clase")
                                                    st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.info("No se pudieron calcular probabilidades para este modelo.")
                        except Exception as e:
                            st.error(f"Error al crear formulario de predicción: {str(e)}")
                            
                            # Formulario manual como fallback
                            st.subheader("Ingrese los valores manualmente:")
                            with st.form("manual_prediction_form"):
                                # Entrada de texto para valores separados por comas
                                values_input = st.text_area(
                                    "Ingrese los valores separados por comas (en el mismo orden que las características del modelo):"
                                )
                                
                                submitted = st.form_submit_button("Hacer Predicción")
                                
                                if submitted and values_input:
                                    try:
                                        # Convertir texto a valores
                                        values_list = [float(x.strip()) for x in values_input.split(',')]
                                        # Crear DataFrame
                                        values = pd.DataFrame([values_list])
                                        
                                        # Hacer predicción
                                        predictions = hacer_prediccion(selected_model_id, values)
                                        
                                        if predictions is not None:
                                            st.success(f"Predicción: {predictions[0]}")
                                    except Exception as e:
                                        st.error(f"Error al procesar la entrada: {str(e)}")
                    else:
                        # Si no tenemos información del dataset, usar un formulario manual
                        st.warning("No hay información disponible sobre el conjunto de datos original. Por favor ingrese los valores manualmente.")
                        
                        with st.form("manual_prediction_form"):
                            # Entrada de texto para valores separados por comas
                            values_input = st.text_area(
                                "Ingrese los valores separados por comas (en el mismo orden que las características del modelo):"
                            )
                            
                            submitted = st.form_submit_button("Hacer Predicción")
                            
                            if submitted and values_input:
                                try:
                                    # Convertir texto a valores
                                    values_list = [float(x.strip()) for x in values_input.split(',')]
                                    # Crear DataFrame
                                    values = pd.DataFrame([values_list])
                                    
                                    # Hacer predicción
                                    predictions = hacer_prediccion(selected_model_id, values)
                                    
                                    if predictions is not None:
                                        st.success(f"Predicción: {predictions[0]}")
                                except Exception as e:
                                    st.error(f"Error al procesar la entrada: {str(e)}")
                
                elif input_option == "Archivo CSV":
                    st.subheader("Cargar archivo CSV para predicciones por lotes")
                    
                    uploaded_file = st.file_uploader("Suba un archivo CSV con datos para predecir", type=["csv"])
                    
                    if uploaded_file is not None:
                        try:
                            prediction_df = pd.read_csv(uploaded_file)
                            st.write("Vista previa de los datos:")
                            st.dataframe(prediction_df.head(), use_container_width=True)
                            
                            # Botón para hacer predicciones
                            if st.button("Hacer Predicciones"):
                                # Hacer predicciones
                                predictions = hacer_prediccion(selected_model_id, prediction_df)
                                
                                if predictions is not None:
                                    # Añadir predicciones al DataFrame
                                    prediction_df['prediccion'] = predictions
                                    
                                    # Mostrar resultados
                                    st.success("¡Predicciones completadas!")
                                    st.subheader("Resultados:")
                                    st.dataframe(prediction_df, use_container_width=True)
                                    
                                    # Opción para descargar resultados
                                    csv = prediction_df.to_csv(index=False)
                                    st.download_button(
                                        label="Descargar resultados como CSV",
                                        data=csv,
                                        file_name="predicciones.csv",
                                        mime="text/csv"
                                    )
                        except Exception as e:
                            st.error(f"Error al procesar el archivo: {str(e)}")
    
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")

# Página de Monitorización del Sistema
elif pagina == "Monitorización":
    st.header("🔍 Monitorización del Sistema")
    st.markdown("""
    Esta sección proporciona información en tiempo real sobre el estado del sistema y los recursos utilizados.
    Visualiza el rendimiento del cluster Ray, el uso de recursos, y las tareas en ejecución.
    """)
    
    # Inicializar Ray si no está inicializado
    if not ray.is_initialized():
        try:
            with st.spinner("Inicializando Ray..."):
                ray.init(ignore_reinit_error=True)
            st.success("Ray inicializado correctamente")
        except Exception as e:
            st.error(f"Error al inicializar Ray: {str(e)}")
    
    # Dashboard de Ray
    st.subheader("Dashboard de Ray")
    if ray.is_initialized():
        try:
            dashboard_url = ray.get_dashboard_url()
            if dashboard_url:
                st.info(f"El dashboard de Ray está disponible en: {dashboard_url}")
                st.markdown(f"""
                Para acceder al dashboard, abra la URL en un navegador.
                El dashboard proporciona una interfaz gráfica completa para monitorizar el cluster.
                """)
            else:
                st.warning("El dashboard de Ray no está disponible en esta configuración.")
        except Exception as e:
            st.error(f"No se pudo obtener la URL del dashboard: {str(e)}")
    else:
        st.warning("Ray no está inicializado. No se puede obtener información del dashboard.")
    
    # Información del Cluster
    st.subheader("Información del Cluster Ray")
    if ray.is_initialized():
        try:
            # Usar nuestra nueva utilidad de monitorización
            try:
                from utils.monitoring import get_ray_status
                ray_status = get_ray_status()
                
                # Información del dashboard
                if ray_status.get("dashboard_url"):
                    st.info(f"El dashboard de Ray está disponible en: {ray_status['dashboard_url']}")
                
                # Crear DataFrame con la información de los nodos
                nodes_data = []
                for node in ray_status["nodes"]:
                    nodes_data.append({
                        "ID de Nodo": node["NodeID"][:8] + "...",
                        "Tipo": "Head" if node.get("RayletSocketName") else "Worker",
                        "Dirección": node["NodeManagerAddress"],
                        "CPUs": f"{node['Resources'].get('CPU', 0)}",
                        "GPUs": f"{node['Resources'].get('GPU', 0)}",
                        "Memoria (GB)": f"{node['Resources'].get('memory', 0) / (1024 * 1024 * 1024):.2f}",
                        "Estado": "Vivo" if node["Alive"] else "Muerto"
                    })
                
                if nodes_data:
                    nodes_df = pd.DataFrame(nodes_data)
                    st.dataframe(nodes_df, use_container_width=True)
                else:
                    st.info("No hay nodos disponibles en el cluster.")
                
                # Resumen de recursos
                st.subheader("Resumen de Recursos")
                
                # Crear gráficos para visualizar el uso de recursos
                fig = go.Figure()
                
                resources = ["CPU", "GPU", "memory", "object_store_memory"]
                resource_labels = ["CPUs", "GPUs", "Memoria", "Memoria de Objetos"]
                
                for i, resource in enumerate(resources):
                    if resource in ray_status["total_resources"]:
                        total = ray_status["total_resources"][resource]
                        used = ray_status["used_resources"].get(resource, 0)
                        available = ray_status["available_resources"].get(resource, 0)
                        
                        # Normalizar la memoria a GB si es necesario
                        if "memory" in resource:
                            total = total / (1024 * 1024 * 1024)
                            used = used / (1024 * 1024 * 1024)
                            available = available / (1024 * 1024 * 1024)
                            unit = "GB"
                        else:
                            unit = "Unidades"
                        
                        fig.add_trace(go.Bar(
                            name=resource_labels[i],
                            x=["Usado", "Disponible"],
                            y=[used, available],
                            text=[f"{used:.2f} {unit}", f"{available:.2f} {unit}"],
                            textposition="auto"
                        ))
                
                fig.update_layout(
                    title="Uso de Recursos en el Cluster",
                    xaxis_title="Estado",
                    yaxis_title="Cantidad",
                    barmode="group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except ImportError:
                # Fallback a la implementación anterior
                # Obtener información del cluster
                nodes_info = ray.nodes()
                
                # Crear DataFrame con la información de los nodos
                nodes_data = []
                for node in nodes_info:
                    nodes_data.append({
                        "ID de Nodo": node["NodeID"][:8] + "...",
                        "Tipo": "Head" if node.get("RayletSocketName") else "Worker",
                        "Dirección": node["NodeManagerAddress"],
                        "CPUs": f"{node['Resources'].get('CPU', 0)}",
                        "GPUs": f"{node['Resources'].get('GPU', 0)}",
                        "Memoria (GB)": f"{node['Resources'].get('memory', 0) / (1024 * 1024 * 1024):.2f}",
                        "Estado": "Vivo" if node["Alive"] else "Muerto"
                    })
                
                if nodes_data:
                    nodes_df = pd.DataFrame(nodes_data)
                    st.dataframe(nodes_df, use_container_width=True)
                else:
                    st.info("No hay nodos disponibles en el cluster.")
                
                # Resumen de recursos
                st.subheader("Resumen de Recursos")
                available_resources = ray.available_resources()
                total_resources = ray.cluster_resources()
                
                # Crear gráficos para visualizar el uso de recursos
                fig = go.Figure()
                
                resources = ["CPU", "GPU", "memory", "object_store_memory"]
                resource_labels = ["CPUs", "GPUs", "Memoria", "Memoria de Objetos"]
                
                for i, resource in enumerate(resources):
                    if resource in total_resources:
                        total = total_resources[resource]
                        available = available_resources.get(resource, 0)
                        used = total - available
                        
                        # Normalizar la memoria a GB si es necesario
                        if "memory" in resource:
                            total = total / (1024 * 1024 * 1024)
                            used = used / (1024 * 1024 * 1024)
                            unit = "GB"
                        else:
                            unit = "Unidades"
                        
                        fig.add_trace(go.Bar(
                            name=resource_labels[i],
                            x=["Usado", "Disponible"],
                            y=[used, available],
                            text=[f"{used:.2f} {unit}", f"{available:.2f} {unit}"],
                            textposition="auto"
                        ))
                
                fig.update_layout(
                    title="Uso de Recursos en el Cluster",
                    xaxis_title="Estado",
                    yaxis_title="Cantidad",
                    barmode="group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error al obtener información del cluster: {str(e)}")
    else:
        st.warning("Ray no está inicializado. No se puede obtener información del cluster.")
    
    # Monitorización del sistema local
    st.subheader("Recursos del Sistema Local")
    
    # Usar nuestra nueva utilidad de monitorización
    try:
        from utils.monitoring import get_system_resources
        system_resources = get_system_resources()
        
        # Uso de CPU
        cpu_info = system_resources["cpu"]
        st.metric("Uso de CPU", f"{cpu_info['percent']}%", f"{cpu_info['count']} cores")
        
        # Uso de memoria
        memory = system_resources["memory"]
        memory_used_gb = memory["used"] / (1024 * 1024 * 1024)
        memory_total_gb = memory["total"] / (1024 * 1024 * 1024)
        st.metric("Memoria Utilizada", f"{memory_used_gb:.2f} GB / {memory_total_gb:.2f} GB", 
                 f"{memory['percent']}%")
        
        # Mostrar un gráfico de memoria
        fig = go.Figure(go.Pie(
            labels=["Utilizada", "Disponible"],
            values=[memory["used"], memory["available"]],
            hole=.4,
            marker_colors=['#ff9999', '#66b3ff']
        ))
        fig.update_layout(title="Distribución de Memoria")
        st.plotly_chart(fig, use_container_width=True)
        
        # Uso de disco
        disk = system_resources["disk"]
        disk_used_gb = disk["used"] / (1024 * 1024 * 1024)
        disk_total_gb = disk["total"] / (1024 * 1024 * 1024)
        st.metric("Espacio en Disco Utilizado", f"{disk_used_gb:.2f} GB / {disk_total_gb:.2f} GB", 
                 f"{disk['percent']}%")
        
        # Mostrar un gráfico de disco
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Espacio en Disco"],
            y=[disk_used_gb],
            name="Utilizado",
            marker_color="#ff9999"
        ))
        fig.add_trace(go.Bar(
            x=["Espacio en Disco"],
            y=[disk_total_gb - disk_used_gb],
            name="Disponible",
            marker_color="#66b3ff"
        ))
        fig.update_layout(
            title="Uso de Disco",
            yaxis_title="GB",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Información de red
        network = system_resources["network"]
        net_sent_mb = network["bytes_sent"] / (1024 * 1024)
        net_recv_mb = network["bytes_recv"] / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datos Enviados", f"{net_sent_mb:.2f} MB")
        with col2:
            st.metric("Datos Recibidos", f"{net_recv_mb:.2f} MB")
    
    except Exception as e:
        # Fallback to the original code
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("Uso de CPU", f"{cpu_percent}%")
        
        # Uso de memoria
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 * 1024 * 1024)
        memory_total_gb = memory.total / (1024 * 1024 * 1024)
        st.metric("Memoria Utilizada", f"{memory_used_gb:.2f} GB / {memory_total_gb:.2f} GB")
        
        # Mostrar un gráfico de memoria
        fig = go.Figure(go.Pie(
            labels=["Utilizada", "Disponible"],
            values=[memory.used, memory.available],
            hole=.4,
            marker_colors=['#ff9999', '#66b3ff']
        ))
        fig.update_layout(title="Distribución de Memoria")
        st.plotly_chart(fig, use_container_width=True)
        
        # Uso de disco
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        st.metric("Espacio en Disco Utilizado", f"{disk_used_gb:.2f} GB / {disk_total_gb:.2f} GB")
        
        # Mostrar un gráfico de disco
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Espacio en Disco"],
            y=[disk_used_gb],
            name="Utilizado",
            marker_color="#ff9999"
        ))
        fig.add_trace(go.Bar(
            x=["Espacio en Disco"],
            y=[disk_total_gb - disk_used_gb],
            name="Disponible",
            marker_color="#66b3ff"
        ))
        fig.update_layout(
            title="Uso de Disco",
            yaxis_title="GB",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)


