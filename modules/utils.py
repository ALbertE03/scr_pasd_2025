import streamlit as st
from datetime import datetime

def get_unique_key(base_key):
    """Genera una key única incrementando un contador"""
    if 'widget_counter' not in st.session_state:
        st.session_state.widget_counter = 0
    st.session_state.widget_counter += 1
    return f"{base_key}_{st.session_state.widget_counter}"

def initialize_session_state():
    """Inicializa todas las variables de session_state necesarias"""
    
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'sequential_results' not in st.session_state:
        st.session_state.sequential_results = None
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = 'iris'
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
    

    if 'test_size' not in st.session_state:
        st.session_state.test_size = 0.3
    if 'enable_fault_tolerance' not in st.session_state:
        st.session_state.enable_fault_tolerance = True

    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    if 'fault_config' not in st.session_state:
        st.session_state.fault_config = {
            'max_retries': 3,
            'timeout_seconds': 300,
            'enable_reconstruction': True,
            'enable_auto_retry': True,
            'monitoring_interval': 30
        }

    if 'fault_logs' not in st.session_state:
        st.session_state.fault_logs = []
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    

    if 'widget_counter' not in st.session_state:
        st.session_state.widget_counter = 0

    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

def load_custom_styles():
    """Carga los estilos CSS personalizados para la aplicación"""
    st.markdown("""
    <style>
        /* Estilos para modo oscuro y claro con detección automática */
        
        /* Estilos globales - Modo oscuro */
        .stApp {
            background: linear-gradient(135deg, #1e1e2e 0%, #171728 100%);
        }
        
        /* Contenedor principal con fondo - Compatible con modo oscuro */
        .main .block-container {
            background: rgba(30, 30, 46, 0.7);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        /* Tarjetas con gradientes y efectos sutiles - Modo oscuro */
        .metric-card {
            background: linear-gradient(120deg, #292a3e 0%, #1e1e2e 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border-left: 4px solid #bd93f9;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            color: #cdd6f4;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .success-card {
            background: linear-gradient(120deg, #20303b 0%, #264531 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #a6e3a1;
            border-left: 4px solid #a6e3a1;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .warning-card {
            background: linear-gradient(120deg, #302a22 0%, #3d2e1b 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #f9e2af;
            border-left: 4px solid #f9e2af;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .error-card {
            background: linear-gradient(120deg, #362231 0%, #3f1d1d 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #f38ba8;
            border-left: 4px solid #f38ba8;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        .info-card {
            background: linear-gradient(120deg, #1e293b 0%, #1a2332 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: #89b4fa;
            border-left: 4px solid #89b4fa;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        
        /* Estilo para los tabs - Modo oscuro */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-radius: 12px;
            padding: 0.6rem;
            background: rgba(30, 30, 46, 0.7);
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(189, 147, 249, 0.1);
            color: #cdd6f4;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(189, 147, 249, 0.15);
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            color: #1e1e2e;
            box-shadow: 0 3px 10px rgba(189, 147, 249, 0.4);
            transform: translateY(-2px);
            font-weight: 600;
        }
        
        /* Animaciones sutiles */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(189, 147, 249, 0.3); }
            70% { box-shadow: 0 0 0 10px rgba(189, 147, 249, 0); }
            100% { box-shadow: 0 0 0 0 rgba(189, 147, 249, 0); }
        }
        .dashboard-container {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Dashboard Header Mejorado - Modo oscuro */
        .dashboard-header {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: #1e1e2e;
            box-shadow: 0 8px 20px rgba(189, 147, 249, 0.3);
            text-align: center;
            animation: fadeIn 0.8s ease-out;
        }
        .dashboard-header h1 {
            font-weight: 700;
            letter-spacing: 1px;
            color: #1e1e2e;
        }
        .dashboard-header h3 {
            opacity: 0.9;
            font-weight: 500;
            color: #2d2d44;
        }
        
        /* Estilos para gráficos y visualizaciones - Modo oscuro */
        .plotly-chart-container {
            background: rgba(30, 30, 46, 0.7);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            border: 1px solid rgba(189, 147, 249, 0.1);
        }
        .plotly-chart-container:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.25);
            transform: translateY(-2px);
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        
        /* Mejorar métricos - Modo oscuro */
        [data-testid="stMetric"] {
            background: linear-gradient(120deg, #292a3e 0%, #1e1e2e 100%);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #cdd6f4;
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #f8f8f2;
            font-size: 1.8rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 500;
        }
        
        /* Etiquetas y contenido - Modo oscuro */
        h1, h2, h3 {
            color: #f8f8f2;
        }
        h1 {
            font-weight: 700;
        }
        h2 {
            font-weight: 600;
        }
        h3 {
            font-weight: 500;
        }
        
        /* Estilo para los selectboxes y demás widgets - Modo oscuro */
        .stSelectbox [data-baseweb="select"] {
            background-color: #292a3e;
            border-radius: 8px;
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        .stMultiSelect [data-baseweb="select"] {
            background-color: #292a3e;
            border-radius: 8px;
            border: 1px solid rgba(189, 147, 249, 0.2);
        }
        
        /* Checkbox y Radio - Modo oscuro */
        .stCheckbox [data-baseweb="checkbox"] {
            padding: 0.5rem;
            border-radius: 6px;
            transition: background 0.2s ease;
        }
        .stCheckbox [data-baseweb="checkbox"]:hover {
            background: rgba(189, 147, 249, 0.1);
        }
        .stRadio [data-baseweb="radio"] {
            padding: 0.5rem;
            border-radius: 6px;
            transition: background 0.2s ease;
        }
        .stRadio [data-baseweb="radio"]:hover {
            background: rgba(189, 147, 249, 0.1);
        }
        
        /* Sliders - Modo oscuro */
        .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background: #bd93f9;
            border: 2px solid #1e1e2e;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Sidebar - Modo oscuro */
        [data-testid="stSidebar"] {
            background-color: rgba(25, 25, 35, 0.8);
            border-right: 1px solid rgba(189, 147, 249, 0.1);
        }
        
        /* Botones - Modo oscuro */
        .stButton > button {
            border-radius: 8px;
            padding: 0.3rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stButton > [data-baseweb="button"][kind="primary"] {
            background: linear-gradient(135deg, #bd93f9 0%, #9d79f6 100%);
            color: #1e1e2e;
        }
        
        /* Data tables - Modo oscuro */
        [data-testid="stTable"] {
            background-color: #292a3e;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(189, 147, 249, 0.1);
        }
        [data-testid="stTable"] table {
            border-collapse: separate;
            border-spacing: 0;
        }
        [data-testid="stTable"] thead th {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-weight: 600;
            border-bottom: 1px solid rgba(189, 147, 249, 0.3);
        }
        [data-testid="stTable"] tbody tr:nth-child(even) {
            background-color: rgba(189, 147, 249, 0.05);
        }
        [data-testid="stTable"] tbody tr:hover {
            background-color: rgba(189, 147, 249, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
