# Script para ejecutar Streamlit standalone
Write-Host "🎨 Iniciando aplicación Streamlit..." -ForegroundColor Green

# Verificar si Streamlit está instalado
try {
    python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
} catch {
    Write-Host "❌ Streamlit no está instalado. Instalando dependencias..." -ForegroundColor Red
    pip install -r requirements.txt
}

# Ejecutar Streamlit
Write-Host "🌐 Abriendo aplicación web en http://localhost:8501" -ForegroundColor Yellow
streamlit run streamlit_app.py

Write-Host "✅ Aplicación finalizada!" -ForegroundColor Green
