import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="PreVida AI", page_icon="💙", layout="centered")

# Estilos CSS personalizados para que se vea profesional
st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    .stButton>button {width: 100%; background-color: #2c3e50; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #34495e; color: white;}
    .metric-card {background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;}
    h1 {color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. ENTRENAMIENTO DEL MODELO (EN TIEMPO REAL)
# ---------------------------------------------------------
@st.cache_resource # Esto hace que no se re-entrene cada vez que tocas un botón
def cargar_modelo():
    # A. Generar Datos Sintéticos (Mismo código del Notebook 1)
    np.random.seed(42)
    n = 1500
    data = {
        'edad': np.random.randint(60, 95, n),
        'patologias_cronicas': np.random.randint(0, 5, n),
        'acompanamiento_citas': np.random.randint(0, 10, n),
        'reclamo_medicinas': np.random.randint(0, 12, n),
        'solicitud_citas': np.random.randint(0, 8, n),
        'ayuda_compras': np.random.randint(0, 5, n),
        'red_apoyo_familiar': np.random.choice([0, 1], n, p=[0.4, 0.6])
    }
    df = pd.DataFrame(data)
    puntaje = (df['edad']-60)*1.5 + (df['patologias_cronicas']*10) + ((1-df['red_apoyo_familiar'])*25) + np.random.normal(0,5,n)
    df['necesita_cuidado'] = (puntaje > 60).astype(int)
    
    # B. Entrenar
    X = df.drop(['necesita_cuidado'], axis=1)
    y = df['necesita_cuidado']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)
    
    return modelo, scaler

modelo, scaler = cargar_modelo()

# ---------------------------------------------------------
# 3. INTERFAZ GRÁFICA (FRONTEND)
# ---------------------------------------------------------
st.title("💙 Sistema Inteligente PreVida")
st.markdown("### Predicción de Riesgo y Operacionalización del Servicio")
st.info("Esta herramienta utiliza IA para diagnosticar la necesidad de cuidado domiciliario y cotizar el servicio en tiempo real.")

with st.form("formulario_paciente"):
    st.subheader("📋 Datos del Paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.slider("Edad", 60, 100, 75)
        patologias = st.slider("Patologías Crónicas", 0, 5, 1)
        red_apoyo_opcion = st.selectbox("Red de Apoyo", ["Sí tiene apoyo", "No tiene apoyo (Vive solo)"])
        red_apoyo = 1 if red_apoyo_opcion == "Sí tiene apoyo" else 0
        
    with col2:
        acomp_citas = st.slider("Solicitud Acomp. Citas (Mes)", 0, 10, 2)
        medicinas = st.slider("Solicitud Reclamo Medicinas", 0, 10, 1)
        horas_servicio = st.slider("🕒 Horas de servicio requeridas", 4, 12, 8)
        
    # Variables ocultas (promedio)
    sol_citas = 2
    compras = 1
    
    submitted = st.form_submit_button("🧠 DIAGNOSTICAR Y COTIZAR")

# ---------------------------------------------------------
# 4. LÓGICA DE NEGOCIO Y RESULTADOS
# ---------------------------------------------------------
if submitted:
    # A. Predecir
    datos_input = pd.DataFrame([[edad, patologias, acomp_citas, medicinas, sol_citas, compras, red_apoyo]], 
                               columns=['edad', 'patologias_cronicas', 'acompanamiento_citas', 'reclamo_medicinas', 'solicitud_citas', 'ayuda_compras', 'red_apoyo_familiar'])
    
    datos_scaled = scaler.transform(datos_input)
    probabilidad = modelo.predict_proba(datos_scaled)[0][1]
    
    st.divider()
    
    if probabilidad < 0.65:
        st.success(f"✅ **PACIENTE INDEPENDIENTE** (Probabilidad de Riesgo: {probabilidad:.1%})")
        st.caption("No se requiere contratación inmediata. Se sugiere monitoreo preventivo.")
        
    else:
        st.error(f"🔴 **ALTO RIESGO DETECTADO** (Probabilidad: {probabilidad:.1%})")
        st.markdown("El sistema ha activado el protocolo de asignación de recursos.")
        
        # B. Asignación Inteligente (Matchmaking)
        if patologias >= 3:
            perfil = "Enfermera Jefe"
            especialidad = "Especialista Clínica (UCI/Crónicos)"
            tarifa_hora = 55000
            imagen = "👩‍⚕️"
        elif red_apoyo == 0:
            perfil = "Gerontólogo"
            especialidad = "Acompañamiento y Estimulación"
            tarifa_hora = 35000
            imagen = "👨‍🦳"
        else:
            perfil = "Auxiliar de Enfermería"
            especialidad = "Soporte Básico"
            tarifa_hora = 24000
            imagen = "💙"
            
        total_turno = tarifa_hora * horas_servicio
        
        # C. Mostrar Ticket de Cotización
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{imagen} Perfil Asignado</h3>
                <h2>{perfil}</h2>
                <p>{especialidad}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Cotización Turno</h3>
                <h2>${total_turno:,.0f} COP</h2>
                <p>Valor por hora: ${tarifa_hora:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.warning("⚠️ **Acción Requerida:** Se recomienda proceder con la contratación inmediata para mitigar el riesgo.")
        st.button("✅ CONTRATAR SERVICIO AHORA")