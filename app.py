import os
import pickle
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="AI Real Estate Valuation Matrix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sci-Fi Dark Theme CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@500;600;700&display=swap');

    /* Global Sci-Fi Dark Grid Background */
    .stApp {
        background-color: #050811 !important;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(0, 243, 255, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(188, 19, 254, 0.08) 0%, transparent 40%),
            linear-gradient(rgba(15, 23, 42, 0.6) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.6) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px;
        color: #e2e8f0 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    /* Holographic Sci-Fi Header Banner */
    .scifi-header {
        background: rgba(10, 16, 30, 0.85);
        border: 1px solid #00f3ff;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.25), inset 0 0 15px rgba(0, 243, 255, 0.1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .scifi-header::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f3ff, transparent);
        animation: scanline 3s linear infinite;
    }
    @keyframes scanline {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .scifi-header h1 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f3ff !important;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 3px;
        text-shadow: 0 0 12px rgba(0, 243, 255, 0.7);
        margin: 0;
    }
    .scifi-header p {
        color: #94a3b8 !important;
        font-size: 1.1rem;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Subheaders & HUD Titles */
    h2, h3, h4, h5, h6, .stSubheader {
        font-family: 'Orbitron', sans-serif !important;
        color: #38bdf8 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* Field Labels */
    label, .stWidgetLabel, div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    /* Sci-Fi Futuristic Inputs */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #0a1122 !important;
        border: 1px solid rgba(0, 243, 255, 0.35) !important;
        border-radius: 6px !important;
        color: #00f3ff !important;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: #00f3ff !important;
        box-shadow: 0 0 18px rgba(0, 243, 255, 0.5) !important;
    }
    div[data-baseweb="input"] input {
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Glowing Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0f766e 0%, #0284c7 50%, #7e22ce 100%) !important;
        color: #ffffff !important;
        border: 1px solid #00f3ff !important;
        padding: 0.85rem 1.5rem;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        border-radius: 8px !important;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.3) !important;
        transition: all 0.3s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.7) !important;
        border-color: #ffffff !important;
    }

    /* Neural Output Matrix Card */
    .scifi-result-card {
        background: rgba(8, 22, 40, 0.95);
        border: 2px solid #00ff66;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 0 30px rgba(0, 255, 102, 0.3), inset 0 0 15px rgba(0, 255, 102, 0.1);
        animation: pulseGlow 2s infinite alternate;
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 20px rgba(0, 255, 102, 0.2); }
        100% { box-shadow: 0 0 40px rgba(0, 255, 102, 0.5); }
    }
    .scifi-result-card h2 {
        color: #94a3b8 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.2rem;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .scifi-result-card h1 {
        color: #00ff66 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 102, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Model Loader
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'houes_price_LR.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"SYSTEM ERROR: Unable to load neural model artifact ({e})")
    st.stop()

# HUD Header
st.markdown("""
    <div class="scifi-header">
        <h1>⚡ QUANTUM REAL ESTATE MATRIX</h1>
        <p>[ NEURAL PREDICTION PROTOCOL v2.0 ] — Enter telemetry parameters below</p>
    </div>
""", unsafe_allow_html=True)

# Form Controls
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛸 Structural Core")
        bedrooms = st.number_input("Bedrooms Count", min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Bathrooms Count", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.number_input("Floors Index", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        condition = st.slider("Structural Condition Rating (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("Construction Grade Rating (1-13)", min_value=1, max_value=13, value=7)

    with col2:
        st.subheader("📡 Spatial Metrics")
        living_area = st.number_input("Living Area (sq ft)", min_value=100, max_value=20000, value=2000)
        lot_area = st.number_input("Lot Sector Area (sq ft)", min_value=100, max_value=100000, value=5000)
        above_area = st.number_input("Ground Area (excl. basement)", min_value=100, max_value=20000, value=1500)
        basement_area = st.number_input("Sub-Level Basement Area", min_value=0, max_value=10000, value=500)
        lot_area_renov = st.number_input("Renovated Sector Area", min_value=100, max_value=100000, value=5000)

    with col3:
        st.subheader("🌐 Telemetry & Sector")
        waterfront = st.selectbox("Waterfront Perimeter", options=[0, 1], format_func=lambda x: "DETECTED" if x == 1 else "NONE")
        views = st.slider("Panoramic View Index (0-4)", min_value=0, max_value=4, value=0)
        built_year = st.number_input("Construction Epoch (Year)", min_value=1800, max_value=2026, value=2000)
        renov_year = st.number_input("Retrofit Epoch (Year, 0=None)", min_value=0, max_value=2026, value=0)
        schools = st.number_input("Educational Hubs Nearby", min_value=0, max_value=10, value=3)
        airport_dist = st.number_input("Aero-Hub Distance (km)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

    st.markdown("---")
    submit_button = st.form_submit_button(label="🚀 EXECUTE NEURAL VALUATION MATRIX")

if submit_button:
    features = np.array([[
        bedrooms, bathrooms, living_area, lot_area, floors,
        waterfront, views, condition, grade, above_area,
        basement_area, built_year, renov_year, lot_area_renov,
        schools, airport_dist
    ]])

    try:
        prediction = model.predict(features)[0]
        price_output = f"₹{max(0, prediction):,.2f}"
        
        st.markdown(f"""
            <div class="scifi-result-card">
                <h2>// QUANTUM VALUATION RESULT</h2>
                <h1>{price_output}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Matrix Failure: {e}")
