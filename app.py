import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling: Apple + Stripe + Vercel Glassmorphism UI
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Theme Background */
    .stApp {
        background-color: #0B1220 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
        color: #F9FAFB !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hero Banner Section */
    .hero-container {
        text-align: center;
        padding: 3.5rem 1.5rem 2.5rem 1.5rem;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        color: #60A5FA;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
    }

    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.15rem;
        max-width: 650px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* Bento Cards Layout */
    .bento-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }

    /* Stat Badges */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38BDF8;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Input & Control Adjustments */
    label, .stWidgetLabel, div[data-testid="stMarkdownContainer"] p {
        color: #E5E7EB !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F9FAFB !important;
        transition: all 0.2s ease-in-out;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* Gradient Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.85rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
    }

    /* Glassmorphism Results Card */
    .result-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
        margin-top: 1rem;
    }

    .result-title {
        color: #A7F3D0;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-price {
        font-size: 3.5rem;
        font-weight: 800;
        color: #10B981;
        margin: 0.5rem 0;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Trained Machine Learning Model
# ---------------------------------------------------------
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
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <span class="badge">✨ HOUSE PRICE PREDICTIVE ANALYTICS</span>
        <h1 class="hero-title">Predict House Prices</h1>
        <p class="hero-subtitle">Get accurate property valuations using machine learning and real estate market insights.</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Dashboard Metrics (Bento Row)
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
        <div class="bento-card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value">Linear Reg</div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
        <div class="bento-card">
            <div class="metric-label">Features Analyzed</div>
            <div class="metric-value">16 Variables</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
        <div class="bento-card">
            <div class="metric-label">Response Time</div>
            <div class="metric-value">&lt; 10ms</div>
        </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
        <div class="bento-card">
            <div class="metric-label">Status</div>
            <div class="metric-value" style="color: #10B981;">● Active</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Property Details Input Form
# ---------------------------------------------------------
st.markdown("### 📋 Property Specifications")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🛏️ Structural Details")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        condition = st.slider("House Condition (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("Construction Grade (1-13)", min_value=1, max_value=13, value=7)

    with col2:
        st.markdown("#### 📐 Spatial Area")
        living_area = st.number_input("Living Area (sq ft)", min_value=100, max_value=20000, value=2000)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=100, max_value=100000, value=5000)
        above_area = st.number_input("Above Basement Area (sq ft)", min_value=100, max_value=20000, value=1500)
        basement_area = st.number_input("Basement Area (sq ft)", min_value=0, max_value=10000, value=500)
        lot_area_renov = st.number_input("Renovated Lot Area (sq ft)", min_value=100, max_value=100000, value=5000)

    with col3:
        st.markdown("#### 📍 Location & History")
        waterfront = st.selectbox("Waterfront View", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        views = st.slider("Panoramic Views (0-4)", min_value=0, max_value=4, value=0)
        built_year = st.number_input("Built Year", min_value=1800, max_value=2026, value=2000)
        renov_year = st.number_input("Renovation Year (0 if none)", min_value=0, max_value=2026, value=0)
        schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3)
        airport_dist = st.number_input("Airport Distance (km)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="✨ Predict Property Value ✨")

# ---------------------------------------------------------
# Prediction Execution & Results Output
# ---------------------------------------------------------
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
            <div class="result-glass-card">
                <div class="result-title">Estimated Market Valuation</div>
                <div class="result-price">{price_output}</div>
                <p style="color: #9CA3AF; font-size: 0.9rem; margin-top: 0.5rem;">Calculated using realtime market regression analytics.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Market Insights & Feature Impact")
        feature_names = [
            'Bedrooms', 'Bathrooms', 'Living Area', 'Lot Area', 'Floors',
            'Waterfront', 'Views', 'Condition', 'Grade', 'Above Area',
            'Basement Area', 'Built Year', 'Renov Year', 'Renov Lot Area',
            'Schools', 'Airport Dist'
        ]
        
        if hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact Factor': model.coef_
            }).sort_values(by='Impact Factor', key=abs, ascending=False)

            st.bar_chart(coef_df.set_index('Feature'))
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
