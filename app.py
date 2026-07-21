import os
import pickle
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with fixed contrast
st.markdown("""
    <style>
    /* Force main background and text colors */
    .stApp {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Ensure all field labels are dark and visible */
    label, .stWidgetLabel, div[data-testid="stMarkdownContainer"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Subheader text visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }

    /* Input box styling */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="input"] input {
        color: #0f172a !important;
    }

    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #e2e8f0 !important;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    /* Custom Submit Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }

    /* Result Metric Card */
    .result-card {
        background: linear-gradient(to right, #38ef7d, #11998e);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-card h2, .result-card h1 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model safely
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Matches exact filename from model artifact
    model_path = os.path.join(BASE_DIR, 'houes_price_LR.pkl')[cite: 1]
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏠 Smart House Price Estimator</h1>
        <p>Enter property specifications below to calculate an estimated market valuation.</p>
    </div>
""", unsafe_allow_html=True)

# Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛏️ Structural Info")
        bedrooms = st.number_input("Number of bedrooms", min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Number of bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.number_input("Number of floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        condition = st.slider("Condition of the house (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("Grade of the house (1-13)", min_value=1, max_value=13, value=7)

    with col2:
        st.subheader("📐 Area & Size")
        living_area = st.number_input("Living area (sq ft)", min_value=100, max_value=20000, value=2000)
        lot_area = st.number_input("Lot area (sq ft)", min_value=100, max_value=100000, value=5000)
        above_area = st.number_input("Area of house (excl. basement)", min_value=100, max_value=20000, value=1500)
        basement_area = st.number_input("Area of basement", min_value=0, max_value=10000, value=500)
        lot_area_renov = st.number_input("Lot area (renovated)", min_value=100, max_value=100000, value=5000)

    with col3:
        st.subheader("📍 Location & History")
        waterfront = st.selectbox("Waterfront present?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        views = st.slider("Number of views (0-4)", min_value=0, max_value=4, value=0)
        built_year = st.number_input("Built Year", min_value=1800, max_value=2026, value=2000)
        renov_year = st.number_input("Renovation Year (0 if never)", min_value=0, max_value=2026, value=0)
        schools = st.number_input("Number of schools nearby", min_value=0, max_value=10, value=3)
        airport_dist = st.number_input("Distance from airport (km)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

    st.markdown("---")
    submit_button = st.form_submit_button(label="💸 Calculate Estimated Price")

if submit_button:
    features = np.array([[
        bedrooms, bathrooms, living_area, lot_area, floors,
        waterfront, views, condition, grade, above_area,
        basement_area, built_year, renov_year, lot_area_renov,
        schools, airport_dist
    ]])

    try:
        prediction = model.predict(features)[0]
        price_output = f"${max(0, prediction):,.2f}"
        
        st.markdown(f"""
            <div class="result-card">
                <h2>Estimated Property Value</h2>
                <h1 style="font-size: 2.8rem; margin: 0;">{price_output}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
