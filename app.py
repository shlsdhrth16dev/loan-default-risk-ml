import streamlit as st
import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path to import src modules
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features

# Page Configuration
st.set_page_config(
    page_title="Loan Hazard | Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0E1117;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #262730;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #0E1117;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    model_path = "reports/trained_model.pkl"
    columns_path = "reports/training_columns.pkl"
    
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run training first.")
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    training_columns = None
    if os.path.exists(columns_path):
        with open(columns_path, "rb") as f:
            training_columns = pickle.load(f)
            
    return model, training_columns

model, training_columns = load_assets()

# --- Header ---
st.title("🛡️ Loan Default Risk Artificial Intelligence")
st.markdown("### Explainable Credit Risk Assessment System")
st.markdown("---")

# --- Layout ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("#### 👤 Applicant Profile")
    with st.container():
        # Personal Info
        c1, c2 = st.columns(2)
        person_age = c1.number_input("Age", min_value=18, max_value=100, value=25)
        person_income = c2.number_input("Annual Income ($)", min_value=0, value=55000, step=1000)
        
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        
        person_emp_length = st.number_input(
            "Employment Length (Years)", 
            min_value=0.0, max_value=60.0, value=2.0, step=0.5
        )

    st.markdown("#### 💰 Loan Details")
    with st.container():
        c3, c4 = st.columns(2)
        loan_amnt = c3.number_input("Loan Amount ($)", min_value=1000, value=10000, step=500)
        loan_intent = c4.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        
        c5, c6 = st.columns(2)
        loan_grade = c5.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = c6.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=11.5, step=0.1)
        
        loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
        st.info(f"Loan to Income Ratio: **{loan_percent_income:.2%}**")

    st.markdown("#### 📜 Credit History")
    with st.container():
        c7, c8 = st.columns(2)
        cb_person_default_on_file = c7.radio("Historical Default?", ["N", "Y"], horizontal=True)
        cb_person_cred_hist_length = c8.number_input("Credit History (Years)", min_value=0, max_value=50, value=4)

    predict_btn = st.button("Analyze Risk Profile")

with col2:
    if predict_btn and model:
        # Prepare Data
        input_data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }
        
        df = pd.DataFrame([input_data])
        
        # Preprocessing
        with st.spinner("Processing features & Running inference..."):
            try:
                df_processed = build_features(df)
                
                # Align columns
                if training_columns:
                    df_processed = df_processed.reindex(columns=training_columns, fill_value=0)
                
                # Predict
                prob = model.predict_proba(df_processed)[0][1]
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.stop()

        # --- Results Display ---
        st.markdown("#### 📊 Risk Analysis Report")
        
        # Determine Card Color based on risk
        risk_label = "HIGH RISK" if prob > 0.6 else "LOW RISK"
        risk_color = "#FF4B4B" if prob > 0.6 else "#09AB3B"  # Red or Green
        
        # Metric Card using HTML
        st.markdown(f"""
        <div style="background-color: {risk_color}1a; border: 2px solid {risk_color}; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <h2 style="color: {risk_color}; margin:0; text-align: center;">{risk_label}</h2>
            <h1 style="color: #0E1117; margin:0; text-align: center; font-size: 3.5rem;">{prob:.1%}</h1>
            <p style="text-align: center; margin:0; color: #555;">Probability of Default</p>
        </div>
        """, unsafe_allow_html=True)

        # Visualization
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Horizontal bar chart for better fit
        categories = ['No Default', 'Default']
        probabilities = [1-prob, prob]
        colors = ['#09AB3B', '#FF4B4B']
        
        y_pos = np.arange(len(categories))
        
        ax.barh(y_pos, probabilities, align='center', color=colors, alpha=0.9, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        
        # Add values inside bars
        for i, v in enumerate(probabilities):
            ax.text(v + 0.02, i, f"{v:.1%}", va='center', fontweight='bold', color='#333')

        fig.patch.set_alpha(0) # Transparent figure background
        ax.patch.set_alpha(0)
        
        st.pyplot(fig)
        
        # Explanation Text
        st.markdown("### 💡 Assessment Summary")
        if prob > 0.6:
            st.warning("""
                **Caution:** This applicant shows characteristics strongly associated with loan default. 
                \nReview credit history length and debt-to-income ratio carefully before proceeding.
            """)
        else:
            st.success("""
                **Good Standing:** The applicant's profile suggests a high likelihood of repayment.
                \nDefault probability is within acceptable limits.
            """)

    elif not model:
        st.warning("⚠️ Model not loaded. Please ensure training artifacts exist.")
    else:
        # Initial State
        st.info("👈 Enter applicant details in the left panel and click **Analyze Risk Profile**.")
